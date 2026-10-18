
# ============================================================
# NEWSLENS: ONTONOTES 5.0 NER MODEL TRAINING
# Model: DistilBERT Base Cased
# Tasks: Train, validate, evaluate, save model and tokenizer
# ============================================================

import os
import random
import json
import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm.auto import tqdm

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

MODEL_NAME = "distilbert/distilbert-base-cased"
OUTPUT_DIR = "./newslens_ner_model"

MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 2
EPOCHS = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
SEED = 42

# Requires the dataset and id2label from our earlier cells.
assert "dataset" in globals(), "Load the OntoNotes dataset first."
assert "id2label" in globals(), "Define the id2label mapping first."

label2id = {label: idx for idx, label in id2label.items()}
NUM_LABELS = len(id2label)

assert set(id2label) == set(range(NUM_LABELS))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )

# ------------------------------------------------------------
# 2. REPAIR MALFORMED BIO LABEL TRANSITIONS
# Original dataset is never modified.
# ------------------------------------------------------------

def repair_bio_tags(tags):
    repaired = []
    active_type = None

    for tag_id in tags:
        label = id2label[int(tag_id)]

        if label == "O":
            repaired.append(0)
            active_type = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            repaired.append(int(tag_id))
            active_type = entity_type

        elif prefix == "I":
            if active_type != entity_type:
                # Recover an orphan/mismatched I-tag.
                repaired.append(label2id[f"B-{entity_type}"])
            else:
                repaired.append(int(tag_id))

            active_type = entity_type

    return repaired


# ------------------------------------------------------------
# 3. PREPARE EXAMPLES
# ------------------------------------------------------------

class NERDataset(Dataset):
    def __init__(self, split):
        self.examples = []

        for row in tqdm(dataset[split], desc=f"Preparing {split}"):
            tokens = row["tokens"]
            tags = repair_bio_tags(row["tags"])

            if len(tokens) != len(tags):
                raise ValueError("Token/tag length mismatch.")

            self.examples.append({
                "tokens": tokens,
                "tags": tags,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


train_data = NERDataset("train")
val_data = NERDataset("validation")
test_data = NERDataset("test")

# ------------------------------------------------------------
# 4. TOKENIZER AND BATCH COLLATOR
# First subword receives the word label.
# Special tokens and remaining subwords use -100.
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    word_lists = [item["tokens"] for item in batch]

    encoded = tokenizer(
        word_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    aligned_labels = []
    batch_word_ids = []

    for i, item in enumerate(batch):
        word_ids = encoded.word_ids(batch_index=i)
        word_labels = item["tags"]

        labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word_id:
                labels.append(word_labels[word_id])
            else:
                labels.append(-100)

            previous_word_id = word_id

        aligned_labels.append(labels)
        batch_word_ids.append(word_ids)

    encoded["labels"] = torch.tensor(
        aligned_labels, dtype=torch.long
    )

    return {
        "model_inputs": {
            key: value
            for key, value in encoded.items()
            if key in ["input_ids", "attention_mask"]
        },
        "labels": encoded["labels"],
        "word_ids": batch_word_ids,
        "gold_tags": [item["tags"] for item in batch],
    }


train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
)

# ------------------------------------------------------------
# 5. INITIALIZE MODEL
# ------------------------------------------------------------

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)

model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

updates_per_epoch = (
    len(train_loader) + GRAD_ACCUMULATION_STEPS - 1
) // GRAD_ACCUMULATION_STEPS

total_steps = updates_per_epoch * EPOCHS
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ------------------------------------------------------------
# 6. EXACT ENTITY-SPAN METRICS
# ------------------------------------------------------------

def normalize_predicted_tags(tag_ids):
    return [
        id2label[tag]
        for tag in repair_bio_tags(tag_ids)
    ]


def extract_spans(labels):
    """Return (start, end-exclusive, entity_type) spans."""
    spans = []
    start = None
    current_type = None

    for i, label in enumerate(labels):
        if label == "O":
            if start is not None:
                spans.append((start, i, current_type))
            start = None
            current_type = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B" or current_type != entity_type:
            if start is not None:
                spans.append((start, i, current_type))

            start = i
            current_type = entity_type

    if start is not None:
        spans.append((start, len(labels), current_type))

    return set(spans)


def calculate_metrics(gold_sequences, pred_sequences):
    counts = defaultdict(lambda: [0, 0, 0])

    for gold, pred in zip(gold_sequences, pred_sequences):
        gold_spans = extract_spans(gold)
        pred_spans = extract_spans(pred)

        for entity_type in set(
            [span[2] for span in gold_spans]
            + [span[2] for span in pred_spans]
        ):
            gold_type = {
                span for span in gold_spans
                if span[2] == entity_type
            }
            pred_type = {
                span for span in pred_spans
                if span[2] == entity_type
            }

            counts[entity_type][0] += len(
                gold_type & pred_type
            )
            counts[entity_type][1] += len(
                pred_type - gold_type
            )
            counts[entity_type][2] += len(
                gold_type - pred_type
            )

    total_tp = sum(v[0] for v in counts.values())
    total_fp = sum(v[1] for v in counts.values())
    total_fn = sum(v[2] for v in counts.values())

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    per_type = {}

    for entity_type, (tp, fp, fn) in sorted(counts.items()):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        score = 2 * p * r / (p + r) if p + r else 0.0

        per_type[entity_type] = {
            "precision": p,
            "recall": r,
            "f1": score,
            "support": tp + fn,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_type": per_type,
    }


# ------------------------------------------------------------
# 7. EVALUATION
# Predictions are converted back to original word-level tags.
# ------------------------------------------------------------

@torch.no_grad()
def evaluate(loader, description="Evaluating"):
    model.eval()

    gold_sequences = []
    pred_sequences = []
    total_loss = 0.0
    batches = 0

    for batch in tqdm(loader, desc=description, leave=False):
        inputs = {
            key: value.to(device)
            for key, value in batch["model_inputs"].items()
        }
        labels = batch["labels"].to(device)

        outputs = model(**inputs, labels=labels)
        total_loss += outputs.loss.item()
        batches += 1

        predicted_ids = outputs.logits.argmax(dim=-1).cpu()

        for i, word_ids in enumerate(batch["word_ids"]):
            seen_words = set()
            pred_word_tags = []
            gold_word_tags = []

            for token_pos, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen_words:
                    continue

                seen_words.add(word_id)

                pred_word_tags.append(
                    int(predicted_ids[i, token_pos].item())
                )
                gold_word_tags.append(
                    batch["gold_tags"][i][word_id]
                )

            pred_labels = normalize_predicted_tags(
                pred_word_tags
            )
            gold_labels = [
                id2label[tag] for tag in gold_word_tags
            ]

            gold_sequences.append(gold_labels)
            pred_sequences.append(pred_labels)

    metrics = calculate_metrics(
        gold_sequences, pred_sequences
    )
    metrics["loss"] = total_loss / max(batches, 1)

    return metrics


def print_metrics(metrics, title):
    print(f"\n{title}")
    print("-" * 65)
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Micro F1:  {metrics['f1']:.4f}")

    print("\nPer-entity results:")
    print(
        f"{'Entity':<15}{'Precision':>12}"
        f"{'Recall':>12}{'F1':>10}{'Support':>10}"
    )

    for entity_type, result in metrics["per_type"].items():
        print(
            f"{entity_type:<15}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>10.4f}"
            f"{result['support']:>10}"
        )


# ------------------------------------------------------------
# 8. TRAINING LOOP
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

best_val_f1 = -1.0
history = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    optimizer_updates = 0

    progress = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for step, batch in progress:
        inputs = {
            key: value.to(device)
            for key, value in batch["model_inputs"].items()
        }
        labels = batch["labels"].to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        running_loss += loss.item()

        (loss / GRAD_ACCUMULATION_STEPS).backward()

        should_update = (
            (step + 1) % GRAD_ACCUMULATION_STEPS == 0
            or (step + 1) == len(train_loader)
        )

        if should_update:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_updates += 1

        progress.set_postfix(
            loss=f"{running_loss / (step + 1):.4f}"
        )

    train_loss = running_loss / len(train_loader)

    val_metrics = evaluate(
        val_loader,
        description="Validation",
    )

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train loss: {train_loss:.4f}")
    print_metrics(val_metrics, "Validation results")

    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_metrics["loss"],
        "validation_precision": val_metrics["precision"],
        "validation_recall": val_metrics["recall"],
        "validation_f1": val_metrics["f1"],
    }
    history.append(epoch_record)

    # Save the checkpoint with the best validation F1.
    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        with open(
            os.path.join(OUTPUT_DIR, "training_history.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(history, f, indent=2)

        print(f"Saved best checkpoint: F1={best_val_f1:.4f}")

# ------------------------------------------------------------
# 9. FINAL TEST EVALUATION
# Only after selecting the best validation checkpoint.
# ------------------------------------------------------------

print("\nLoading best validation checkpoint...")

model = AutoModelForTokenClassification.from_pretrained(
    OUTPUT_DIR
).to(device)

test_metrics = evaluate(
    test_loader,
    description="Final test evaluation",
)

print_metrics(test_metrics, "FINAL TEST RESULTS")

# Save evaluation results.
with open(
    os.path.join(OUTPUT_DIR, "test_metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(test_metrics, f, indent=2)

print("\nTraining complete.")
print("Saved model and tokenizer to:", OUTPUT_DIR)
print("Best validation F1:", round(best_val_f1, 4))
print("Final test F1:", round(test_metrics["f1"], 4))
