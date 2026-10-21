# NewsLens — News Entity Intelligence

**Extract. Understand. Explore.**

NewsLens is an NLP-powered news analysis application that identifies named entities in news articles and transforms unstructured text into structured, searchable information. Built using a fine-tuned DistilBERT model, it extracts people, organizations, locations, dates, and other entity types, with interactive visualizations for exploring entity mentions and their co-occurrences.

The project combines **Named Entity Recognition (NER), contextual text processing, entity exploration, and relationship visualization** in a single Streamlit application.

---

## Overview

News articles contain valuable information about people, organizations, locations, events, and other entities. Finding these entities manually across multiple articles can be time-consuming.

NewsLens automates this process by identifying and classifying named entities, highlighting them in the original article, and organizing the results into an interactive analysis workspace.

## Features

### 1. Smart Entity Extraction
- Analyze news articles by pasting text or uploading `.txt` and `.md` files.
- Identify and classify entities using a fine-tuned Transformer model.
- Highlight detected entities directly in the original article.
- Display entity categories and model confidence scores.
- Filter results by entity type.
- Export extracted entities as CSV or JSON.

### 2. Entity Atlas
- Search entity mentions across analyzed articles.
- Filter results by entity type and article.
- Explore mention counts and frequently occurring entity names.
- Inspect which articles contain a particular entity.

### 3. Entity Relationship Explorer
- Visualize entities that appear together in the same sentence.
- Explore co-occurrence patterns through an interactive network graph.
- Inspect supporting sentences for individual connections.
- Identify recurring connections across analyzed articles.

*Note: Co-occurrence indicates that entities appear together in a sentence; it does not establish a real-world relationship or causation.*

### 4. Model Evaluation Dashboard
- View test precision, recall, and micro-F1.
- Explore per-entity-type F1 scores.
- Review training and validation loss.
- Examine model configuration and evaluation limitations.

### 5. Editorial Dashboard
- Custom blush-pink and rose visual theme.
- Interactive analysis panels and entity filters.
- Session-level article statistics.
- Exportable entity records and analyzed article data.

---

## Model and Dataset

### Model

NewsLens uses **DistilBERT**, fine-tuned for token classification and Named Entity Recognition.

| Parameter | Configuration |
|---|---|
| Pretrained checkpoint | `distilbert/distilbert-base-cased` |
| Task | Named Entity Recognition |
| Architecture | Transformer encoder |
| Number of entity types | 18 |
| Number of BIO labels | 37, including `O` |
| Maximum training sequence length | 256 tokens |
| Optimizer | AdamW |
| Learning rate | `3e-5` |
| Training epochs | 3 |
| Batch size | 8 |
| Gradient accumulation | 2 steps |
| Weight decay | `0.01` |
| Learning-rate schedule | Linear |
| Random seed | 42 |

### Dataset

The model was trained and evaluated using **OntoNotes 5.0**, accessed through the Hugging Face dataset repository [`tner/ontonotes5`](https://huggingface.co/datasets/tner/ontonotes5).

The dataset contains annotated text from multiple genres, including news. It is not exclusively a contemporary news corpus.

| Split | Sequences |
|---|---:|
| Training | 59,924 |
| Validation | 8,528 |
| Test | 8,262 |
| **Total** | **76,714** |

### Supported Entity Types

NewsLens recognizes 18 entity types:

| Category | Entity labels |
|---|---|
| People | `PERSON` |
| Organizations | `ORG` |
| Locations | `GPE`, `LOC`, `FAC` |
| Dates and times | `DATE`, `TIME` |
| Numbers and values | `CARDINAL`, `ORDINAL`, `MONEY`, `PERCENT`, `QUANTITY` |
| Other entities | `NORP`, `EVENT`, `LAW`, `LANGUAGE`, `PRODUCT`, `WORK_OF_ART` |

The model retains the original entity labels internally, while the interface groups related labels into more accessible categories.

---

## Model Performance

The fine-tuned model achieved the following results on the OntoNotes 5.0 test split.

| Metric | Score |
|---|---:|
| Micro precision | 86.95% |
| Micro recall | 89.17% |
| **Micro-F1** | **88.04%** |
| Test loss | 0.0786 |

### Selected Per-Class F1 Scores

| Entity type | F1 score |
|---|---:|
| `GPE` | 94.29% |
| `NORP` | 93.50% |
| `PERSON` | 92.83% |
| `PERCENT` | 89.96% |
| `ORG` | 86.75% |
| `DATE` | 86.52% |
| `LAW` | 52.50% |

Performance varies across entity types, partly because some categories have fewer training and test examples.

**Evaluation note:** The reported result is micro-F1, not accuracy. It was measured on the OntoNotes test split using exact entity-span matching and the project's BIO-tag repair policy. Contemporary news articles may produce different results, and the model has not yet been established as superior to alternative NER systems.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and NLP processing |
| PyTorch | Model training and inference |
| Hugging Face Transformers | Pretrained model and tokenizer |
| Hugging Face Datasets | Dataset loading |
| Pandas | Structured entity data and exports |
| Streamlit | Interactive web application |
| Plotly | Interactive visualizations |
| NetworkX | Entity co-occurrence graph construction |

Additional Python standard-library modules are used for text processing, serialization, and file handling.

---

## System Architecture

```text
                News Article
                     |
          Text Input / File Upload
                     |
                     v
             Text Preprocessing
                     |
                     v
        Fine-Tuned DistilBERT Model
                     |
                     v
          Named Entity Recognition
                     |
          +----------+-----------+
          |          |           |
          v          v           v
     Entity       Character    Confidence
     Labels       Offsets        Scores
          |          |           |
          +----------+-----------+
                     |
                     v
            Structured Results
                     |
          +----------+-----------+
          |          |           |
          v          v           v
      Article     Entity      Relationship
       Lab        Atlas          Graph
          |          |           |
          +----------+-----------+
                     |
                     v
             Streamlit Dashboard
```

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip
- A compatible PyTorch environment
- The trained NewsLens model checkpoint

A CUDA-compatible NVIDIA GPU is recommended for faster inference, but CPU execution is also possible.

### 1. Clone the Repository

```bash
git clone https://
