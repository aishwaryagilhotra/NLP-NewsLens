# NewsLens UI prototype

## Files
- `app.py`: multi-page Streamlit dashboard with Article Lab, Entity Atlas, Connection Map, and Model Card.
- `requirements.txt`: UI and model dependencies.

## Connect your existing model
Copy your trained `newslens_ner_model/` directory into this folder. Create `ner_engine.py` beside `app.py` containing the tested model loading code, `id2label`, and `extract_entities(text)` function.

`extract_entities()` must return a list of dictionaries with:
- `text`: exact entity substring from the original article
- `label`: original NER label, e.g. `PERSON`, `ORG`, `GPE`, `DATE`
- `start`, `end`: character offsets into the original article (end-exclusive)
- `confidence`: optional float from 0 to 1

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Current scope / honesty notes
- Articles analyzed during the current Streamlit session are kept in session state, not in SQLite. Use “Export session library” to save the analyzed data as JSON.
- Entity Atlas searches the session's analyzed articles and does not perform automatic alias resolution.
- Connection Map shows same-sentence co-occurrence, with sentence evidence. Co-occurrence is not proof of a real-world relationship.
- Model Card displays the recorded OntoNotes test metrics from the existing experiment.
- This UI does not fetch article URLs. Paste text or upload a .txt/.md file.
