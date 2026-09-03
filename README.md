# Census Income API

Machine learning DevOps project for predicting census income using FastAPI and scikit-learn.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

- `data/` — census training data
- `ml/` — data processing and model training code
- `model/` — serialized model artifacts
- `main.py` — FastAPI application
- `train_model.py` — model training script
