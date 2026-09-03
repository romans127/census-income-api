from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "census.csv"
LABEL = "salary"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _load_sample(n_rows=200):
    """Load a small cleaned slice of the census CSV for unit tests."""
    data = pd.read_csv(DATA_PATH, nrows=n_rows)
    data.columns = data.columns.str.strip()
    if "income" in data.columns and LABEL not in data.columns:
        data = data.rename(columns={"income": LABEL})
    str_cols = data.select_dtypes(include=["object", "string"]).columns
    data[str_cols] = data[str_cols].apply(lambda series: series.str.strip())
    return data


def _processed_sample():
    data = _load_sample()
    return process_data(
        data, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )


def test_train_model_returns_fitted_classifier():
    X, y, _, _ = _processed_sample()
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_") or hasattr(model, "n_features_in_")


def test_inference_output_shape_and_values():
    X, y, _, _ = _processed_sample()
    model = train_model(X, y)
    preds = inference(model, X)
    assert np.ndim(preds) == 1
    assert len(preds) == X.shape[0]
    assert np.isin(preds, [0, 1]).all()


def test_compute_model_metrics_range():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    preds = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    for metric in (precision, recall, fbeta):
        assert 0.0 <= metric <= 1.0


def test_process_data_binarizes_label():
    data = _load_sample()
    X, y, encoder, _ = process_data(
        data, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    assert set(np.unique(y)).issubset({0, 1})
    n_continuous = data.drop(columns=CAT_FEATURES + [LABEL]).shape[1]
    n_encoded = encoder.transform(data[CAT_FEATURES].values).shape[1]
    assert X.shape[1] == n_continuous + n_encoded
