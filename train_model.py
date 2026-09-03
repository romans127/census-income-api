# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "census.csv"
MODEL_DIR = ROOT / "model"
SLICE_OUTPUT_PATH = ROOT / "slice_output.txt"
LABEL = "salary"

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_census_data(path=DATA_PATH):
    """Load the census CSV and strip leftover whitespace from the starter file."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    if "income" in data.columns and LABEL not in data.columns:
        data = data.rename(columns={"income": LABEL})
    str_cols = data.select_dtypes(include=["object", "string"]).columns
    data[str_cols] = data[str_cols].apply(lambda series: series.str.strip())
    return data


def main():
    data = load_census_data()
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, MODEL_DIR / "model.pkl")
    save_model(encoder, MODEL_DIR / "encoder.pkl")
    save_model(lb, MODEL_DIR / "lb.pkl")
    print("Model saved to model/model.pkl")

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    overall = (
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}"
    )
    print(overall)

    slice_lines = performance_on_categorical_slice(
        test, "education", model, encoder, lb, cat_features, LABEL
    )
    with open(SLICE_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(overall + "\n")
        handle.write("\n".join(slice_lines) + "\n")


if __name__ == "__main__":
    main()
