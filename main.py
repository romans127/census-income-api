# FastAPI app serving the census income classifier.
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ml.data import process_data
from ml.model import inference, load_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

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

model = load_model(MODEL_DIR / "model.pkl")
encoder = load_model(MODEL_DIR / "encoder.pkl")
lb = load_model(MODEL_DIR / "lb.pkl")

app = FastAPI(title="Census Income Classifier")


class ModelWithAlias(BaseModel):
    """Census record submitted for inference.

    Hyphenated census column names are exposed through field aliases so the
    JSON payload matches the original training data.
    """

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlwgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )


@app.get("/")
async def greeting():
    """Return a welcome message."""
    return {"message": "Hello, welcome to the census income classifier."}


@app.post("/predict")
async def predict(record: ModelWithAlias):
    """Run inference on a single census record."""
    row = pd.DataFrame(
        [record.model_dump(by_alias=True)],
        columns=[
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
        ],
    )
    X, _, _, _ = process_data(
        row,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X)
    prediction = lb.classes_[int(preds[0])]
    return {"prediction": prediction}
