from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"


class ChurnInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=50)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=10)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)


def get_risk_level(probability: float) -> str:
    if probability > 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


app = FastAPI(title="Churn Prediction API", version="1.0.0")
model_bundle = None


@app.on_event("startup")
def load_model() -> None:
    global model_bundle

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run model/train_model.py first."
        )

    model_bundle = joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(payload: ChurnInput) -> dict[str, float | str]:
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    model = model_bundle["model"]
    input_frame = pd.DataFrame([payload.model_dump()])
    churn_probability = float(model.predict_proba(input_frame)[0, 1])

    return {
        "churn_probability": round(churn_probability, 4),
        "risk_level": get_risk_level(churn_probability),
    }
