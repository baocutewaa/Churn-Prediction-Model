from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"


DEFAULT_RISK_THRESHOLDS = {"medium": 0.4, "high": 0.7}
DEFAULT_SEGMENT_THRESHOLDS = {"VIP": 0.4, "Regular": 0.4}
DEFAULT_SEGMENTATION_CONFIG = {
    "vip_balance_threshold": 100000.0,
    "vip_salary_threshold": 120000.0,
}


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


class BatchChurnInput(BaseModel):
    records: list[ChurnInput] = Field(..., min_length=1, max_length=500)


def get_risk_level(probability: float, medium_threshold: float, high_threshold: float) -> str:
    if probability >= high_threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


def get_customer_segment(customer: ChurnInput, segmentation_config: dict | None) -> str:
    config = DEFAULT_SEGMENTATION_CONFIG.copy()
    if isinstance(segmentation_config, dict):
        config.update(segmentation_config)

    vip_balance_threshold = float(config.get("vip_balance_threshold", 100000.0))
    vip_salary_threshold = float(config.get("vip_salary_threshold", 120000.0))

    is_vip = (
        float(customer.Balance) >= vip_balance_threshold
        or float(customer.EstimatedSalary) >= vip_salary_threshold
    )
    return "VIP" if is_vip else "Regular"


def get_thresholds_for_segment(model_bundle: dict, customer_segment: str) -> tuple[float, float]:
    risk_thresholds = model_bundle.get("risk_thresholds", DEFAULT_RISK_THRESHOLDS)
    global_medium_threshold = float(risk_thresholds.get("medium", 0.4))
    global_high_threshold = float(risk_thresholds.get("high", 0.7))

    segment_thresholds = model_bundle.get("segment_thresholds", DEFAULT_SEGMENT_THRESHOLDS)
    medium_threshold = float(segment_thresholds.get(customer_segment, global_medium_threshold))
    medium_threshold = float(max(0.05, min(0.95, medium_threshold)))

    threshold_gap = max(0.05, global_high_threshold - global_medium_threshold)
    high_threshold = float(min(0.95, medium_threshold + threshold_gap))
    return medium_threshold, high_threshold


def get_global_medium_threshold(model_bundle: dict) -> float:
    risk_thresholds = model_bundle.get("risk_thresholds", DEFAULT_RISK_THRESHOLDS)
    return float(risk_thresholds.get("medium", 0.4))


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


@app.get("/model-info")
def model_info() -> dict:
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    return {
        "model_name": model_bundle.get("model_name"),
        "selection_metric": model_bundle.get("selection_metric"),
        "calibration": model_bundle.get("calibration"),
        "cost_config": model_bundle.get("cost_config"),
        "risk_thresholds": model_bundle.get("risk_thresholds", DEFAULT_RISK_THRESHOLDS),
        "segment_thresholds": model_bundle.get("segment_thresholds", DEFAULT_SEGMENT_THRESHOLDS),
        "segmentation_config": model_bundle.get("segmentation_config", DEFAULT_SEGMENTATION_CONFIG),
        "best_metrics": model_bundle.get("best_metrics"),
        "train_timestamp_utc": model_bundle.get("train_timestamp_utc"),
    }


@app.post("/predict")
def predict(payload: ChurnInput) -> dict[str, float | str | int | bool]:
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    model = model_bundle["model"]
    segmentation_config = model_bundle.get("segmentation_config", DEFAULT_SEGMENTATION_CONFIG)
    customer_segment = get_customer_segment(payload, segmentation_config)
    medium_threshold, high_threshold = get_thresholds_for_segment(model_bundle, customer_segment)
    global_medium_threshold = get_global_medium_threshold(model_bundle)

    input_frame = pd.DataFrame([payload.model_dump()])
    churn_probability = float(model.predict_proba(input_frame)[0, 1])
    will_churn = int(churn_probability >= medium_threshold)

    return {
        "churn_probability": round(churn_probability, 4),
        "customer_segment": customer_segment,
        "risk_level": get_risk_level(churn_probability, medium_threshold, high_threshold),
        "will_churn": will_churn,
        "applied_threshold": round(medium_threshold, 4),
        "global_threshold": round(global_medium_threshold, 4),
        "is_segment_threshold_adjusted": bool(abs(medium_threshold - global_medium_threshold) > 1e-12),
    }


@app.post("/predict-batch")
def predict_batch(payload: BatchChurnInput) -> dict[str, list[dict[str, float | str | int | bool]]]:
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    model = model_bundle["model"]
    segmentation_config = model_bundle.get("segmentation_config", DEFAULT_SEGMENTATION_CONFIG)
    global_medium_threshold = get_global_medium_threshold(model_bundle)

    input_frame = pd.DataFrame([record.model_dump() for record in payload.records])
    churn_probabilities = model.predict_proba(input_frame)[:, 1]

    predictions: list[dict[str, float | str | int]] = []
    for record, probability in zip(payload.records, churn_probabilities):
        churn_probability = float(probability)
        customer_segment = get_customer_segment(record, segmentation_config)
        medium_threshold, high_threshold = get_thresholds_for_segment(model_bundle, customer_segment)
        predictions.append(
            {
                "churn_probability": round(churn_probability, 4),
                "customer_segment": customer_segment,
                "risk_level": get_risk_level(churn_probability, medium_threshold, high_threshold),
                "will_churn": int(churn_probability >= medium_threshold),
                "applied_threshold": round(medium_threshold, 4),
                "global_threshold": round(global_medium_threshold, 4),
                "is_segment_threshold_adjusted": bool(abs(medium_threshold - global_medium_threshold) > 1e-12),
            }
        )

    return {"predictions": predictions}
