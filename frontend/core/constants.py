from __future__ import annotations

REQUIRED_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

DEFAULT_PREDICT_URL = "http://127.0.0.1:8000/predict"
DEFAULT_BATCH_URL = "http://127.0.0.1:8000/predict-batch"
