from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Churn_Modelling.csv"
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET = "Exited"
FEATURE_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

GEOGRAPHY_COLUMN = ["Geography"]
GENDER_COLUMN = ["Gender"]


def build_logistic_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("geo", OneHotEncoder(handle_unknown="ignore"), GEOGRAPHY_COLUMN),
            # Label encoding for gender: Female -> 0, Male -> 1.
            ("gender", OrdinalEncoder(categories=[["Female", "Male"]]), GENDER_COLUMN),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1000, random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def build_xgboost_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("geo", OneHotEncoder(handle_unknown="ignore"), GEOGRAPHY_COLUMN),
            # Label encoding for gender: Female -> 0, Male -> 1.
            ("gender", OrdinalEncoder(categories=[["Female", "Male"]]), GENDER_COLUMN),
            ("num", "passthrough", NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLUMNS)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": build_logistic_pipeline(),
        "xgboost": build_xgboost_pipeline(),
    }

    all_metrics: dict[str, dict[str, float]] = {}

    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        all_metrics[name] = evaluate_model(pipeline, x_test, y_test)

    best_model_name = max(all_metrics, key=lambda model_name: all_metrics[model_name]["roc_auc"])
    best_model = models[best_model_name]

    payload = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "best_metrics": all_metrics[best_model_name],
        "all_metrics": all_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload["all_metrics"], metrics_file, indent=2)

    print("Model training complete.")
    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print("Evaluation metrics:")
    print(json.dumps(payload["all_metrics"], indent=2))


if __name__ == "__main__":
    main()
