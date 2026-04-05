from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from model.feature_engineering import add_engineered_features
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
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

RAW_NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "BalanceSalaryRatio",
    "BalancePerProduct",
    "TenureAgeRatio",
    "CreditScoreAgeInteraction",
    "ActivityBalanceInteraction",
    "IsSenior",
]

NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS

GEOGRAPHY_COLUMN = ["Geography"]
GENDER_COLUMN = ["Gender"]
RANDOM_STATE = 42
CV_FOLDS = 5
CALIBRATION_METHOD = "sigmoid"

# Business costs for threshold tuning.
COST_FALSE_POSITIVE = 1.0
COST_FALSE_NEGATIVE = 5.0


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

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
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
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def find_best_threshold_by_cost(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_false_positive: float,
    cost_false_negative: float,
) -> tuple[float, float]:
    candidate_thresholds = np.unique(np.concatenate(([0.0], y_prob, [1.0])))

    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        false_positives = float(((y_pred == 1) & (y_true == 0)).sum())
        false_negatives = float(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = cost_false_positive * false_positives + cost_false_negative * false_negatives

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(threshold)

    return float(np.clip(best_threshold, 0.05, 0.95)), float(best_cost)


def evaluate_model(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    cost_false_positive: float,
    cost_false_negative: float,
) -> dict[str, float]:
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    false_positives = float(((y_pred == 1) & (y_test == 0)).sum())
    false_negatives = float(((y_pred == 0) & (y_test == 1)).sum())
    total_cost = cost_false_positive * false_positives + cost_false_negative * false_negatives

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "brier_score": float(brier_score_loss(y_test, y_prob)),
        "expected_cost": float(total_cost / max(len(y_test), 1)),
        "threshold": float(threshold),
    }


def tune_model(pipeline: Pipeline, param_grid: dict, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, float]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, float(search.best_score_)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLUMNS)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    x_calibration, x_threshold, y_calibration, y_threshold = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = max(1.0, negative_count / max(1, positive_count))

    models = {
        "logistic_regression": {
            "pipeline": build_logistic_pipeline(),
            "param_grid": {
                "classifier__C": [0.5, 1.0, 2.0],
                "classifier__solver": ["lbfgs"],
            },
        },
        "xgboost": {
            "pipeline": build_xgboost_pipeline(scale_pos_weight=scale_pos_weight),
            "param_grid": {
                "classifier__n_estimators": [250, 350],
                "classifier__max_depth": [3, 4, 5],
                "classifier__learning_rate": [0.03, 0.05],
            },
        },
    }

    all_metrics: dict[str, dict[str, float]] = {}
    cv_scores: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    fitted_models: dict[str, CalibratedClassifierCV] = {}
    threshold_costs: dict[str, float] = {}

    for name, model_config in models.items():
        tuned_model, cv_score = tune_model(
            model_config["pipeline"],
            model_config["param_grid"],
            x_train,
            y_train,
        )

        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(tuned_model),
            method=CALIBRATION_METHOD,
            cv=None,
        )
        calibrated_model.fit(x_calibration, y_calibration)

        threshold_prob = calibrated_model.predict_proba(x_threshold)[:, 1]
        threshold, threshold_cost = find_best_threshold_by_cost(
            y_threshold,
            threshold_prob,
            cost_false_positive=COST_FALSE_POSITIVE,
            cost_false_negative=COST_FALSE_NEGATIVE,
        )

        thresholds[name] = threshold
        threshold_costs[name] = threshold_cost
        cv_scores[name] = cv_score
        all_metrics[name] = evaluate_model(
            calibrated_model,
            x_test,
            y_test,
            threshold,
            cost_false_positive=COST_FALSE_POSITIVE,
            cost_false_negative=COST_FALSE_NEGATIVE,
        )
        fitted_models[name] = calibrated_model

    best_model_name = min(all_metrics, key=lambda model_name: all_metrics[model_name]["expected_cost"])
    best_model = fitted_models[best_model_name]

    medium_threshold = float(thresholds[best_model_name])
    high_threshold = float(min(0.95, medium_threshold + 0.2))

    payload = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "selection_metric": "expected_cost",
        "calibration": {
            "enabled": True,
            "method": CALIBRATION_METHOD,
        },
        "cv_roc_auc": cv_scores,
        "model_thresholds": thresholds,
        "threshold_total_cost": threshold_costs,
        "cost_config": {
            "false_positive": COST_FALSE_POSITIVE,
            "false_negative": COST_FALSE_NEGATIVE,
        },
        "risk_thresholds": {
            "medium": medium_threshold,
            "high": high_threshold,
        },
        "train_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": {
            "train": 0.6,
            "calibration": 0.1,
            "threshold_tuning": 0.1,
            "test": 0.2,
        },
        "class_balance": {
            "positive_count_train": positive_count,
            "negative_count_train": negative_count,
            "scale_pos_weight_xgboost": float(scale_pos_weight),
        },
        "best_metrics": all_metrics[best_model_name],
        "all_metrics": all_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    metrics_payload = {
        "model_name": payload["model_name"],
        "selection_metric": payload["selection_metric"],
        "calibration": payload["calibration"],
        "cv_roc_auc": payload["cv_roc_auc"],
        "model_thresholds": payload["model_thresholds"],
        "threshold_total_cost": payload["threshold_total_cost"],
        "cost_config": payload["cost_config"],
        "risk_thresholds": payload["risk_thresholds"],
        "train_timestamp_utc": payload["train_timestamp_utc"],
        "split": payload["split"],
        "class_balance": payload["class_balance"],
        "best_metrics": payload["best_metrics"],
        "all_metrics": payload["all_metrics"],
    }

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)

    print("Model training complete.")
    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print("Evaluation metrics:")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
