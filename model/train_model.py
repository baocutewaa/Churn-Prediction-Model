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
VIP_FALSE_NEGATIVE_COST_MULTIPLIER = 1.5

# Segment-based threshold settings.
VIP_BALANCE_THRESHOLD = 100000.0
VIP_SALARY_THRESHOLD = 120000.0

VIP_THRESHOLD_MIN = 0.12
VIP_THRESHOLD_MAX = 0.20
REGULAR_THRESHOLD_MIN = 0.24
REGULAR_THRESHOLD_MAX = 0.34


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


def find_best_threshold_by_cost_with_bounds(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_false_positive: float,
    cost_false_negative: float,
    min_threshold: float,
    max_threshold: float,
) -> tuple[float, float]:
    min_threshold = float(max(0.0, min_threshold))
    max_threshold = float(min(1.0, max_threshold))
    if min_threshold > max_threshold:
        min_threshold, max_threshold = max_threshold, min_threshold

    candidate_thresholds = np.unique(np.concatenate(([min_threshold], y_prob, [max_threshold])))
    candidate_thresholds = candidate_thresholds[
        (candidate_thresholds >= min_threshold) & (candidate_thresholds <= max_threshold)
    ]

    if candidate_thresholds.size == 0:
        candidate_thresholds = np.linspace(min_threshold, max_threshold, num=100)

    best_threshold = float(min_threshold)
    best_cost = float("inf")

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        false_positives = float(((y_pred == 1) & (y_true == 0)).sum())
        false_negatives = float(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = cost_false_positive * false_positives + cost_false_negative * false_negatives

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(threshold)

    return float(best_threshold), float(best_cost)


def assign_customer_segments(input_frame: pd.DataFrame, segmentation_config: dict[str, float]) -> pd.Series:
    vip_balance_threshold = float(segmentation_config["vip_balance_threshold"])
    vip_salary_threshold = float(segmentation_config["vip_salary_threshold"])

    is_vip = (
        (input_frame["Balance"] >= vip_balance_threshold)
        | (input_frame["EstimatedSalary"] >= vip_salary_threshold)
    )
    return is_vip.map({True: "VIP", False: "Regular"})


def get_segment_cost_weights(segment_name: str) -> tuple[float, float]:
    if segment_name == "VIP":
        return 1.0, VIP_FALSE_NEGATIVE_COST_MULTIPLIER
    return 1.0, 1.0


def get_segment_threshold_bounds(segment_name: str) -> tuple[float, float]:
    if segment_name == "VIP":
        return VIP_THRESHOLD_MIN, VIP_THRESHOLD_MAX
    return REGULAR_THRESHOLD_MIN, REGULAR_THRESHOLD_MAX


def find_best_segment_thresholds_by_cost(
    x_frame: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    segmentation_config: dict[str, float],
    cost_false_positive: float,
    cost_false_negative: float,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    global_threshold, global_cost = find_best_threshold_by_cost(
        y_true=y_true,
        y_prob=y_prob,
        cost_false_positive=cost_false_positive,
        cost_false_negative=cost_false_negative,
    )

    segments = assign_customer_segments(x_frame, segmentation_config)
    segment_thresholds: dict[str, float] = {}
    segment_costs: dict[str, float] = {}

    for segment_name in ["VIP", "Regular"]:
        segment_mask = segments == segment_name
        if int(segment_mask.sum()) == 0:
            segment_thresholds[segment_name] = global_threshold
            segment_costs[segment_name] = 0.0
            continue

        fp_weight, fn_weight = get_segment_cost_weights(segment_name)

        min_threshold, max_threshold = get_segment_threshold_bounds(segment_name)

        seg_threshold, seg_cost = find_best_threshold_by_cost_with_bounds(
            y_true=y_true.loc[segment_mask],
            y_prob=y_prob[segment_mask.values],
            cost_false_positive=cost_false_positive * fp_weight,
            cost_false_negative=cost_false_negative * fn_weight,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        segment_thresholds[segment_name] = seg_threshold
        segment_costs[segment_name] = seg_cost

    return segment_thresholds, segment_costs, global_threshold, global_cost


def evaluate_model_segmented(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    segment_thresholds: dict[str, float],
    segmentation_config: dict[str, float],
    cost_false_positive: float,
    cost_false_negative: float,
) -> dict[str, float | dict[str, float]]:
    y_prob = model.predict_proba(x_test)[:, 1]
    segments = assign_customer_segments(x_test, segmentation_config)

    per_row_threshold = np.array([
        float(segment_thresholds.get(segment, segment_thresholds.get("Regular", 0.4)))
        for segment in segments
    ])
    y_pred = (y_prob >= per_row_threshold).astype(int)

    false_positives = float(((y_pred == 1) & (y_test == 0)).sum())
    false_negatives = float(((y_pred == 0) & (y_test == 1)).sum())

    vip_mask = segments == "VIP"
    regular_mask = segments == "Regular"
    vip_fp = float(((y_pred == 1) & (y_test == 0) & vip_mask).sum())
    vip_fn = float(((y_pred == 0) & (y_test == 1) & vip_mask).sum())
    regular_fp = float(((y_pred == 1) & (y_test == 0) & regular_mask).sum())
    regular_fn = float(((y_pred == 0) & (y_test == 1) & regular_mask).sum())

    _, vip_fn_weight = get_segment_cost_weights("VIP")
    _, regular_fn_weight = get_segment_cost_weights("Regular")
    total_cost = (
        cost_false_positive * (vip_fp + regular_fp)
        + cost_false_negative * vip_fn_weight * vip_fn
        + cost_false_negative * regular_fn_weight * regular_fn
    )

    segment_distribution = {
        "VIP": float((segments == "VIP").mean()),
        "Regular": float((segments == "Regular").mean()),
    }

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
        "threshold": float(segment_thresholds.get("Regular", 0.4)),
        "segment_distribution": segment_distribution,
        "segment_cost_config": {
            "vip_fn_multiplier": float(VIP_FALSE_NEGATIVE_COST_MULTIPLIER),
            "regular_fn_multiplier": 1.0,
        },
    }


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

    segmentation_config = {
        "vip_balance_threshold": VIP_BALANCE_THRESHOLD,
        "vip_salary_threshold": VIP_SALARY_THRESHOLD,
        "rule": "VIP if Balance >= vip_balance_threshold or EstimatedSalary >= vip_salary_threshold",
    }

    all_metrics: dict[str, dict[str, float | dict[str, float]]] = {}
    cv_scores: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    fitted_models: dict[str, CalibratedClassifierCV] = {}
    threshold_costs: dict[str, float] = {}
    segment_thresholds_by_model: dict[str, dict[str, float]] = {}
    segment_threshold_costs_by_model: dict[str, dict[str, float]] = {}

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
        segment_thresholds, segment_threshold_costs, global_threshold, global_threshold_cost = find_best_segment_thresholds_by_cost(
            x_frame=x_threshold,
            y_true=y_threshold,
            y_prob=threshold_prob,
            segmentation_config=segmentation_config,
            cost_false_positive=COST_FALSE_POSITIVE,
            cost_false_negative=COST_FALSE_NEGATIVE,
        )

        thresholds[name] = global_threshold
        threshold_costs[name] = global_threshold_cost
        segment_thresholds_by_model[name] = segment_thresholds
        segment_threshold_costs_by_model[name] = segment_threshold_costs
        cv_scores[name] = cv_score
        all_metrics[name] = evaluate_model_segmented(
            calibrated_model,
            x_test,
            y_test,
            segment_thresholds=segment_thresholds,
            segmentation_config=segmentation_config,
            cost_false_positive=COST_FALSE_POSITIVE,
            cost_false_negative=COST_FALSE_NEGATIVE,
        )
        fitted_models[name] = calibrated_model

    best_model_name = min(all_metrics, key=lambda model_name: all_metrics[model_name]["expected_cost"])
    best_model = fitted_models[best_model_name]

    medium_threshold = float(segment_thresholds_by_model[best_model_name]["Regular"])
    high_threshold = float(min(0.95, medium_threshold + 0.2))
    segment_thresholds = segment_thresholds_by_model[best_model_name]

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
        "model_segment_thresholds": segment_thresholds_by_model,
        "segment_threshold_total_cost": segment_threshold_costs_by_model,
        "cost_config": {
            "false_positive": COST_FALSE_POSITIVE,
            "false_negative": COST_FALSE_NEGATIVE,
            "vip_false_negative_multiplier": VIP_FALSE_NEGATIVE_COST_MULTIPLIER,
        },
        "risk_thresholds": {
            "medium": medium_threshold,
            "high": high_threshold,
        },
        "segment_thresholds": segment_thresholds,
        "segmentation_config": segmentation_config,
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
        "model_segment_thresholds": payload["model_segment_thresholds"],
        "segment_threshold_total_cost": payload["segment_threshold_total_cost"],
        "cost_config": payload["cost_config"],
        "risk_thresholds": payload["risk_thresholds"],
        "segment_thresholds": payload["segment_thresholds"],
        "segmentation_config": payload["segmentation_config"],
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
