from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "model" / "metrics.json"


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}

    try:
        with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except (json.JSONDecodeError, OSError):
        return {}


st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")
st.markdown("<h1>Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Use the sidebar to switch between single prediction and batch prediction.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
### Pages
- Single Predict: predict one customer at a time.
- Batch Predict: upload CSV or paste JSON for multiple customers.
"""
)

metrics = load_metrics()
best_metrics = metrics.get("best_metrics", {})
segment_thresholds = metrics.get("segment_thresholds", {})

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Accuracy", f"{float(best_metrics.get('accuracy', 0.0)):.1%}")
metric_col2.metric(
    "Recall",
    f"{float(best_metrics.get('recall', 0.0)):.1%}",
    help="Rate of finding customers who actually leave.",
)
metric_col3.metric("Precision", f"{float(best_metrics.get('precision', 0.0)):.1%}")
metric_col4.metric("ROC-AUC", f"{float(best_metrics.get('roc_auc', 0.0)):.3f}")


st.subheader("Segment Thresholds")
threshold_col1, threshold_col2 = st.columns(2)
threshold_col1.metric("VIP Threshold", f"{float(segment_thresholds.get('VIP', 0.0)):.3f}")
threshold_col2.metric("Regular Threshold", f"{float(segment_thresholds.get('Regular', 0.0)):.3f}")

st.info("**Strategy:** Cost-sensitive churn optimization with segment-based thresholds.")
