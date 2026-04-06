from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")
st.title("Churn Prediction Dashboard")
st.caption("Use the sidebar to switch between single prediction and batch prediction.")

st.markdown(
    """
### Pages
- Single Predict: predict one customer at a time.
- Batch Predict: upload CSV or paste JSON for multiple customers.
"""
)
col1, col2, col3 = st.columns(3)
col1.metric("Recall (Recallability)", "83.3%", help="Rate of finding customers who actually leave.")
col2.metric("ROC-AUC", "0.86", "Excellent")
col3.metric("Expected Cost", "0.406", "-25% compared to previous", delta_color="normal")

st.info("**Strategy:** Model optimized to minimize customer loss costs (Churn Cost).")
