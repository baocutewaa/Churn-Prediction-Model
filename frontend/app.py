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

### Deploy Secrets
Set these in Streamlit Cloud to target separate APIs:
- API_URL_PREDICT
- API_URL_BATCH
"""
)
