from __future__ import annotations

import json
from urllib import error

import pandas as pd
import streamlit as st

from core.constants import REQUIRED_COLUMNS
from core.settings import get_batch_url
from services.api_client import predict_batch
from utils.validators import missing_columns_in_frame, validate_json_records

st.set_page_config(page_title="Batch Churn Predictor", layout="wide")
st.title("Batch Churn Prediction")
st.caption("Upload CSV or paste JSON array.")

batch_api_url = get_batch_url()

input_mode = st.radio("Input mode", ["CSV", "JSON"], horizontal=True)
records: list[dict] = []

if input_mode == "CSV":
    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            frame = pd.read_csv(uploaded_file)
            st.subheader("Input Preview")
            st.dataframe(frame, use_container_width=True)

            missing_columns = missing_columns_in_frame(frame=frame, required_columns=REQUIRED_COLUMNS)
            if missing_columns:
                st.error("Missing columns: " + ", ".join(missing_columns))
            else:
                records = frame[REQUIRED_COLUMNS].to_dict(orient="records")
                st.success(f"Loaded {len(records)} records from CSV.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to read CSV: {exc}")
else:
    json_text = st.text_area(
        "Paste JSON array",
        height=260,
        placeholder='[{"CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 40, "Tenure": 5, "Balance": 60000.0, "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 90000.0}]',
    )
    if json_text.strip():
        try:
            parsed = json.loads(json_text)
            is_valid, message = validate_json_records(records=parsed, required_columns=REQUIRED_COLUMNS)
            if not is_valid:
                st.error(message)
            else:
                records = parsed
                st.success(f"Loaded {len(records)} records from JSON.")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")

send_disabled = len(records) == 0
if st.button("Predict Batch", type="primary", disabled=send_disabled):
    try:
        result = predict_batch(api_url=batch_api_url, records=records)

        predictions = result.get("predictions", [])
        if not predictions:
            st.warning("API returned an empty predictions list.")
        else:
            pred_frame = pd.DataFrame(predictions)
            st.subheader("Prediction Results")
            st.dataframe(pred_frame, use_container_width=True)

            churn_count = int((pred_frame["will_churn"] == 1).sum()) if "will_churn" in pred_frame.columns else 0
            st.write(f"Total records: {len(pred_frame)}")
            st.write(f"Predicted churn records: {churn_count}")

            st.download_button(
                label="Download predictions CSV",
                data=pred_frame.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

    except error.HTTPError as http_error:
        detail = http_error.read().decode("utf-8", errors="ignore")
        st.error(f"API error: {http_error.code} - {http_error.reason}")
        if detail:
            st.code(detail)
    except error.URLError:
        st.error("Cannot connect to API. Make sure FastAPI server is running.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
