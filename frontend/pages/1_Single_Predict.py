from __future__ import annotations

from urllib import error

import streamlit as st

from core.settings import get_predict_url
from services.api_client import predict_single

st.set_page_config(page_title="Single Churn Predictor", page_icon="📉", layout="centered")
st.title("Customer Churn Prediction")
st.caption("Submit customer details to get churn probability and risk level.")

api_url = get_predict_url()

with st.form("churn_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=100.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
    has_card = st.selectbox("Has Credit Card", options=[0, 1])
    is_active = st.selectbox("Is Active Member", options=[0, 1])
    salary = st.number_input("Estimated Salary", min_value=0.0, value=90000.0, step=100.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": float(salary),
    }

    try:
        data = predict_single(api_url=api_url, payload=payload)

        probability = float(data["churn_probability"])
        risk = str(data["risk_level"])
        will_churn = int(data.get("will_churn", 0))
        applied_threshold = float(data.get("applied_threshold", 0.4))

        st.success("Prediction complete")
        st.metric("Churn Probability", f"{probability:.2%}")
        st.write(f"Decision Threshold: **{applied_threshold:.2f}**")
        st.write(f"Predicted Class: **{'Churn' if will_churn == 1 else 'No Churn'}**")

        if risk == "High":
            st.error(f"Risk Level: {risk}")
        elif risk == "Medium":
            st.warning(f"Risk Level: {risk}")
        else:
            st.info(f"Risk Level: {risk}")

    except error.HTTPError as http_error:
        st.error(f"API error: {http_error.code} - {http_error.reason}")
    except error.URLError:
        st.error("Cannot connect to API. Ensure FastAPI server is running.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
