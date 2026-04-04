from __future__ import annotations

import json
from urllib import error, request

import streamlit as st


st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("Customer Churn Prediction")
st.caption("Submit customer details to get churn probability and risk level.")

api_url = st.text_input("FastAPI /predict URL", value="http://127.0.0.1:8000/predict")

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

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        probability = float(data["churn_probability"])
        risk = str(data["risk_level"])

        st.success("Prediction complete")
        st.metric("Churn Probability", f"{probability:.2%}")

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
