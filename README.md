# Churn Prediction System

This project builds an end-to-end churn prediction workflow using `Churn_Modelling.csv`:
- Model training with Logistic Regression (baseline) and XGBoost (main model)
- FastAPI backend for real-time scoring
- Streamlit frontend form for user input

## 1) Train the model

From project root:

```bash
.venv\Scripts\python.exe model/train_model.py
```

What it does:
- Drops `RowNumber`, `CustomerId`, `Surname`
- Encodes `Geography` with one-hot encoding
- Encodes `Gender` with label encoding (`Female=0`, `Male=1`)
- Uses 80/20 train/test split
- Trains Logistic Regression and XGBoost
- Evaluates with Accuracy, Precision, Recall, ROC-AUC
- Saves best model (by ROC-AUC) to `model/churn_model.pkl`

## 2) Run FastAPI backend

```bash
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs
- Prediction endpoint: `POST /predict`

Example request body:

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Female",
  "Age": 40,
  "Tenure": 5,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 90000
}
```

Example response:

```json
{
  "churn_probability": 0.3814,
  "risk_level": "Low"
}
```

Risk rule:
- `> 0.7` -> High
- `0.4 to 0.7` -> Medium
- `< 0.4` -> Low

## 3) Run Streamlit frontend

In a second terminal:

```bash
.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

- Open the shown local URL (usually http://localhost:8501)
- Ensure FastAPI is running before submitting predictions
