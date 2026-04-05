# Churn Prediction System

This project builds an end-to-end churn prediction workflow using `Churn_Modelling.csv`:
- Model training with Logistic Regression (baseline) and XGBoost (main model)
- FastAPI backend for real-time scoring
- Streamlit frontend form for user input

1. Train the model

From project root:

```bash
.venv\Scripts\python.exe model/train_model.py
```

What it does:
- Drops `RowNumber`, `CustomerId`, `Surname`
- Encodes `Geography` with one-hot encoding
- Encodes `Gender` with label encoding (`Female=0`, `Male=1`)
- Adds engineered features (`BalanceSalaryRatio`, `BalancePerProduct`, `TenureAgeRatio`, `CreditScoreAgeInteraction`, `ActivityBalanceInteraction`, `IsSenior`)
- Uses train/calibration/threshold/test split (`60/10/10/20`)
- Tunes both Logistic Regression and XGBoost using `GridSearchCV` + stratified 5-fold CV
- Handles class imbalance (`class_weight='balanced'` and `scale_pos_weight`)
- Calibrates predicted probabilities (`CalibratedClassifierCV`, sigmoid)
- Optimizes a decision threshold using business cost (default: `FP=1`, `FN=5`)
- Evaluates with `Accuracy`, `Precision`, `Recall`, `F1`, `F2`, `ROC-AUC`, `PR-AUC`, `Brier Score`, `Expected Cost`
- Selects best model by `Expected Cost`
- Saves best model, thresholds, metadata to `model/churn_model.pkl`
- Saves detailed metrics and metadata to `model/metrics.json`

2. Run FastAPI backend

```bash
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs
- Prediction endpoint: `POST /predict`
- Batch prediction endpoint: `POST /predict-batch`
- Model metadata endpoint: `GET /model-info`

`GET /model-info` now also returns calibration settings and cost configuration.

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
  "risk_level": "Low",
  "will_churn": 0,
  "applied_threshold": 0.42
}
```

Risk rule:
- Dynamic thresholds are loaded from trained model metadata (`risk_thresholds`)
- `prob >= high_threshold` -> High
- `medium_threshold <= prob < high_threshold` -> Medium
- `< medium_threshold` -> Low

Cost-based threshold tuning:
- Prediction threshold is chosen to minimize: `total_cost = FP_cost * FP + FN_cost * FN`
- You can change costs in `model/train_model.py`:
  - `COST_FALSE_POSITIVE`
  - `COST_FALSE_NEGATIVE`

Batch request example (`POST /predict-batch`):

```json
{
  "records": [
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
  ]
}
```

3. Run Streamlit frontend

In a second terminal:

```bash
.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

- Open the shown local URL (usually http://localhost:8501)
- Ensure FastAPI is running before submitting predictions
