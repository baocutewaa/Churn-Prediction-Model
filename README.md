# Churn Prediction System

End-to-end customer churn prediction project with:
- Model training and evaluation (Logistic Regression + XGBoost)
- FastAPI backend for real-time and batch predictions
- Streamlit frontend for single and batch scoring

Dataset: `data/Churn_Modelling.csv`

## Quick Start (Windows)

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Train The Model

```powershell
.\.venv\Scripts\python.exe model\train_model.py
```

What training does:
- Drops `RowNumber`, `CustomerId`, `Surname`
- Encodes `Geography` using one-hot encoding
- Encodes `Gender` using ordinal mapping (`Female=0`, `Male=1`)
- Adds engineered features:
  - `BalanceSalaryRatio`
  - `BalancePerProduct`
  - `TenureAgeRatio`
  - `CreditScoreAgeInteraction`
  - `ActivityBalanceInteraction`
  - `IsSenior`
- Uses train/calibration/threshold/test split (`60/10/10/20`)
- Tunes models with `GridSearchCV` + stratified 5-fold CV
- Handles class imbalance (`class_weight='balanced'`, `scale_pos_weight`)
- Calibrates probabilities with `CalibratedClassifierCV` (`sigmoid`)
- Chooses threshold by business cost (`FP=1`, `FN=5` by default)
- Selects the best model by expected cost

Saved artifacts:
- `model/churn_model.pkl` (model + thresholds + metadata)
- `model/metrics.json` (detailed evaluation metrics)

## 2. Run FastAPI Backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Useful endpoints:
- `GET /` health check
- `GET /docs` interactive API docs
- `POST /predict` single prediction
- `POST /predict-batch` batch prediction
- `GET /model-info` model metadata (thresholds, calibration, cost config, metrics)

### `POST /predict` Request Example

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

### Response Example

```json
{
  "churn_probability": 0.3814,
  "risk_level": "Low",
  "will_churn": 0,
  "applied_threshold": 0.42
}
```

Risk assignment:
- Thresholds are loaded from trained model metadata (`risk_thresholds`)
- `prob >= high_threshold` => `High`
- `medium_threshold <= prob < high_threshold` => `Medium`
- `prob < medium_threshold` => `Low`

## 3. Run Streamlit Frontend

In another terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\Home.py
```

Notes:
- Open the URL shown in terminal (usually `http://localhost:8501`)
- Keep FastAPI backend running before submitting predictions from UI

## Batch API Example

`POST /predict-batch`

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

## Project Structure

```text
api/             FastAPI service
data/            Input datasets
frontend/        Streamlit app (Home + pages)
model/           Training code, metrics, and model artifact
```
