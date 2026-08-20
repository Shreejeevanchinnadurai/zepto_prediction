
# Zepto Smart Commerce AI Platform — Module 1.3: Rider Acceptance Prediction

Part of the Zepto Smart Commerce AI Platform capstone (GUVI). This submission covers
**Module 1 — ML Delivery Intelligence Engine**, specifically **Submodule 1.3: Rider
Acceptance Prediction**.

## Problem Statement

Not every delivery partner accepts every order. A long-distance order in heavy rain
with a low incentive is frequently rejected, and the system needs to know this
*before* assigning the order — otherwise the order sits unassigned and the delivery
promise breaks. This module predicts, in real time, whether a given delivery
partner is likely to **accept** or **reject** an order based on the order's
conditions and the rider's current state.

## What's Included

| Component | Description |
|---|---|
| Trained model | Random Forest classifier, scikit-learn Pipeline (preprocessing + model) |
| Streamlit dashboard | 5-page interactive app — prediction form, model performance, feature importance, EDA, sample data |
| FastAPI service | REST API with Swagger/OpenAPI docs, `/health` and `/predict-rider-acceptance` endpoints |

## Model Performance

Evaluated on a held-out 20% stratified test split (16,104 orders):

| Metric | Score |
|---|---|
| Accuracy | 89.17% |
| Precision | 90.04% |
| Recall | 92.38% |
| F1 Score | 91.20% |
| ROC-AUC | 0.9614 |

**Confusion matrix** (test set, n=16,104):

| | Predicted: Rejected | Predicted: Accepted |
|---|---|---|
| **Actual: Rejected** | 5,322 (TN) | 1,001 (FP) |
| **Actual: Accepted** | 745 (FN) | 9,036 (TP) |

**Top features by importance** (from the trained Random Forest):
1. `distance_km` (23.8%)
2. `rider_earnings_today` (22.0%)
3. `delivery_charge` (6.8%)
4. `current_incentive` (6.5%)
5. `vehicle_type` (5.8%)


## Tech Stack

Python, scikit-learn, pandas, NumPy, Streamlit, Plotly, FastAPI, Uvicorn, Pydantic

## Input Features

23 features across three categories:

- **Order details**: city, road type, distance (km), item count, order amount (₹),
  order weight (kg), delivery charge (₹)
- **Timing & conditions**: hour of day, weekday, peak hour flag, weekend flag,
  festival day flag, traffic index (0–1), rainfall (mm)
- **Rider profile**: vehicle type, rider experience (months), current active load,
  rider rating, previous acceptance rate, earnings today (₹), current incentive (₹),
  customer membership tier, demand level

## Project Structure

```
├── app.py                      # Streamlit dashboard (5 pages)
├── api.py                      # FastAPI service
├── rider_acceptance_model.pkl  # Trained model (scikit-learn Pipeline)
├── requirements.txt            # Python dependencies
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt

# Dashboard
streamlit run app.py

# API (separate terminal)
uvicorn api:app --host 0.0.0.0 --port 8000
```

Streamlit dashboard: `http://localhost:8501`
FastAPI Swagger docs: `http://localhost:8000/docs`

**Note:** `requirements.txt` pins `scikit-learn==1.6.1` to match the version the
model was trained and pickled with. Installing a different scikit-learn version can
cause a pickle-compatibility error when loading the model — if you're on a very
recent Python version (3.14+) without prebuilt wheels for 1.6.1, use a Python
3.11–3.13 virtual environment.

## Running in Google Colab

Since training was done on Colab, the dashboard and API can also be served directly
from the notebook using Colab's built-in port proxy (no external tunnel needed):

```python
# Streamlit
import subprocess, time
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
)
time.sleep(8)
from google.colab.output import serve_kernel_port_as_iframe
serve_kernel_port_as_iframe(8501, height=800)
```

```python
# FastAPI
api_proc = subprocess.Popen(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])
time.sleep(10)
serve_kernel_port_as_iframe(8000, path="/docs", height=800)
```

## API Reference

### `GET /health`
Returns service and model-load status.

### `POST /predict-rider-acceptance`
Accepts a JSON order payload (see `/docs` for the full schema and an interactive
example) and returns:

```json
{
  "prediction": "ACCEPTED",
  "acceptance_probability": 0.98,
  "model_version": "random_forest_v1"
}
```

## Dashboard Pages

1. **Predict Rider Acceptance** — interactive form; enter order/rider conditions,
   get a prediction with probability and a plain-language recommendation
2. **Model Performance** — accuracy/precision/recall/F1/ROC-AUC, confusion matrix,
   ROC curve, classification report
3. **Feature Importance** — live-extracted from the trained Random Forest
4. **EDA / Data Insights** — acceptance rate broken down by city, demand level,
   vehicle type, rainfall, peak hour, and membership tier
5. **Sample Order Data** — a scored sample of orders in the training schema, with
   CSV export
