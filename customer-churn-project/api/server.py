"""
api/server.py — FastAPI backend for Customer Churn Prediction
Run: uvicorn api.server:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using trained ML models. Run `python app.py` first to train and save the model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model on startup ──────────────────────────────────────────────────────
_model = None
_meta = None


def get_model():
    global _model, _meta
    if _model is None:
        from src.Task7_save_model import load_model
        _model, _meta = load_model()
    return _model, _meta


# ── Schemas ────────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    age: float = Field(35, example=35)
    gender: int = Field(1, description="1=Male, 0=Female", example=1)
    tenure_months: float = Field(12, example=12)
    monthly_charges: float = Field(65.5, example=65.5)
    total_charges: float = Field(786.0, example=786.0)
    contract_type: str = Field("Month-to-Month", example="Month-to-Month")
    internet_service: str = Field("Fiber Optic", example="Fiber Optic")
    tech_support: int = Field(0, description="1=Yes, 0=No", example=0)
    online_security: int = Field(0, description="1=Yes, 0=No", example=0)
    num_products: int = Field(2, example=2)
    payment_method: str = Field("Electronic Check", example="Electronic Check")
    paperless_billing: int = Field(1, description="1=Yes, 0=No", example=1)
    customer_service_calls: int = Field(3, example=3)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    churn_label: str
    confidence: str
    model_used: str


# ── Helper ─────────────────────────────────────────────────────────────────────
def prepare_features(customer: CustomerInput, feature_cols: list) -> pd.DataFrame:
    """Convert raw input to a feature-aligned DataFrame the model expects."""
    raw = {
        "age": customer.age,
        "gender": customer.gender,
        "tenure_months": customer.tenure_months,
        "monthly_charges": customer.monthly_charges,
        "total_charges": customer.total_charges,
        "tech_support": customer.tech_support,
        "online_security": customer.online_security,
        "num_products": customer.num_products,
        "paperless_billing": customer.paperless_billing,
        "customer_service_calls": customer.customer_service_calls,
        # Engineered features
        "charges_per_month_ratio": customer.monthly_charges / (customer.tenure_months + 1),
        "calls_per_month": customer.customer_service_calls / (customer.tenure_months + 1),
        "charge_growth": customer.total_charges / (customer.monthly_charges + 1),
    }

    # One-hot encode contract_type
    for ct in ["One Year", "Two Year"]:
        col = f"contract_type_{ct}"
        raw[col] = 1 if customer.contract_type == ct else 0

    # One-hot encode internet_service
    for svc in ["Fiber Optic", "No"]:
        col = f"internet_service_{svc}"
        raw[col] = 1 if customer.internet_service == svc else 0

    # One-hot encode payment_method
    for pm in ["Credit Card", "Electronic Check", "Mailed Check"]:
        col = f"payment_method_{pm}"
        raw[col] = 1 if customer.payment_method == pm else 0

    df = pd.DataFrame([raw])

    # Align to expected feature columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return df


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Customer Churn API is running. Visit /docs for Swagger UI."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata about the loaded model."""
    try:
        _, meta = get_model()
        return meta
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """Predict churn for a single customer."""
    try:
        model, meta = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    df = prepare_features(customer, meta["feature_columns"])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    confidence = "High" if probability > 0.75 or probability < 0.25 else "Medium"
    if 0.4 <= probability <= 0.6:
        confidence = "Low"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        churn_label="Will Churn" if prediction == 1 else "Will Stay",
        confidence=confidence,
        model_used=meta["model_name"],
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerInput]):
    """Predict churn for a batch of customers (max 100)."""
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100.")
    try:
        model, meta = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = []
    for i, c in enumerate(customers):
        df = prepare_features(c, meta["feature_columns"])
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
        results.append({
            "index": i,
            "churn_prediction": pred,
            "churn_probability": round(prob, 4),
            "churn_label": "Will Churn" if pred == 1 else "Will Stay",
        })

    return {"total": len(results), "predictions": results}


@app.get("/stats", tags=["Analytics"])
def get_stats():
    """Return basic dataset statistics if available."""
    stats_path = "data/descriptive_stats.csv"
    comparison_path = "data/model_comparison.csv"

    result = {}
    if os.path.exists(stats_path):
        import csv
        with open(stats_path) as f:
            result["descriptive_stats"] = "Available — see /data/descriptive_stats.csv"
    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)
        result["model_comparison"] = df.to_dict(orient="records")

    if not result:
        raise HTTPException(status_code=404, detail="Run app.py first to generate statistics.")

    return result