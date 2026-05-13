"""
frontend/pages/2_Predict.py — Single Customer Churn Prediction
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

st.set_page_config(page_title="Predict Churn", page_icon="🔮", layout="wide")
st.title("🔮 Predict Customer Churn")
st.caption("Enter customer details below to get a churn prediction.")


@st.cache_resource
def load_model():
    from src.Task7_save_model import load_model as _load
    return _load()


try:
    model, meta = load_model()
    st.success(f"✅ Model loaded: **{meta['model_name']}** ({meta['n_features']} features)", icon="✅")
except Exception as e:
    st.error(f"❌ Model not found. Please run `python app.py` first.\n\n{e}")
    st.stop()


# ── Input Form ─────────────────────────────────────────────────────────────────
st.subheader("👤 Customer Information")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        num_products = st.slider("Number of Products", 1, 5, 2)

    with c2:
        monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 200.0, 65.5, step=0.5)
        total_charges = st.number_input("Total Charges ($)", 50.0, 10000.0, 786.0, step=10.0)
        service_calls = st.slider("Customer Service Calls", 0, 10, 3)

    with c3:
        contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
        payment = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)


def prepare_input(feature_cols):
    raw = {
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "tech_support": 1 if tech_support == "Yes" else 0,
        "online_security": 1 if online_security == "Yes" else 0,
        "num_products": num_products,
        "paperless_billing": 1 if paperless == "Yes" else 0,
        "customer_service_calls": service_calls,
        "charges_per_month_ratio": monthly_charges / (tenure + 1),
        "calls_per_month": service_calls / (tenure + 1),
        "charge_growth": total_charges / (monthly_charges + 1),
        "contract_type_One Year": 1 if contract == "One Year" else 0,
        "contract_type_Two Year": 1 if contract == "Two Year" else 0,
        "internet_service_Fiber Optic": 1 if internet == "Fiber Optic" else 0,
        "internet_service_No": 1 if internet == "No" else 0,
        "payment_method_Credit Card": 1 if payment == "Credit Card" else 0,
        "payment_method_Electronic Check": 1 if payment == "Electronic Check" else 0,
        "payment_method_Mailed Check": 1 if payment == "Mailed Check" else 0,
    }
    df = pd.DataFrame([raw])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


if submitted:
    df = prepare_input(meta["feature_columns"])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    st.divider()
    st.subheader("📊 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    if prediction == 1:
        res_col1.error(f"⚠️ **WILL CHURN**")
    else:
        res_col1.success(f"✅ **WILL STAY**")

    res_col2.metric("Churn Probability", f"{probability:.1%}")
    res_col3.metric("Model", meta["model_name"])

    # Probability bar
    st.progress(probability, text=f"Churn Risk: {probability:.1%}")

    # Risk level
    if probability >= 0.75:
        st.error("🔴 **High Risk** — Immediate retention action recommended")
    elif probability >= 0.5:
        st.warning("🟡 **Medium Risk** — Consider a retention offer")
    else:
        st.success("🟢 **Low Risk** — Customer likely to stay")