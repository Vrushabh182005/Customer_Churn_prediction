"""
frontend/app_ui.py — Streamlit main page
Run: streamlit run frontend/app_ui.py
"""

import streamlit as st

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# Custom CSS (Light Theme)
# ---------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #e74c3c;
        margin-bottom: 0.3rem;
    }

    .sub-header {
        font-size: 1.15rem;
        color: #555555;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: #ffffff;
        color: #222222;
        border: 1px solid #dddddd;
        border-radius: 14px;
        padding: 1.5rem;
        min-height: 220px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    .feature-card h3 {
        color: #e74c3c;
        margin-bottom: 0.6rem;
    }

    .feature-card p {
        color: #444444;
        line-height: 1.5;
    }

    .stButton>button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown(
    '<div class="main-header">📉 Customer Churn Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">ML-powered churn prediction using Random Forest, XGBoost & LightGBM</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("🤖 ML Models", "3", "RF · XGB · LGBM")
col2.metric("📊 Metrics", "5", "Acc · P · R · F1 · AUC")
col3.metric("🔧 Pipeline Tasks", "8", "End-to-End")
col4.metric("🎯 Target", "Churn", "Binary Classification")

st.divider()

# ---------------------------------------------------
# Navigation Cards
# ---------------------------------------------------
st.subheader("🗺️ Navigate the App")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Analytics</h3>
        <p>Explore churn trends, feature distributions, correlations, and descriptive statistics from the dataset.</p>
        <p><b>→ Sidebar / Analytics</b></p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔮 Predict</h3>
        <p>Enter customer details and get an instant churn prediction with probability score from the best model.</p>
        <p><b>→ Sidebar / Predict</b></p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <h3>📈 Model Report</h3>
        <p>View model comparison, ROC curves, confusion matrices, and feature importance charts.</p>
        <p><b>→ Sidebar / Model Report</b></p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# Quick Setup
# ---------------------------------------------------
st.subheader("⚙️ Quick Setup")

st.code("python app.py", language="bash")
st.write("This generates:")
st.write("- data/ processed files + charts")
st.write("- models/best_model.pkl")
st.write("- data/model_comparison.csv")

st.info("Use the sidebar on the left to navigate between pages.")