import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Model Report", page_icon="📈", layout="wide")
st.title("📈 Model Report")

DATA_DIR = "data"

def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

results = load_csv("model_comparison.csv")

if results is None:
    st.warning("Run python app.py first.")
    st.stop()

st.subheader("🏆 Model Comparison")
st.dataframe(results, use_container_width=True)

charts = [
    "model_comparison.png",
    "roc_curves.png",
    "confusion_matrices.png",
]

for chart in charts:
    path = os.path.join(DATA_DIR, chart)
    if os.path.exists(path):
        st.image(path, use_container_width=True)

# show feature importance images automatically
for file in os.listdir(DATA_DIR):
    if file.startswith("feature_importance_") and file.endswith(".png"):
        st.image(os.path.join(DATA_DIR, file), use_container_width=True)