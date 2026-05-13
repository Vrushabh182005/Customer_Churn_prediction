import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
st.title("📊 Data Analytics")
st.caption("Insights from the customer churn dataset")

DATA_DIR = "data"


# -------------------------------------------------
# Load CSV
# -------------------------------------------------
def load_data(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# -------------------------------------------------
# Detect target column
# -------------------------------------------------
def ensure_churn_column(df):
    possible = [
        "churn", "exited", "attrition",
        "target", "label", "churn_value"
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for col in possible:
        if col in lower_map:
            real = lower_map[col]
            if real != "churn":
                df = df.rename(columns={real: "churn"})
            return df

    return df


# -------------------------------------------------
# Convert churn values to 0/1
# -------------------------------------------------
def clean_churn_values(df):
    if "churn" not in df.columns:
        return df

    df["churn"] = (
        df["churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })
        .fillna(0)
        .astype(int)
    )

    return df


# -------------------------------------------------
# Load files
# -------------------------------------------------
raw_df = load_data("raw_data.csv")
stats_df = load_data("descriptive_stats.csv")

if raw_df is None:
    st.warning("Run python app.py first.")
    st.stop()

raw_df = ensure_churn_column(raw_df)
raw_df = clean_churn_values(raw_df)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

total = len(raw_df)

if "churn" in raw_df.columns:
    churn_count = int(raw_df["churn"].sum())
    churn_rate = raw_df["churn"].mean()
else:
    churn_count = 0
    churn_rate = 0

col1.metric("Total Customers", f"{total:,}")
col2.metric("Churn Count", f"{churn_count:,}")
col3.metric("Churn Rate", f"{churn_rate:.1%}")
col4.metric("Features", raw_df.shape[1])

st.divider()

# -------------------------------------------------
# Charts
# -------------------------------------------------
st.subheader("🖼️ Visualizations")

charts = [
    "churn_distribution.png",
    "analytics_dashboard.png",
    "correlation_heatmap.png",
    "roc_curves.png",
    "confusion_matrices.png",
    "model_comparison.png"
]

for chart in charts:
    path = os.path.join(DATA_DIR, chart)
    if os.path.exists(path):
        st.image(path, use_container_width=True)

for file in os.listdir(DATA_DIR):
    if file.startswith("feature_importance_") and file.endswith(".png"):
        st.image(os.path.join(DATA_DIR, file), use_container_width=True)

st.divider()

# -------------------------------------------------
# Tables
# -------------------------------------------------
st.subheader("🔍 Raw Data")
st.dataframe(raw_df.head(100), use_container_width=True)

if stats_df is not None:
    st.subheader("📐 Descriptive Statistics")
    st.dataframe(stats_df, use_container_width=True)