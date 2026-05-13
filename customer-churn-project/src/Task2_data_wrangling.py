"""
Task 2: Data Wrangling
- Feature engineering
- Outlier handling (IQR method)
- Data normalization (MinMaxScaler)
- Descriptive statistics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def remove_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Cap outliers using IQR method (Winsorization)."""
    df = df.copy()
    for col in cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower, upper)
            if before > 0:
                print(f"[Task 2] Outliers capped in '{col}': {before} values")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful derived features."""
    df = df.copy()

    if "monthly_charges" in df.columns and "tenure_months" in df.columns:
        df["charges_per_month_ratio"] = (
            df["monthly_charges"] / (df["tenure_months"] + 1)
        ).round(4)

    if "customer_service_calls" in df.columns and "tenure_months" in df.columns:
        df["calls_per_month"] = (
            df["customer_service_calls"] / (df["tenure_months"] + 1)
        ).round(4)

    if "total_charges" in df.columns and "monthly_charges" in df.columns:
        df["charge_growth"] = (df["total_charges"] / (df["monthly_charges"] + 1)).round(4)

    print(f"[Task 2] Feature engineering done. New columns added.")
    return df


def normalize(df: pd.DataFrame, exclude_cols: list = None) -> tuple:
    """Apply MinMaxScaler to numeric columns, return (scaled_df, scaler)."""
    df = df.copy()
    exclude_cols = exclude_cols or []
    scale_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"[Task 2] Normalized {len(scale_cols)} numeric columns.")
    return df, scaler


def descriptive_stats(df: pd.DataFrame, output_dir: str = "data"):
    """Save descriptive statistics to CSV."""
    stats = df.describe(include="all").T
    stats["missing"] = df.isnull().sum()
    stats["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    stats.to_csv(f"{output_dir}/descriptive_stats.csv")
    print(f"[Task 2] Descriptive stats saved → {output_dir}/descriptive_stats.csv")
    return stats


def run(input_path="data/preprocessed_data.csv", output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    print(f"[Task 2] Loaded data: {df.shape}")

    # Outlier handling
    outlier_cols = ["monthly_charges", "total_charges", "age", "customer_service_calls"]
    df = remove_outliers_iqr(df, outlier_cols)

    # Feature engineering
    df = feature_engineering(df)

    # Descriptive stats (before normalization)
    descriptive_stats(df, output_dir)

    # Normalization (exclude target + id)
    df, scaler = normalize(df, exclude_cols=["churn", "customer_id"])

    df.to_csv(f"{output_dir}/wrangled_data.csv", index=False)
    print(f"[Task 2] Wrangled data saved → {output_dir}/wrangled_data.csv")
    return df, scaler


if __name__ == "__main__":
    run()