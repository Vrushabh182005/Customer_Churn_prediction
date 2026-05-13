

import pandas as pd
import numpy as np
import os


DATASET_PATH = "data/customer_churn.csv"


def load_data(file_path=DATASET_PATH):
    """Load customer churn dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[Task 1] Dataset loaded → {file_path}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
 
    print(f"[Task 1] Original shape: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"[Task 1] After dedup: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert TotalCharges-like object numeric columns if needed
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_numeric(df[col], errors="ignore")
                df[col] = converted
            except:
                pass

    # Missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[Task 1] Missing values after fill: {df.isnull().sum().sum()}")

    # Rename target column if needed
    if "churn" not in df.columns:
        for possible in ["exited", "target", "label"]:
            if possible in df.columns:
                df.rename(columns={possible: "churn"}, inplace=True)

    # Binary mapping
    binary_map = {
        "yes": 1, "no": 0,
        "male": 1, "female": 0,
        "true": 1, "false": 0
    }

    for col in categorical_cols:
        unique_vals = df[col].astype(str).str.lower().unique()
        if len(unique_vals) == 2:
            df[col] = df[col].astype(str).str.lower().map(binary_map).fillna(df[col])

    # One-hot encode remaining object columns except IDs
    exclude_cols = ["customerid", "customer_id", "id"]
    remaining_cats = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in exclude_cols
    ]

    if remaining_cats:
        df = pd.get_dummies(df, columns=remaining_cats, drop_first=True)

    # Ensure churn is numeric
    if "churn" in df.columns:
        df["churn"] = pd.to_numeric(df["churn"], errors="coerce").fillna(0).astype(int)

    print(f"[Task 1] Final shape after encoding: {df.shape}")
    return df


def run(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_data()
    raw_df.to_csv(f"{output_dir}/raw_data.csv", index=False)
    print(f"[Task 1] Raw data saved → {output_dir}/raw_data.csv")

    processed_df = preprocess(raw_df.copy())
    processed_df.to_csv(f"{output_dir}/preprocessed_data.csv", index=False)
    print(f"[Task 1] Preprocessed data saved → {output_dir}/preprocessed_data.csv")

    return processed_df


if __name__ == "__main__":
    run()