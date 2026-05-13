"""
Task 3: Split Data
- Remove ID/text columns
- Convert bools
- Stratified split
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split(df, target_col="churn", test_size=0.2, random_state=42):
    df = df.copy()

    # Drop common ID columns
    drop_cols = [
        c for c in df.columns
        if c.lower() in [
            "customerid", "customer_id", "id",
            "surname", "name", "rownumber"
        ]
    ]

    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[Task 3] Dropped columns: {drop_cols}")

    # Remove any remaining object/string columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)
        print(f"[Task 3] Removed text columns: {obj_cols}")

    # Convert bool to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"[Task 3] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run(input_path="data/wrangled_data.csv"):
    df = pd.read_csv(input_path)
    return split(df)


if __name__ == "__main__":
    run()