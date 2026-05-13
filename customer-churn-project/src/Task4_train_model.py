"""
Task 4: Train Models
- Random Forest
- XGBoost
- LightGBM
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


def get_models():
    """Return dict of baseline ML models."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
        ),
    }


def train_all(X_train, y_train) -> dict:
    """Train all models and return fitted models dict."""
    models = get_models()
    trained = {}
    for name, model in models.items():
        print(f"[Task 4] Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"[Task 4] {name} trained ✓")
    return trained


def run(X_train, y_train):
    return train_all(X_train, y_train)


if __name__ == "__main__":
    from Task1_data_preprocessing import run as preprocess
    from Task2_data_wrangling import run as wrangle
    from Task3_split_data import run as split

    preprocess()
    wrangle()
    X_train, X_test, y_train, y_test = split()
    trained_models = train_all(X_train, y_train)