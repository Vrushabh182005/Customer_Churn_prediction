"""
Task 5: Hyperparameter Tuning
- GridSearchCV on all 3 models
- 5-fold cross-validation
- Optimizing for ROC-AUC
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


PARAM_GRIDS = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [-1, 10],
            "learning_rate": [0.05, 0.1],
        },
    },
}


def tune_all(X_train, y_train, cv=3, scoring="roc_auc") -> dict:
    """Run GridSearchCV for all models. Returns dict of best estimators."""
    best_models = {}
    for name, config in PARAM_GRIDS.items():
        print(f"[Task 5] Tuning {name} (cv={cv}, scoring={scoring})...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"[Task 5] {name} best params: {grid.best_params_}")
        print(f"[Task 5] {name} best CV {scoring}: {grid.best_score_:.4f}")
    return best_models


def run(X_train, y_train):
    return tune_all(X_train, y_train)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from Task1_data_preprocessing import run as preprocess
    from Task2_data_wrangling import run as wrangle
    from Task3_split_data import run as split

    preprocess()
    wrangle()
    X_train, X_test, y_train, y_test = split()
    best = tune_all(X_train, y_train)