"""
Task 6: Evaluate Models
- Accuracy, Precision, Recall, F1, ROC-AUC
- Model comparison table
- Identify best model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Return metrics dict for a single model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def evaluate_all(trained_models: dict, X_test, y_test, output_dir: str = "data") -> tuple:
    """
    Evaluate all models. Returns (results_df, best_model_name, best_model).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for name, model in trained_models.items():
        m = evaluate_model(name, model, X_test, y_test)
        results.append(m)
        print(f"[Task 6] {name}: Acc={m['Accuracy']}, F1={m['F1 Score']}, AUC={m['ROC-AUC']}")

        # Detailed classification report
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(f"\n[Task 6] Model comparison saved → {output_dir}/model_comparison.csv")

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n[Task 6] ✅ Best model: {best_name} (ROC-AUC={results_df.iloc[0]['ROC-AUC']})")
    return results_df, best_name, best_model


def run(trained_models, X_test, y_test):
    return evaluate_all(trained_models, X_test, y_test)


if __name__ == "__main__":
    print("Run via main.py or app.py")