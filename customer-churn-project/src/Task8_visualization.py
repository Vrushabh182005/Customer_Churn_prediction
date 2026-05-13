"""
Task 8: Data Visualization
- Churn distribution
- Correlation heatmap
- Feature importance
- ROC curves
- Confusion matrices
- Analytics charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import warnings
warnings.filterwarnings("ignore")

COLORS = {
    "churn": "#e74c3c",
    "no_churn": "#2ecc71",
    "primary": "#2c3e50",
    "accent": "#3498db"
}

sns.set_theme(style="whitegrid", palette="muted")


# =====================================================
# Helper: Ensure target column is named churn
# =====================================================
def ensure_churn_column(df):
    possible_targets = [
        "churn", "exited", "attrition",
        "target", "label", "churn_value"
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for col in possible_targets:
        if col in lower_map:
            real_col = lower_map[col]
            if real_col != "churn":
                df = df.rename(columns={real_col: "churn"})
            return df

    raise KeyError("No target column found.")


# =====================================================
# Charts
# =====================================================
def plot_churn_distribution(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Churn Distribution", fontsize=16, fontweight="bold")

    churn_counts = df["churn"].value_counts().sort_index()

    labels = ["No Churn", "Churn"]
    colors = [COLORS["no_churn"], COLORS["churn"]]

    axes[0].pie(
        churn_counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )
    axes[0].set_title("Churn Proportion")

    sns.countplot(
        x="churn",
        data=df,
        palette=colors,
        ax=axes[1]
    )
    axes[1].set_title("Churn Count")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/churn_distribution.png", dpi=120)
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    numeric_df = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        center=0,
        ax=ax
    )
    ax.set_title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names, model_name, output_dir):
    if not hasattr(model, "feature_importances_"):
        return

    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.nlargest(15).sort_values()

    plt.figure(figsize=(10, 7))
    imp.plot(kind="barh", color=COLORS["accent"])
    plt.title(f"Top Features - {model_name}")
    plt.tight_layout()

    name = model_name.lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/feature_importance_{name}.png", dpi=120)
    plt.close()


def plot_roc_curves(models, X_test, y_test, output_dir):
    plt.figure(figsize=(9, 7))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} ({score:.3f})")

    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=120)
    plt.close()


def plot_confusion_matrices(models, X_test, y_test, output_dir):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=120)
    plt.close()


def plot_model_comparison(results_df, output_dir):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.25

    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i*width, vals, width, label=row["Model"])

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=120)
    plt.close()


def plot_analytics(df, output_dir):
    plt.figure(figsize=(10, 6))

    if "churn" in df.columns:
        df["churn"].value_counts().plot(kind="bar")

    plt.title("Analytics Overview")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analytics_dashboard.png", dpi=120)
    plt.close()


# =====================================================
# Main Run
# =====================================================
def run(df, trained_models, X_test, y_test, results_df, best_model_name, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, "raw_data.csv")
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else df.copy()

    raw_df = ensure_churn_column(raw_df)
    df = ensure_churn_column(df)

    plot_churn_distribution(raw_df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_roc_curves(trained_models, X_test, y_test, output_dir)
    plot_confusion_matrices(trained_models, X_test, y_test, output_dir)
    plot_model_comparison(results_df, output_dir)
    plot_analytics(raw_df, output_dir)

    best_model = trained_models[best_model_name]
    plot_feature_importance(
        best_model,
        list(X_test.columns),
        best_model_name,
        output_dir
    )

    print(f"[Task 8] All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("Run via main.py or app.py")