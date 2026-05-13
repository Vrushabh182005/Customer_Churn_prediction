

import sys
import os

# Ensure src/ is in path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from Task1_data_preprocessing import run as task1
from Task2_data_wrangling import run as task2
from Task3_split_data import run as task3
from Task4_train_model import run as task4
from Task5_hyperparameter_tuning import run as task5
from Task6_evaluate_model import run as task6
from Task7_save_model import run as task7
from Task8_visualization import run as task8


def run_pipeline(data_dir="data", model_dir="models", tune=True):
    print("\n" + "=" * 60)
    print("  CUSTOMER CHURN PREDICTION — ML PIPELINE")
    print("=" * 60)

    # Task 1: Preprocess
    print("\n📌 Task 1: Data Preprocessing")
    processed_df = task1(output_dir=data_dir)

    # Task 2: Wrangle
    print("\n📌 Task 2: Data Wrangling")
    wrangled_df, scaler = task2(
        input_path=f"{data_dir}/preprocessed_data.csv",
        output_dir=data_dir,
    )

    # Task 3: Split
    print("\n📌 Task 3: Split Data")
    X_train, X_test, y_train, y_test = task3(
        input_path=f"{data_dir}/wrangled_data.csv"
    )

    # Task 4: Train
    print("\n📌 Task 4: Train Models")
    trained_models = task4(X_train, y_train)

    # Task 5: Hyperparameter Tuning
    if tune:
        print("\n📌 Task 5: Hyperparameter Tuning (this may take ~1-2 min)")
        tuned_models = task5(X_train, y_train)
    else:
        print("\n📌 Task 5: Skipping tuning (tune=False)")
        tuned_models = trained_models

    # Task 6: Evaluate
    print("\n📌 Task 6: Evaluate Models")
    results_df, best_name, best_model = task6(tuned_models, X_test, y_test)

    # Task 7: Save Best Model
    print("\n📌 Task 7: Save Best Model")
    task7(best_model, best_name, X_train.columns.tolist())

    # Task 8: Visualizations
    print("\n📌 Task 8: Visualizations")
    task8(wrangled_df, tuned_models, X_test, y_test, results_df, best_name, data_dir)

    print("\n" + "=" * 60)
    print(f"  ✅ PIPELINE COMPLETE! Best model: {best_name}")
    print("=" * 60)
    print(f"  📁 Data & charts → {data_dir}/")
    print(f"  📁 Saved model   → {model_dir}/best_model.pkl")
    print("  🚀 Start API:      uvicorn api.server:app --reload --port 8000")
    print("  🎨 Start UI:       streamlit run frontend/app_ui.py")
    print("=" * 60 + "\n")

    return {
        "models": tuned_models,
        "best_name": best_name,
        "best_model": best_model,
        "results_df": results_df,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    run_pipeline()