"""
Task 7: Save & Load Model
- Save best model
- Save metadata
- Load model utility
"""

import joblib
import json
import os


def save_model(model, model_name: str, feature_cols: list, output_dir: str = "models"):
    """Save trained model + metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    print(f"[Task 7] Model saved → {model_path}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "feature_columns": list(feature_cols),
        "n_features": len(feature_cols)
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Task 7] Metadata saved → {metadata_path}")
    return model_path


def load_model(output_dir: str = "models"):
    """Load saved model + metadata."""
    model_path = os.path.join(output_dir, "best_model.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run python app.py first.")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata not found.")

    model = joblib.load(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"[Task 7] Loaded {metadata['model_name']} ({metadata['n_features']} features)")
    return model, metadata


def run(model, model_name, feature_cols):
    return save_model(model, model_name, feature_cols)


if __name__ == "__main__":
    print("Run from main.py or app.py")