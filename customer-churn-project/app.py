"""
app.py — Root entry point for the Customer Churn Prediction project.
Run: python app.py
"""

import sys
import os

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from main import run_pipeline

if __name__ == "__main__":
    run_pipeline(data_dir="data", model_dir="models", tune=True)