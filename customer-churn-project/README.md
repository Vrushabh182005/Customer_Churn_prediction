# 📉 Customer Churn Prediction

A complete, beginner-friendly ML project that predicts whether a customer will churn. Includes data preprocessing, model training, hyperparameter tuning, evaluation, a **FastAPI** backend, and a **Streamlit** frontend.

---

## 🗂️ Project Structure

```
customer-churn-project/
├── app.py                          # Entry point (runs full pipeline)
├── requirements.txt
├── README.md
├── .gitignore
├── data/                           # Raw & processed data (CSV files)
├── models/                         # Saved trained models
├── src/
│   ├── Task1_data_preprocessing.py
│   ├── Task2_data_wrangling.py
│   ├── Task3_split_data.py
│   ├── Task4_train_model.py
│   ├── Task5_hyperparameter_tuning.py
│   ├── Task6_evaluate_model.py
│   ├── Task7_save_model.py
│   ├── Task8_visualization.py
│   └── main.py
├── api/
│   └── server.py                   # FastAPI backend
├── frontend/
│   ├── app_ui.py                   # Streamlit main page
│   └── pages/
│       ├── 1_Analytics.py
│       ├── 2_Predict.py
│       └── 3_Model_Report.py
└── postman/
    └── ChurnAPI.postman_collection.json
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone or unzip the project
cd customer-churn-project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 — Run the full ML pipeline
```bash
python app.py
```
This will:
- Generate synthetic churn data
- Preprocess & wrangle data
- Train 3 ML models (Random Forest, XGBoost, LightGBM)
- Tune hyperparameters
- Evaluate & compare models
- Save the best model to `models/`
- Generate all visualizations to `data/`

### Step 2 — Start the FastAPI backend
```bash
uvicorn api.server:app --reload --port 8000
```
API docs available at: http://127.0.0.1:8000/docs

### Step 3 — Start the Streamlit frontend
```bash
streamlit run frontend/app_ui.py
```
Opens at: http://localhost:8501

---

## 📬 Postman

Import `postman/ChurnAPI.postman_collection.json` into Postman to test all API endpoints.

---

## 📊 ML Models Used

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting (high performance) |
| LightGBM | Fast gradient boosting by Microsoft |

---

## 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🧠 ML Pipeline Tasks

| Task | File | Description |
|------|------|-------------|
| 1 | Task1_data_preprocessing.py | Missing values, encoding, duplicates |
| 2 | Task2_data_wrangling.py | Feature engineering, outliers, normalization |
| 3 | Task3_split_data.py | Train/test split with stratification |
| 4 | Task4_train_model.py | Train 3 ML algorithms |
| 5 | Task5_hyperparameter_tuning.py | GridSearchCV tuning |
| 6 | Task6_evaluate_model.py | Metrics & comparison |
| 7 | Task7_save_model.py | Save/load best model |
| 8 | Task8_visualization.py | All charts & plots |

---

## 👨‍💻 Tech Stack
- **Python 3.10+**
- **scikit-learn**, **XGBoost**, **LightGBM**
- **FastAPI** + Uvicorn
- **Streamlit**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Plotly**