# 💳 FinSight: Loan Repayment Prediction & Underwriting Dashboard

FinSight is a production-grade machine learning system designed to assess applicant default risk. Built using Python, Scikit-Learn, and Streamlit, it features a modern Fintech dashboard inspired by Stripe, Vercel, and Microsoft Fluent UI, complete with interactive dashboards, prediction probability confidence scores, and downloadable applicant credit risk reports.

---

## 🚀 Key Features

* **Fintech Dashboard Interface:** Dynamic dark/light mode toggle with Stripe-style cards, modern typography, metrics overview, and responsive column layouts.
* **Modular Pipeline Architecture:** Clean decoupling of ingestion, missing value imputation, outlier handling, training, evaluation, and prediction.
* **Outlier Capping & Preprocessing:** Automatic handling of numeric outliers using the IQR method (capping values to bounds) and deterministic mapping for categorical purposes.
* **Grid Search CV Optimization:** Automatic comparison of candidate estimators (Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost) and hyperparameter tuning using GridSearchCV.
* **Interactive Diagnostics:** Interactive Plotly heatmaps for Confusion Matrices, line charts for ROC curves, and horizontal bar charts for feature importance.
* **Applicant Risk Underwriting:** Real-time risk status badges (Low Risk, Medium Risk, High Risk) and calculated confidence scores based on model class probabilities.
* **Exportable Reports:** One-click downloads to export applicant parameters and prediction details in CSV format or professional, auto-formatted ReportLab PDF report sheets.

---

## 📁 Project Directory Structure

```text
Loan-Repayment-Prediction/
├── data/
│   └── loan_data.csv          # Historic LendingClub credit dataset
├── models/
│   └── best_model.pkl         # Saved optimized classifier (Gradient Boosting)
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis notebook
├── src/                       # Production python modules
│   ├── data_loader.py         # Data loading, clean up, and outlier capping
│   ├── preprocess.py          # Value encoding, scaling, and train-test splits
│   ├── model_training.py      # CV evaluation and GridSearchCV tuning
│   ├── evaluation.py          # Metric calculation and feature importances
│   ├── predict.py             # Inference wrapper for single applicant prediction
│   └── utils.py               # Plotly charts and PDF/CSV report generation
├── app.py                     # Streamlit multi-tab Fintech dashboard app
├── requirements.txt           # Python library requirements list
├── .gitignore                 # Files and folders to exclude from git tracking
└── README.md                  # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/loan-repayment-prediction.git
   cd loan-repayment-prediction
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

### 1. Training the Model
To re-train the models, run the cross-validation comparison, tune hyperparameters, and save the best model artifact to disk:
```bash
python -m src.train
```

### 2. Launching the Web App
To start the interactive Fintech Streamlit dashboard locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📈 Model Performance CV Benchmark

Following 5-Fold Stratified Cross-Validation, candidate models were benchmarked:

| Model Class | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **83.1%** | **33.3%** | **5.7%** | **9.7%** | **65.0%** |
| Decision Tree | 84.6% | 0.0% | 0.0% | 0.0% | 50.0% |
| AdaBoost | 84.0% | 58.0% | 2.0% | 5.0% | 51.0% |
| Random Forest | 84.7% | 44.0% | 2.0% | 3.0% | 51.0% |

---

## 📷 Screenshots

### 1. Risk Analysis Dashboard
*(Insert screenshot of dashboard view here)*

### 2. Underwriting Decision Page
*(Insert screenshot of prediction outputs and PDF download buttons here)*

---

## ℹ️ Disclaimer
This predictive model is for educational and portfolio demonstration purposes. Financial loan approval decisions should incorporate independent credit underwriting analysis in compliance with internal policy and regulations.
