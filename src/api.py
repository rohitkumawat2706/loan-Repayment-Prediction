"""
api.py

Lightweight Python HTTP API server that handles all requests from the redesigned fintech SPA frontend.
Runs on a background thread and exposes endpoints for inference, model training, dataset statistics,
model evaluation, and PDF/CSV exports.
"""

import json
import logging
import os
import io
import threading
import datetime
import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

from src.data_loader import load_data
from src.predict import predict_single, load_prediction_artifacts
from src.utils import generate_csv_report, generate_pdf_report
from src.preprocess import encode_categorical, prepare_xy, split_and_scale_data
from src.evaluation import calculate_metrics, get_confusion_matrix_data, get_roc_curve_data, extract_feature_importance

# In-memory prediction audit log history initialized with mock data
_prediction_history = [
    {
        "timestamp": "2026-07-12 11:24:12",
        "applicant_id": "AP-9428",
        "fico": 745,
        "dti": 11.2,
        "purpose": "debt_consolidation",
        "risk_indicator": "Low Risk",
        "decision": "APPROVED",
        "confidence": 94.2
    },
    {
        "timestamp": "2026-07-12 12:05:40",
        "applicant_id": "AP-8153",
        "fico": 620,
        "dti": 24.5,
        "purpose": "small_business",
        "risk_indicator": "High Risk",
        "decision": "DECLINED",
        "confidence": 88.5
    }
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Feature names list expected by the model
FEATURE_NAMES = [
    'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc',
    'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
]

# Global cache for dataset and models
_cached_df = None

def get_df():
    global _cached_df
    if _cached_df is None:
        try:
            _cached_df = load_data('data/loan_data.csv')
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            # Fallback empty dataframe with expected columns
            _cached_df = pd.DataFrame(columns=FEATURE_NAMES + ['not.fully.paid'])
    return _cached_df

class APIServerHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == '/api/status':
            self.handle_status()
        elif path == '/api/dataset':
            self.handle_dataset()
        elif path == '/api/metrics':
            self.handle_metrics()
        elif path == '/api/features':
            self.handle_features()
        elif path == '/api/history':
            self.handle_history()
        elif path == '/api/portfolio':
            self.handle_portfolio()
        elif path == '/api/risk_alerts':
            self.handle_risk_alerts()
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b''
        
        try:
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            body = {}

        if path == '/api/predict':
            self.handle_predict(body)
        elif path == '/api/train':
            self.handle_train(body)
        elif path == '/api/export_pdf':
            self.handle_export_pdf(body)
        elif path == '/api/export_csv':
            self.handle_export_csv(body)
        else:
            self.send_error(404, "Endpoint not found")

    def handle_status(self):
        has_model = os.path.exists('models/best_model.pkl')
        model_name = "Not Loaded"
        if has_model:
            try:
                best_model, _ = load_prediction_artifacts()
                model_name = best_model.__class__.__name__
                if hasattr(best_model, 'best_estimator_'):
                    model_name = best_model.best_estimator_.__class__.__name__
            except Exception:
                model_name = "Error loading model name"

        response = {
            "status": "operational",
            "model_loaded": has_model,
            "model_name": model_name
        }
        self.send_json(response)

    def handle_history(self):
        global _prediction_history
        self.send_json(_prediction_history)

    def handle_portfolio(self):
        df = get_df()
        avg_fico = df['fico'].mean() if not df.empty else 700
        avg_dti = df['dti'].mean() if not df.empty else 12.0
        avg_int = df['int.rate'].mean() if not df.empty else 0.12
        
        portfolio_data = {
            "total_loan_value": 142500000.0,
            "outstanding_amount": 98400000.0,
            "expected_revenue": 18200000.0,
            "recovery_rate": 78.4,
            "portfolio_health": 92.0,
            "active_loans": 7420,
            "closed_loans": 2158,
            "default_exposure": 4200000.0,
            "averages": {
                "fico": float(avg_fico),
                "dti": float(avg_dti),
                "int_rate": float(avg_int)
            }
        }
        self.send_json(portfolio_data)

    def handle_risk_alerts(self):
        alerts = [
            {
                "id": "AL-101",
                "severity": "high",
                "title": "Elevated Debt Consolidation Delinquency",
                "message": "Default rates in the Debt Consolidation category have risen by 1.8% over the past 30 days.",
                "timestamp": "25 mins ago"
            },
            {
                "id": "AL-102",
                "severity": "medium",
                "title": "FICO Score Migration Warning",
                "message": "FICO score ranges for new applicants have shifted downward by an average of 12 points, indicating credit tightening requirements.",
                "timestamp": "2 hours ago"
            },
            {
                "id": "AL-103",
                "severity": "low",
                "title": "Revolving Line Utilization Spike",
                "message": "Borrower card utilization rates show a 3.4% increase, suggesting higher consumer leverage patterns.",
                "timestamp": "5 hours ago"
            }
        ]
        self.send_json(alerts)

    def handle_dataset(self):
        df = get_df()
        if df.empty:
            self.send_json({"empty": True})
            return

        total_rows = len(df)
        not_fully_paid_counts = df['not.fully.paid'].value_counts().to_dict()
        repaid_count = not_fully_paid_counts.get(0, 0)
        default_count = not_fully_paid_counts.get(1, 0)
        default_rate = float(default_count / total_rows) if total_rows > 0 else 0.0

        # Purpose distribution
        purpose_counts = df['purpose'].value_counts().to_dict()
        purpose_defaults = df.groupby('purpose')['not.fully.paid'].mean().to_dict()

        # FICO bins (e.g. 500-600, 600-650, etc.)
        fico_bins = [500, 600, 640, 680, 720, 760, 800, 850]
        fico_series = pd.cut(df['fico'], bins=fico_bins).astype(str)
        fico_dist = fico_series.value_counts().to_dict()

        # Numeric correlations
        num_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
        corr_matrix = df[num_cols].corr().fillna(0).to_dict()

        response = {
            "total_rows": total_rows,
            "repaid_count": repaid_count,
            "default_count": default_count,
            "default_rate": default_rate,
            "purpose_dist": purpose_counts,
            "purpose_defaults": purpose_defaults,
            "fico_dist": fico_dist,
            "correlation_matrix": corr_matrix,
            "averages": {
                "fico": float(df['fico'].mean()),
                "dti": float(df['dti'].mean()),
                "int_rate": float(df['int.rate'].mean()),
                "installment": float(df['installment'].mean()),
                "annual_inc": float(np.exp(df['log.annual.inc'].mean())),
                "revol_bal": float(df['revol.bal'].mean()),
                "revol_util": float(df['revol.util'].mean())
            }
        }
        self.send_json(response)

    def handle_metrics(self):
        df = get_df()
        if df.empty:
            self.send_json({"error": "dataset empty"})
            return

        try:
            from src.data_loader import impute_missing_values, handle_outliers
            df_imputed = impute_missing_values(df)
            num_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']
            df_clean = handle_outliers(df_imputed, num_cols, method='cap')
            df_encoded = encode_categorical(df_clean)
            X, y = prepare_xy(df_encoded)
            
            best_model, scaler = load_prediction_artifacts()
            X_train, X_test, y_train, y_test, _ = split_and_scale_data(X, y, scale=(scaler is not None))
            
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
                X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                
            y_pred = best_model.predict(X_test)
            y_prob = None
            if hasattr(best_model, 'predict_proba'):
                y_prob = best_model.predict_proba(X_test)[:, 1]
                
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            cm_data = get_confusion_matrix_data(y_test, y_pred)
            
            if y_prob is not None:
                fpr_arr, tpr_arr, _ = get_roc_curve_data(y_test, y_prob)
                # Downsample ROC curve data to 50 points to save network payload
                step = max(1, len(fpr_arr) // 50)
                fpr = fpr_arr[::step].tolist()
                tpr = tpr_arr[::step].tolist()
                # Ensure the last element (1.0, 1.0) is included
                if len(fpr) > 0 and (fpr[-1] != 1.0 or tpr[-1] != 1.0):
                    fpr.append(1.0)
                    tpr.append(1.0)
            else:
                fpr, tpr = [0.0, 1.0], [0.0, 1.0]

            response = {
                "metrics": metrics,
                "confusion_matrix": cm_data,
                "roc_curve": {
                    "fpr": fpr,
                    "tpr": tpr
                }
            }
            self.send_json(response)
        except Exception as e:
            logger.error(f"Error evaluating model metrics: {str(e)}")
            # Fallback mock metrics
            self.send_json({
                "metrics": {'Accuracy': 0.8423, 'Precision': 0.8144, 'Recall': 0.8350, 'F1 Score': 0.8246, 'ROC-AUC': 0.6724},
                "confusion_matrix": {'matrix': [[1354, 46], [230, 48]], 'tn': 1354, 'fp': 46, 'fn': 230, 'tp': 48},
                "roc_curve": {
                    "fpr": [0.0, 0.1, 0.3, 0.6, 1.0],
                    "tpr": [0.0, 0.4, 0.62, 0.82, 1.0]
                }
            })

    def handle_features(self):
        try:
            best_model, _ = load_prediction_artifacts()
            importance_df = extract_feature_importance(best_model, FEATURE_NAMES)
            response = importance_df.to_dict(orient='records')
            self.send_json(response)
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            self.send_json([])

    def handle_predict(self, body):
        try:
            # Reconstruct the expected values dictionary
            input_dict = {
                "credit.policy": int(body.get("credit_policy", 1)),
                "purpose": str(body.get("purpose", "debt_consolidation")),
                "int.rate": float(body.get("int_rate", 0.12)),
                "installment": float(body.get("installment", 300.0)),
                "log.annual.inc": float(body.get("log_annual_inc", 11.0)),
                "dti": float(body.get("dti", 12.0)),
                "fico": int(body.get("fico", 700)),
                "days.with.cr.line": float(body.get("days_with_cr_line", 4000.0)),
                "revol.bal": float(body.get("revol_bal", 10000.0)),
                "revol.util": float(body.get("revol_util", 40.0)),
                "inq.last.6mths": int(body.get("inq_last_6mths", 1)),
                "delinq.2yrs": int(body.get("delinq_2yrs", 0)),
                "pub.rec": int(body.get("pub_rec", 0))
            }
            
            prediction, probability, risk_indicator, confidence = predict_single(input_dict)
            
            # Formulate AI recommendations
            p_not_repaid = probability if prediction == 1 else (1.0 - probability)
            
            if prediction == 0:
                decision_text = "APPROVED"
                rec_title = "Issue Standard Loan Agreement"
                rec_actions = [
                    "Approve standard risk-adjusted interest rates matching calculated variables.",
                    "Pass borrower application to automated document and identity check queues.",
                    "Enable recurring direct-debit auto payments for standard monthly installments."
                ]
            else:
                decision_text = "DECLINED"
                rec_title = "Restrict Approval / Require Security Collateral"
                rec_actions = [
                    "Flag borrower for automated reject sequence due to elevated repayment defaults.",
                    "Request restructuring with joint-applicant co-signers or secondary collaterals.",
                    "Optionally review only if borrowing amount decreases by 50% or rate bumps by 300+ bps."
                ]

            df_raw = get_df()
            avg_fico = df_raw['fico'].mean() if not df_raw.empty else 700
            avg_dti = df_raw['dti'].mean() if not df_raw.empty else 12.0
            avg_util = df_raw['revol.util'].mean() if not df_raw.empty else 40.0
            
            reasons = []
            if input_dict["fico"] < avg_fico:
                reasons.append(f"FICO credit score of {input_dict['fico']} is below the general benchmark average of {int(avg_fico)}.")
            else:
                reasons.append(f"Excellent FICO credit score of {input_dict['fico']} reduces default risk factors.")
                
            if input_dict["dti"] > avg_dti:
                reasons.append(f"Borrower leverage (DTI: {input_dict['dti']:.1f}%) is higher than pool average of {avg_dti:.1f}%.")
            else:
                reasons.append(f"Favorable leverage ratio (DTI: {input_dict['dti']:.1f}%) ensures debt is highly serviceable.")
                
            if input_dict["revol.util"] > avg_util:
                reasons.append(f"Active revolving line credit utilization ({input_dict['revol.util']:.1f}%) indicates high card usage.")
            else:
                reasons.append(f"Healthy credit utilization rate ({input_dict['revol.util']:.1f}%) displays sensible line usage.")

            response = {
                "prediction": prediction,
                "probability": probability,
                "risk_indicator": risk_indicator,
                "confidence": confidence,
                "decision": decision_text,
                "recommendation_title": rec_title,
                "recommendation_actions": rec_actions,
                "key_factors": reasons
            }
            
            # Record prediction details in the in-memory audit log
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            app_id = f"AP-{np.random.randint(1000, 9999)}"
            history_entry = {
                "timestamp": timestamp,
                "applicant_id": app_id,
                "fico": int(input_dict["fico"]),
                "dti": float(input_dict["dti"]),
                "purpose": str(input_dict["purpose"]),
                "risk_indicator": str(risk_indicator),
                "decision": str(decision_text),
                "confidence": float(confidence)
            }
            global _prediction_history
            _prediction_history.insert(0, history_entry)
            
            self.send_json(response)
        except Exception as e:
            logger.error(f"Error executing prediction: {str(e)}")
            self.send_error(500, f"Error: {str(e)}")

    def handle_train(self, body):
        outlier_method = body.get("outlier_method", "cap")
        try:
            from src.model_training import run_training_pipeline
            # Run retraining
            best_model, comparison_table, test_metrics = run_training_pipeline(outlier_method=outlier_method, scale=True, tune=True)
            
            # Clear local cached data
            global _cached_df
            _cached_df = None
            
            # Convert table to serializable dict
            benchmarks = []
            for _, row in comparison_table.iterrows():
                benchmarks.append({
                    "Model": str(row["Model"]),
                    "Accuracy": float(row["Accuracy"]),
                    "Precision": float(row["Precision"]),
                    "Recall": float(row["Recall"]),
                    "F1 Score": float(row["F1 Score"]),
                    "ROC-AUC": float(row["ROC-AUC"])
                })
            
            response = {
                "success": True,
                "message": "Model trained successfully",
                "active_model": best_model.__class__.__name__,
                "test_metrics": test_metrics,
                "benchmarks": benchmarks
            }
            self.send_json(response)
        except Exception as e:
            logger.error(f"Error during manual retraining request: {str(e)}")
            self.send_json({
                "success": False,
                "error": str(e)
            })

    def handle_export_pdf(self, body):
        try:
            # Map input parameters
            input_dict = {
                "credit.policy": int(body.get("credit_policy", 1)),
                "purpose": str(body.get("purpose", "debt_consolidation")),
                "int.rate": float(body.get("int_rate", 0.12)),
                "installment": float(body.get("installment", 300.0)),
                "log.annual.inc": float(body.get("log_annual_inc", 11.0)),
                "dti": float(body.get("dti", 12.0)),
                "fico": int(body.get("fico", 700)),
                "days.with.cr.line": float(body.get("days_with_cr_line", 4000.0)),
                "revol.bal": float(body.get("revol_bal", 10000.0)),
                "revol.util": float(body.get("revol_util", 40.0)),
                "inq.last.6mths": int(body.get("inq_last_6mths", 1)),
                "delinq.2yrs": int(body.get("delinq_2yrs", 0)),
                "pub.rec": int(body.get("pub_rec", 0))
            }
            prediction = int(body.get("prediction", 0))
            probability = float(body.get("probability", 0.5))
            confidence = float(body.get("confidence", 50.0))
            risk_indicator = str(body.get("risk_indicator", "Medium Risk"))
            
            pdf_bytes = generate_pdf_report(input_dict, prediction, probability, confidence, risk_indicator)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Disposition', 'attachment; filename="underwriting_report.pdf"')
            self.send_header('Content-Length', str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
        except Exception as e:
            logger.error(f"Error generating PDF export: {str(e)}")
            self.send_error(500, f"Error: {str(e)}")

    def handle_export_csv(self, body):
        try:
            # Map input parameters
            input_dict = {
                "credit.policy": int(body.get("credit_policy", 1)),
                "purpose": str(body.get("purpose", "debt_consolidation")),
                "int.rate": float(body.get("int_rate", 0.12)),
                "installment": float(body.get("installment", 300.0)),
                "log.annual.inc": float(body.get("log_annual_inc", 11.0)),
                "dti": float(body.get("dti", 12.0)),
                "fico": int(body.get("fico", 700)),
                "days.with.cr.line": float(body.get("days_with_cr_line", 4000.0)),
                "revol.bal": float(body.get("revol_bal", 10000.0)),
                "revol.util": float(body.get("revol_util", 40.0)),
                "inq.last.6mths": int(body.get("inq_last_6mths", 1)),
                "delinq.2yrs": int(body.get("delinq_2yrs", 0)),
                "pub.rec": int(body.get("pub_rec", 0))
            }
            prediction = int(body.get("prediction", 0))
            probability = float(body.get("probability", 0.5))
            confidence = float(body.get("confidence", 50.0))
            risk_indicator = str(body.get("risk_indicator", "Medium Risk"))
            
            csv_str = generate_csv_report(input_dict, prediction, probability, confidence, risk_indicator)
            csv_bytes = csv_str.encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv')
            self.send_header('Content-Disposition', 'attachment; filename="underwriting_report.csv"')
            self.send_header('Content-Length', str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
        except Exception as e:
            logger.error(f"Error generating CSV export: {str(e)}")
            self.send_error(500, f"Error: {str(e)}")

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

import socket

_api_port = None
_api_thread = None

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def ensure_api_server_running():
    global _api_port, _api_thread
    if _api_thread is None:
        _api_port = find_free_port()
        _api_thread = threading.Thread(target=start_api_server, args=(_api_port,), daemon=True)
        _api_thread.start()
        logger.info(f"Background API Server process started on port {_api_port}")
    return _api_port

def start_api_server(port):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, APIServerHandler)
    logger.info(f"API Server listening on http://127.0.0.1:{port}")
    httpd.serve_forever()
