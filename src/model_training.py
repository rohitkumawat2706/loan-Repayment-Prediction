"""
model_training.py

This module contains the logic for training, comparing, tuning, and saving ML models.
It can also be run directly as a script.
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.data_loader import load_data, impute_missing_values, handle_outliers
from src.preprocess import encode_categorical, prepare_xy, split_and_scale_data, save_scaler
from src.evaluation import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate models for comparison.
    """
    return {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
        'AdaBoost (Decision Tree)': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
            learning_rate=0.5,
            random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.05, random_state=random_state),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=random_state)
    }

def compare_models(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Compares candidate classifiers using Stratified K-Fold cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv_folds (int): Number of folds for cross-validation.
        random_state (int): Random state seed.
        
    Returns:
        pd.DataFrame: A table of average cross-validation metrics for each model.
    """
    models = get_candidate_models(random_state)
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    comparison_results = []
    
    logger.info(f"Starting model comparison using {cv_folds}-Fold Stratified CV...")
    
    for name, model in models.items():
        try:
            logger.info(f"Evaluating {name}...")
            # Perform cross validation
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)
            
            # Record average metrics
            comparison_results.append({
                'Model': name,
                'Accuracy': float(np.mean(cv_results['test_accuracy'])),
                'Precision': float(np.mean(cv_results['test_precision'])),
                'Recall': float(np.mean(cv_results['test_recall'])),
                'F1 Score': float(np.mean(cv_results['test_f1'])),
                'ROC-AUC': float(np.mean(cv_results['test_roc_auc']))
            })
        except Exception as e:
            logger.error(f"Error evaluating model '{name}': {str(e)}")
            
    comparison_df = pd.DataFrame(comparison_results)
    logger.info("Model comparison completed.")
    return comparison_df

def tune_hyperparameters(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5, random_state: int = 42) -> Any:
    """
    Performs grid search to find the best hyperparameters for the selected model.
    
    Args:
        model_name (str): The name of the model ('Decision Tree', 'Random Forest', etc.)
        X_train (pd.DataFrame): Preprocessed training features.
        y_train (pd.Series): Training target.
        cv_folds (int): CV folds.
        random_state (int): Seed.
        
    Returns:
        Any: Best estimator from GridSearchCV.
    """
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    if model_name == 'Decision Tree':
        estimator = DecisionTreeClassifier(random_state=random_state)
        param_grid = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == 'Random Forest':
        estimator = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200, 400],
            'max_depth': [5, 8, 12, None],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == 'AdaBoost (Decision Tree)':
        estimator = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=random_state),
            random_state=random_state
        )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }
    elif model_name == 'Gradient Boosting':
        estimator = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        }
    else:  # XGBoost
        estimator = XGBClassifier(eval_metric='logloss', random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
    logger.info(f"Running GridSearchCV hyperparameter tuning for {model_name}...")
    grid_search = GridSearchCV(estimator, param_grid, scoring='recall', cv=kfold, n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train, y_train)
    logger.info(f"Grid search complete. Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def save_best_model(model: Any, file_path: str = 'models/best_model.pkl') -> None:
    """
    Saves the best trained estimator to disk.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Successfully saved best model to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save best model: {str(e)}")
        raise e

def run_training_pipeline(data_path: str = 'data/loan_data.csv', outlier_method: str = 'cap', scale: bool = False, tune: bool = True) -> Tuple[Any, pd.DataFrame, Dict[str, float]]:
    """
    Runs the complete ML pipeline: loads, cleans, preprocesses, splits, compares,
    optionally tunes, trains best model, and saves outputs.
    
    Args:
        data_path (str): File path to raw data CSV.
        outlier_method (str): 'cap', 'remove', or 'none'.
        scale (bool): Whether to scale features.
        tune (bool): Whether to run GridSearchCV tuning.
        
    Returns:
        Tuple[best_model, comparison_table, test_metrics]: Trained model, comparison metrics, and final test scores.
    """
    # 1. Load data
    df = load_data(data_path)
    
    # 2. Impute missing values
    df_imputed = impute_missing_values(df)
    
    # 3. Handle outliers
    num_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']
    df_clean = handle_outliers(df_imputed, num_cols, method=outlier_method)
    
    # 4. Encode purpose
    df_encoded = encode_categorical(df_clean)
    
    # 5. Split feature target
    X, y = prepare_xy(df_encoded)
    
    # 6. Train-test split
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, scale=scale)
    
    # 7. Save scaler if fit, otherwise clean up any existing scaler
    if scaler is not None:
        save_scaler(scaler)
    else:
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            logger.info(f"Removed old scaler at {scaler_path} as scaling is disabled.")
        
    # 8. Compare multiple models
    comparison_table = compare_models(X_train, y_train)
    
    # Identify the best performing model based on F1-Score or ROC-AUC
    # We will sort by ROC-AUC and select the top one
    best_row = comparison_table.sort_values(by='ROC-AUC', ascending=False).iloc[0]
    best_model_name = best_row['Model']
    logger.info(f"Top performing model in cross-validation: {best_model_name}")
    
    # 9. Tuning / Retraining best model
    if tune:
        best_model = tune_hyperparameters(best_model_name, X_train, y_train)
    else:
        models = get_candidate_models()
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        
    # 10. Fit on full training set (GridSearchCV does this automatically with refit=True)
    # Validate on test set
    y_pred = best_model.predict(X_test)
    y_prob = None
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
    test_metrics = calculate_metrics(y_test, y_pred, y_prob)
    logger.info(f"Test performance metrics for best model ({best_model_name}): {test_metrics}")
    
    # 11. Save the best model
    save_best_model(best_model)
    
    return best_model, comparison_table, test_metrics

if __name__ == '__main__':
    # Script entry point to run training locally
    logger.info("Executing training pipeline script...")
    run_training_pipeline()
    logger.info("Pipeline script execution finished successfully!")
