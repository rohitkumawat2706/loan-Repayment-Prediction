"""
evaluation.py

This module handles calculation of classification metrics, confusion matrices, ROC curves,
and feature importance extraction.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes key classification metrics.
    
    Args:
        y_true (np.ndarray): True target labels.
        y_pred (np.ndarray): Predicted target labels.
        y_prob (np.ndarray, optional): Predicted class probabilities for positive class.
        
    Returns:
        Dict[str, float]: Dictionary containing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    metrics = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1 Score': float(f1_score(y_true, y_pred, zero_division=0)),
        'ROC-AUC': 0.5  # Default baseline if probabilities aren't provided
    }
    
    if y_prob is not None:
        try:
            metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {str(e)}")
            
    return metrics

def get_confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Generates confusion matrix structure for plotting/displaying.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        
    Returns:
        Dict[str, Any]: Confusion matrix elements (TN, FP, FN, TP).
    """
    cm = confusion_matrix(y_true, y_pred)
    # Handle edge case where only one class is present/predicted
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if len(np.unique(y_true)) == 1:
            val = np.unique(y_true)[0]
            if val == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)
                
    return {
        'matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

def get_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates false positive rate, true positive rate, and thresholds for the ROC curve.
    
    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: FPR, TPR, thresholds.
    """
    if y_prob is None:
        return np.array([]), np.array([]), np.array([])
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds

def extract_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """
    Extracts feature importances/coefficients from the trained model.
    
    Args:
        model (Any): Trained classifier model (Tree-based, AdaBoost, LogisticRegression, etc.).
        feature_names (list): List of feature name strings.
        
    Returns:
        pd.DataFrame: DataFrame containing feature names and importances, sorted descending.
    """
    importance = np.zeros(len(feature_names))
    
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            logger.info("Extracted feature importances using feature_importances_ attribute.")
        elif hasattr(model, 'coef_'):
            # Use absolute value of coefficients for linear models as absolute magnitude of importance
            importance = np.abs(model.coef_[0])
            logger.info("Extracted feature coefficients using coef_ attribute.")
        elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
            # In case model is a Meta-Estimator like Bagging or GridSearchCV
            importance = model.estimator.feature_importances_
            logger.info("Extracted feature importances from base estimator.")
        elif hasattr(model, 'best_estimator_'):
            # GridSearchCV / RandomizedSearchCV wrapper
            return extract_feature_importance(model.best_estimator_, feature_names)
        else:
            logger.warning("The model does not expose feature importances or coefficients.")
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        
    # Compile into DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return importance_df
