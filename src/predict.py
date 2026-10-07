"""
predict.py

This module exposes wrapper functions to make predictions on single observations or batch data,
integrating feature encoding, standard scaling, and custom prediction confidence scores.
"""

import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

from src.preprocess import encode_categorical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache objects in memory to avoid repeated disk reads
_model = None
_scaler = None

def load_prediction_artifacts(model_path: str = 'models/best_model.pkl', scaler_path: str = 'models/scaler.pkl') -> Tuple[Any, Optional[Any]]:
    """
    Loads saved model and scaler objects, caching them to avoid redundant file loads.
    
    Args:
        model_path (str): File path to saved model pickle.
        scaler_path (str): File path to saved scaler pickle.
        
    Returns:
        Tuple[model, scaler]: Loaded classifier model and scaler (None if scaler not found).
    """
    global _model, _scaler
    
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run training first.")
        logger.info(f"Loading prediction model from {model_path}...")
        _model = joblib.load(model_path)
        
    if _scaler is None:
        if os.path.exists(scaler_path):
            logger.info(f"Loading feature scaler from {scaler_path}...")
            _scaler = joblib.load(scaler_path)
        else:
            logger.info("No scaler file found; proceeding with unscaled predictions.")
            _scaler = None
            
    return _model, _scaler

def predict_single(input_data: Dict[str, Any], model_path: str = 'models/best_model.pkl', scaler_path: str = 'models/scaler.pkl') -> Tuple[int, float, str, float]:
    """
    Predicts loan repayment status for a single applicant entry.
    
    Args:
        input_data (Dict[str, Any]): Dictionary of input feature values (keys match raw column names).
        model_path (str): File path to saved model.
        scaler_path (str): File path to saved scaler.
        
    Returns:
        Tuple[prediction, probability, risk_indicator, confidence]:
            - prediction (int): 0 for Fully Repaid, 1 for Not Fully Repaid.
            - probability (float): Model probability score of the class.
            - risk_indicator (str): Risk status indicator ("Low Risk", "Medium Risk", "High Risk").
            - confidence (float): Prediction confidence percentage (0 to 100).
    """
    try:
        model, scaler = load_prediction_artifacts(model_path, scaler_path)
        
        # Convert dictionary to single-row DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Encode categorical columns
        df_processed = encode_categorical(df_input)
        
        # Scale values if scaling was used in training
        if scaler is not None:
            # We must fit structure names to match standard scaling columns
            df_scaled = scaler.transform(df_processed)
            df_processed = pd.DataFrame(df_scaled, columns=df_processed.columns)
            
        # Predict class
        prediction = int(model.predict(df_processed)[0])
        
        # Predict probability
        probability = 0.5
        confidence = 50.0
        risk_indicator = "Medium Risk"
        
        if hasattr(model, 'predict_proba'):
            prob_array = model.predict_proba(df_processed)[0]
            # prob_array[1] is the probability of class 1 ("Not Fully Repaid")
            # prob_array[0] is the probability of class 0 ("Fully Repaid")
            p_not_repaid = float(prob_array[1])
            
            # Prediction class is based on 0.5 threshold by default
            if prediction == 0:
                probability = float(prob_array[0])
                confidence = float(probability * 100.0)
            else:
                probability = p_not_repaid
                confidence = float(probability * 100.0)
                
            # Determine Risk category based on the probability of not paying back
            if p_not_repaid < 0.15:
                risk_indicator = "Low Risk"
            elif p_not_repaid < 0.35:
                risk_indicator = "Medium Risk"
            else:
                risk_indicator = "High Risk"
        else:
            # Decission tree or models without predict_proba fallback
            if prediction == 0:
                risk_indicator = "Low Risk"
                confidence = 100.0
            else:
                risk_indicator = "High Risk"
                confidence = 100.0
                
        return prediction, probability, risk_indicator, confidence
        
    except Exception as e:
        logger.error(f"Error during single prediction pipeline: {str(e)}")
        raise e
