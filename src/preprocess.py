"""
preprocess.py

This module handles feature encoding, scaling, and train-test split logic.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Deterministic purpose mapping (alphabetical ordering matching LabelEncoder.fit_transform)
PURPOSE_MAP = {
    "all_other": 0,
    "credit_card": 1,
    "debt_consolidation": 2,
    "educational": 3,
    "home_improvement": 4,
    "major_purchase": 5,
    "small_business": 6
}

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the categorical 'purpose' column to numerical values using the standard mapping.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with encoded categorical column.
    """
    df_encoded = df.copy()
    if 'purpose' in df_encoded.columns:
        # Check if 'purpose' is text/object and needs mapping
        if df_encoded['purpose'].dtype == 'object' or isinstance(df_encoded['purpose'].iloc[0], str):
            logger.info("Encoding 'purpose' column mapping text values to integers.")
            df_encoded['purpose'] = df_encoded['purpose'].map(PURPOSE_MAP)
            # Fill missing mapping values with 'all_other' (0) if any unmapped value appears
            df_encoded['purpose'] = df_encoded['purpose'].fillna(0).astype(int)
    return df_encoded

def prepare_xy(df: pd.DataFrame, target_column: str = 'not.fully.paid') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features (X) and target (y).
    
    Args:
        df (pd.DataFrame): Cleaned and encoded DataFrame.
        target_column (str): The column name representing target labels.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and Target (y).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    logger.info(f"Prepared X with shape {X.shape} and y with shape {y.shape}")
    return X, y

def split_and_scale_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42, scale: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[StandardScaler]]:
    """
    Splits the feature and target matrices into train/test sets and optionally scales features.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.
        scale (bool): Whether to fit and apply StandardScaler.
        
    Returns:
        Tuple[X_train, X_test, y_train, y_test, scaler]: Split datasets and scaler (None if scale=False).
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Split data into train (size={len(X_train)}) and test (size={len(X_test)}) sets.")
    
    scaler = None
    if scale:
        logger.info("Applying StandardScaler to numerical features...")
        scaler = StandardScaler()
        # Scale only training numerical features, fit scaler on X_train
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve feature names/structure
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        logger.info("Standard scaling completed successfully.")
        
    return X_train, X_test, y_train, y_test, scaler

def save_scaler(scaler: StandardScaler, file_path: str = 'models/scaler.pkl') -> None:
    """
    Persists the fitted StandardScaler object.
    
    Args:
        scaler (StandardScaler): Scaler to save.
        file_path (str): File path to save standard scaler.
    """
    try:
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(scaler, file_path)
        logger.info(f"Scaler saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save scaler: {str(e)}")
        raise e
