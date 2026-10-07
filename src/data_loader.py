"""
data_loader.py

This module handles loading the dataset, handling missing values, and outlier detection.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw CSV data from the specified file path.
    
    Args:
        file_path (str): The absolute or relative path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist at the path.
        Exception: For any other load errors.
    """
    try:
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found at path: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise e

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values by:
    - Filling numeric column NaNs with the column median.
    - Filling categorical/object column NaNs with the column mode (most frequent value).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    df_imputed = df.copy()
    try:
        null_count = df_imputed.isnull().sum().sum()
        if null_count == 0:
            logger.info("No missing values found in the dataset.")
            return df_imputed
            
        logger.info(f"Imputing missing values. Total nulls: {null_count}")
        for col in df_imputed.columns:
            if df_imputed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_imputed[col]):
                    median_val = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(median_val)
                    logger.info(f"Filled missing values in numerical column '{col}' with median: {median_val}")
                else:
                    mode_val = df_imputed[col].mode()[0]
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
                    logger.info(f"Filled missing values in categorical column '{col}' with mode: {mode_val}")
        return df_imputed
    except Exception as e:
        logger.error(f"Error during missing value imputation: {str(e)}")
        raise e

def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], threshold: float = 1.5) -> dict:
    """
    Detects outliers in specific numerical columns using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to check for outliers.
        threshold (float): IQR threshold multiplier (default 1.5).
        
    Returns:
        dict: A dictionary mapping column names to lists of indices that contain outliers.
    """
    outliers_dict = {}
    try:
        for col in columns:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Identify indices containing outliers
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            if outlier_indices:
                outliers_dict[col] = outlier_indices
                logger.info(f"Detected {len(outlier_indices)} outliers in column '{col}' using IQR.")
        return outliers_dict
    except Exception as e:
        logger.error(f"Error during outlier detection: {str(e)}")
        raise e

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'cap', threshold: float = 1.5) -> pd.DataFrame:
    """
    Handles outliers in numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of columns to clean.
        method (str): Outlier handling method:
                      'remove' - drop rows containing outliers.
                      'cap' - cap values to lower and upper bounds.
                      'none' - do nothing.
        threshold (float): IQR multiplier.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if method == 'none' or not columns:
        return df.copy()
        
    df_cleaned = df.copy()
    try:
        if method == 'remove':
            indices_to_drop = set()
            outliers_dict = detect_outliers_iqr(df_cleaned, columns, threshold)
            for indices in outliers_dict.values():
                indices_to_drop.update(indices)
            df_cleaned = df_cleaned.drop(index=list(indices_to_drop)).reset_index(drop=True)
            logger.info(f"Removed {len(indices_to_drop)} rows containing outliers. New shape: {df_cleaned.shape}")
        elif method == 'cap':
            for col in columns:
                if col not in df_cleaned.columns:
                    continue
                q1 = df_cleaned[col].quantile(0.25)
                q3 = df_cleaned[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                # Cap values
                df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
            logger.info(f"Capped outliers to IQR bounds ({lower_bound:.2f}, {upper_bound:.2f}) for columns: {columns}")
        return df_cleaned
    except Exception as e:
        logger.error(f"Error handling outliers: {str(e)}")
        raise e
