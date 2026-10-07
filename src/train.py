"""
train.py

Main entry point script to train and save the machine learning model.
"""

import sys
import logging
from src.model_training import run_training_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing the model training process...")
    try:
        # Run training pipeline with outliers handled by capping, no scaling (same as notebook), and tuning enabled
        best_model, comparison_table, test_metrics = run_training_pipeline(
            data_path='data/loan_data.csv',
            outlier_method='cap',
            scale=False,
            tune=True
        )
        logger.info("Training pipeline finished successfully!")
        logger.info(f"Test Set Performance: {test_metrics}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
