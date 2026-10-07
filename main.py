"""
main.py

Main CLI utility to guide the user on launching Streamlit or retraining the model.
"""

import sys
import os

def main():
    print("=" * 60)
    print("      FinSight: Loan Repayment Prediction & Underwriting System      ")
    print("=" * 60)
    print("To train the predictive model, run:")
    print("    python -m src.train")
    print("\nTo start the Streamlit Fintech Dashboard app, run:")
    print("    streamlit run app.py")
    print("=" * 60)
    
    # Check if a model is trained
    model_path = 'models/best_model.pkl'
    if not os.path.exists(model_path):
        print(f"WARNING: Persisted model file not found at '{model_path}'.")
        print("Please train the model first by running: python -m src.train")
        print("=" * 60)
        
if __name__ == '__main__':
    main()
