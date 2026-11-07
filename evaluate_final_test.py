import numpy as np
import pandas as pd
import joblib
from src.data_loader import load_and_split_edr_data
from src.metrics import evaluate_predictions

def run_final_test_evaluation():
    print("=" * 70)
    print("STEP 11: ONE-TIME FINAL TEST SET EVALUATION (FROZEN UNTOUCHED COHORT)")
    print("=" * 70)
    
    bundle = joblib.load('models/edr_pipeline.joblib')
    model = bundle['model']
    threshold = bundle['threshold']
    
    data = load_and_split_edr_data('datasets/edr_events.csv')
    X_test_proc = data['X_test']
    y_test = data['y_test']
    hosts_test = data['hosts_test']
    
    # Predict on Final Test Set (EVALUATED ONLY ONCE!)
    test_probs = model.predict_proba(X_test_proc)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    
    m_test = evaluate_predictions(y_test, test_preds, test_probs)
    
    print(f"Final Test Cohort: {len(y_test)} events across {len(hosts_test)} completely unseen hosts")
    print(f"Operational Decision Threshold: {threshold:.2f}\n")
    print(f"  • Detection Rate (Recall):     {m_test['Recall']:.4f} ({m_test['Recall']:.1%})")
    print(f"  • Alert Precision:             {m_test['Precision']:.4f} ({m_test['Precision']:.1%})")
    print(f"  • F1-Score:                    {m_test['F1']:.4f}")
    print(f"  • False Positive Rate (FPR):   {m_test['FPR']:.4f} ({m_test['FPR']:.2%})")
    print(f"  • ROC-AUC:                     {m_test['ROC-AUC']:.4f}")
    print(f"  • PR-AUC (Average Precision):  {m_test['PR-AUC']:.4f}\n")
    print(f"Confusion Matrix (Total: {len(y_test)}):")
    print(f"  • True Positives  (TP): {m_test['TP']}")
    print(f"  • False Positives (FP): {m_test['FP']}")
    print(f"  • False Negatives (FN): {m_test['FN']}")
    print(f"  • True Negatives  (TN): {m_test['TN']}")
    print("=" * 70)
    
    return m_test

if __name__ == '__main__':
    run_final_test_evaluation()

