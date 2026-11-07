import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """
    Evaluates predictions across 6 cybersecurity-essential metrics:
    - Precision: TP / (TP + FP)
    - Recall (Detection Rate): TP / (TP + FN)
    - F1-Score: Harmonic mean of Precision and Recall
    - FPR (False Positive Rate): FP / (FP + TN) -> Key for SOC Alert Budget!
    - ROC-AUC: Receiver Operating Characteristic Area Under Curve
    - PR-AUC: Precision-Recall Area Under Curve (Average Precision)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    roc_auc = np.nan
    pr_auc = np.nan
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            pass
            
    return {
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'FPR': fpr,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn)
    }

