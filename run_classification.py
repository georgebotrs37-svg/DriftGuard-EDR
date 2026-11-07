import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data_loader import load_and_split_edr_data
from src.metrics import evaluate_predictions

def run_benchmarks():
    print("=" * 70)
    print("EDR PROCESS CLASSIFICATION: BASELINE & MODEL BENCHMARK")
    print("=" * 70)
    
    # 1. Load data with strict Host-Exclusive split
    print("[1/4] Loading EDR Telemetry & creating Host-Exclusive Splits...")
    data = load_and_split_edr_data('datasets/edr_events.csv')
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    print(f"  Training Set:   {X_train.shape[0]} events across {len(data['hosts_train'])} unique hosts (Malicious rate: {y_train.mean():.1%})")
    print(f"  Validation Set: {X_val.shape[0]} events across {len(data['hosts_val'])} unique hosts (Malicious rate: {y_val.mean():.1%})")
    print(f"  Final Test Set: {data['X_test'].shape[0]} events across {len(data['hosts_test'])} unique hosts (FROZEN - Evaluated ONCE at end)")
    print(f"  Input Features: {len(data['feature_names'])} engineered behavioral dimensions\n")
    
    # 2. Define models
    models = {
        'Majority Class Baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression (Baseline)': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    roc_curves = {}
    pr_curves = {}
    
    print("[2/4] Training and evaluating models on Validation Set...")
    print("-" * 70)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
        
        metrics = evaluate_predictions(y_val, y_pred, y_prob)
        metrics['Model'] = name
        results.append(metrics)
        
        if y_prob is not None and not np.isnan(metrics['ROC-AUC']):
            fpr_arr, tpr_arr, _ = roc_curve(y_val, y_prob)
            prec_arr, rec_arr, _ = precision_recall_curve(y_val, y_prob)
            roc_curves[name] = (fpr_arr, tpr_arr, metrics['ROC-AUC'])
            pr_curves[name] = (rec_arr, prec_arr, metrics['PR-AUC'])
            
        print(f"-> {name:<30} | F1: {metrics['F1']:.3f} | Recall: {metrics['Recall']:.3f} | FPR: {metrics['FPR']:.3f} | ROC-AUC: {metrics['ROC-AUC']:.3f}")

    # 3. Present formatted comparison table
    print("\n[3/4] Multi-Metric Security Comparison Table (Validation Set):")
    df_res = pd.DataFrame(results)[['Model', 'Precision', 'Recall', 'F1', 'FPR', 'ROC-AUC', 'PR-AUC', 'TP', 'FP', 'FN', 'TN']]
    print(df_res.to_string(index=False, justify='left', float_format=lambda x: f"{x:.4f}"))
    
    # 4. Generate Visual Comparison Graphics
    print("\n[4/4] Generating high-resolution performance comparison figures...")
    os.makedirs('C:/Users/DELL/.gemini/antigravity/brain/302c1ab4-bde3-41f6-919e-7891ea22eab5', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Plot 1: Security Metrics Bar Chart
    comp_models = [r for r in results if r['Model'] != 'Majority Class Baseline']
    m_names = [m['Model'].replace(' (Baseline)', '') for m in comp_models]
    x = np.arange(len(m_names))
    width = 0.2
    
    axes[0].bar(x - 1.5*width, [m['Recall'] for m in comp_models], width, label='Recall (Catch Rate)', color='#10b981')
    axes[0].bar(x - 0.5*width, [m['Precision'] for m in comp_models], width, label='Precision (Alert Quality)', color='#38bdf8')
    axes[0].bar(x + 0.5*width, [m['F1'] for m in comp_models], width, label='F1-Score', color='#a855f7')
    axes[0].bar(x + 1.5*width, [m['FPR'] for m in comp_models], width, label='FPR (False Alarms)', color='#ef4444')
    
    axes[0].set_title('Detection vs False Alarm Tradeoff', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(m_names, rotation=20, ha='right', color='#cbd5e1', fontsize=9)
    axes[0].set_ylabel('Score (0.0 to 1.0)', color='#94a3b8')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
    axes[0].grid(axis='y', linestyle='--', alpha=0.2, color='#cbd5e1')

    # Plot 2: ROC Curves
    palette = {'Logistic Regression (Baseline)': '#38bdf8', 'Decision Tree': '#f59e0b', 
               'Random Forest': '#10b981', 'Gradient Boosting': '#ec4899'}
    
    for name, (fpr_val, tpr_val, auc_val) in roc_curves.items():
        color = palette.get(name, '#cbd5e1')
        axes[1].plot(fpr_val, tpr_val, lw=2.2, color=color, label=f"{name.split(' ')[0]} (AUC={auc_val:.3f})")
    
    axes[1].plot([0, 1], [0, 1], linestyle='--', lw=1.2, color='#64748b', label='Random Guess')
    axes[1].set_title('ROC Curves (Sensitivity vs 1-Specificity)', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate (FPR)', color='#94a3b8')
    axes[1].set_ylabel('True Positive Rate (Recall)', color='#94a3b8')
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc='lower right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
    axes[1].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    # Plot 3: Precision-Recall Curves
    for name, (rec_val, prec_val, pr_auc_val) in pr_curves.items():
        color = palette.get(name, '#cbd5e1')
        axes[2].plot(rec_val, prec_val, lw=2.2, color=color, label=f"{name.split(' ')[0]} (PR-AUC={pr_auc_val:.3f})")
        
    axes[2].set_title('Precision-Recall Curves (Security Essential)', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Recall', color='#94a3b8')
    axes[2].set_ylabel('Precision', color='#94a3b8')
    axes[2].set_xlim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(loc='lower left', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
    axes[2].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    plt.suptitle('Step 4 Benchmark: EDR Model Comparison on Validation Cohort', color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path = 'C:/Users/DELL/.gemini/antigravity/brain/302c1ab4-bde3-41f6-919e-7891ea22eab5/model_benchmark_comparison.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Successfully saved benchmark plot to: {img_path}")
    
    return df_res

if __name__ == '__main__':
    run_benchmarks()
