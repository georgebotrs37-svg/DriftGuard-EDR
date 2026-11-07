import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_and_split_edr_data
from src.metrics import evaluate_predictions
from drift_check import check_feature_drift

def run_evaluation_shift():
    print("=" * 70)
    print("STEP 8: FROZEN MODEL EVALUATION ON DRIFTED CHALLENGE COHORT")
    print("=" * 70)
    
    # 1. Load baseline data and train frozen model
    data = load_and_split_edr_data('datasets/edr_events.csv')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    preprocessor = data['preprocessor']
    
    frozen_rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42)
    frozen_rf.fit(X_train, y_train)
    
    threshold = 0.38
    
    # Baseline validation evaluation
    val_probs = frozen_rf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= threshold).astype(int)
    m_base = evaluate_predictions(y_val, val_preds, val_probs)
    
    # 2. Ingest Challenge Cohort (2000 events)
    df_chal = pd.read_csv('datasets/edr_events_challenge.csv')
    id_cols = ['event_id', 'process_id', 'parent_process_id', 'is_malicious', 'host_name']
    X_chal_raw = df_chal[[c for c in df_chal.columns if c not in id_cols]]
    y_chal = df_chal['is_malicious'].values
    
    # Transform using FROZEN preprocessor (No retraining, no fitting on challenge data)
    X_chal_proc = preprocessor.transform(X_chal_raw)
    chal_probs = frozen_rf.predict_proba(X_chal_proc)[:, 1]
    chal_preds = (chal_probs >= threshold).astype(int)
    m_chal = evaluate_predictions(y_chal, chal_preds, chal_probs)
    
    # 3. Print Performance Degradation Table
    print("\n[PERFORMANCE DEGRADATION AUDIT]")
    metrics_list = ['Recall', 'Precision', 'F1', 'FPR', 'ROC-AUC', 'PR-AUC']
    rows = []
    for met in metrics_list:
        b_val = m_base[met]
        c_val = m_chal[met]
        delta = c_val - b_val
        pct_change = (delta / b_val) * 100 if b_val != 0 else np.nan
        rows.append({
            'Metric': met,
            'Baseline Val': f"{b_val:.4f}",
            'Challenge Cohort': f"{c_val:.4f}",
            'Absolute Delta': f"{delta:+.4f}",
            'Relative Change': f"{pct_change:+.1f}%"
        })
        
    df_comp = pd.DataFrame(rows)
    print(df_comp.to_string(index=False, justify='left'))
    
    print("\nConfusion Matrix Breakdown:")
    print(f"  • Baseline (1,113 events):  TP={m_base['TP']} | FP={m_base['FP']} | FN={m_base['FN']} | TN={m_base['TN']}")
    print(f"  • Challenge (2,000 events): TP={m_chal['TP']} | FP={m_chal['FP']} | FN={m_chal['FN']} | TN={m_chal['TN']}")
    
    # 4. Generate Visualization
    print("\nGenerating Drift & Performance Degradation Visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Panel 1: Metric Drop Bar Chart
    x = np.arange(len(metrics_list))
    w = 0.35
    b_scores = [m_base[m] for m in metrics_list]
    c_scores = [m_chal[m] for m in metrics_list]
    
    axes[0].bar(x - w/2, b_scores, width=w, label='Baseline Validation', color='#38bdf8')
    axes[0].bar(x + w/2, c_scores, width=w, label='Challenge Cohort (Drifted)', color='#ef4444')
    axes[0].set_title('Performance Degradation Under Concept Drift', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_list, color='#cbd5e1', fontsize=9.5)
    axes[0].set_ylabel('Score', color='#94a3b8')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[0].grid(axis='y', linestyle='--', alpha=0.2, color='#cbd5e1')

    # Panel 2: Predicted Probability Density Shift
    axes[1].hist(val_probs, bins=25, alpha=0.5, color='#38bdf8', density=True, label='Baseline Cohort Probs')
    axes[1].hist(chal_probs, bins=25, alpha=0.5, color='#ef4444', density=True, label='Challenge Cohort Probs')
    axes[1].axvline(threshold, color='#f59e0b', linestyle='--', lw=2, label=f'Decision Threshold ({threshold})')
    axes[1].set_title('Model Confidence Distribution Shift', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Malicious Probability', color='#94a3b8')
    axes[1].set_ylabel('Probability Density', color='#94a3b8')
    axes[1].legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[1].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    # Panel 3: KS-Drift Severity per Feature
    df_drift = check_feature_drift('datasets/edr_events.csv', 'datasets/edr_events_challenge.csv')
    df_drift = df_drift.sort_values(by='KS_Stat', ascending=True)
    y_pos = np.arange(len(df_drift))
    
    axes[2].barh(y_pos, df_drift['KS_Stat'], color='#f59e0b', alpha=0.85)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(df_drift['Feature'], color='#cbd5e1', fontsize=8.5)
    axes[2].set_title('Kolmogorov-Smirnov (KS) Feature Drift Severity', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('KS Statistic (Higher = More Severe Drift)', color='#94a3b8')
    axes[2].set_xlim(0, 1.0)
    axes[2].grid(axis='x', linestyle='--', alpha=0.2, color='#cbd5e1')

    plt.suptitle('Step 8 Generalization Stress-Testing: Evaluating Frozen Model on Challenge Cohort', color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path = 'C:/Users/DELL/.gemini/antigravity/brain/302c1ab4-bde3-41f6-919e-7891ea22eab5/drift_evaluation_comparison.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Saved drift evaluation comparison figure to: {img_path}")

if __name__ == '__main__':
    run_evaluation_shift()

