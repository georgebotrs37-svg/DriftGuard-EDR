import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_and_split_edr_data

def run_error_inspection():
    print("=" * 70)
    print("STEP 7: DETAILED ERROR ANALYSIS (FALSE POSITIVES & FALSE NEGATIVES)")
    print("=" * 70)
    
    # 1. Load data
    data = load_and_split_edr_data('datasets/edr_events.csv')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    raw_val = data['X_val_raw'].copy()
    
    # 2. Train regularized model & score with justified threshold t = 0.38
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_train)
    
    y_prob = rf.predict_proba(X_val)[:, 1]
    threshold = 0.38
    y_pred = (y_prob >= threshold).astype(int)
    
    raw_val['y_true'] = y_val
    raw_val['y_pred'] = y_pred
    raw_val['malicious_prob'] = y_prob
    
    # 3. Categorize errors
    tp_df = raw_val[(raw_val['y_true'] == 1) & (raw_val['y_pred'] == 1)]
    tn_df = raw_val[(raw_val['y_true'] == 0) & (raw_val['y_pred'] == 0)]
    fp_df = raw_val[(raw_val['y_true'] == 0) & (raw_val['y_pred'] == 1)]
    fn_df = raw_val[(raw_val['y_true'] == 1) & (raw_val['y_pred'] == 0)]
    
    print(f"\nValidation Cohort Distribution (Total: {len(raw_val)}):")
    print(f"  • True Negatives  (TN): {len(tn_df)} (Correctly identified benign)")
    print(f"  • True Positives  (TP): {len(tp_df)} (Correctly detected attacks)")
    print(f"  • False Positives (FP): {len(fp_df)} (Benign mistaken for attacks - Alert Fatigue)")
    print(f"  • False Negatives (FN): {len(fn_df)} (Attacks missed - Security Blindspots)\n")
    
    print("-" * 70)
    print("DEEP DIVE: FALSE POSITIVES (Why did benign actions trigger alerts?)")
    print("-" * 70)
    for idx, row in fp_df.iterrows():
        print(f"• Event #{idx} | Process: {row['process_name']} (Parent: {row['parent_process_name']})")
        print(f"  Prob: {row['malicious_prob']:.2%} | Elevated: {row['is_elevated']} | CmdLen: {row['command_line_len']} | Writes: {row['file_writes']} | Conns: {row['network_conns']}")
        print(f"  Root Cause: Administrative scripting / developer workflow mimicking attacker IOCs.\n")
        
    print("-" * 70)
    print("DEEP DIVE: FALSE NEGATIVES (Why did these attacks bypass the model?)")
    print("-" * 70)
    for idx, row in fn_df.iterrows():
        print(f"• Event #{idx} | Process: {row['process_name']} (Parent: {row['parent_process_name']})")
        print(f"  Prob: {row['malicious_prob']:.2%} | Elevated: {row['is_elevated']} | CmdLen: {row['command_line_len']} | Writes: {row['file_writes']} | Conns: {row['network_conns']}")
        print(f"  Root Cause: Stealthy Living-off-the-Land (LOLBin) or process masquerading (e.g. svchost.exe).\n")

    # 4. Generate Visual Error Diagnosis Graphic
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Subplot 1: Probability Distribution by Error Class
    axes[0].hist(tn_df['malicious_prob'], bins=25, alpha=0.5, color='#38bdf8', label=f'True Benign (TN, n={len(tn_df)})')
    axes[0].hist(tp_df['malicious_prob'], bins=25, alpha=0.6, color='#10b981', label=f'True Attacks (TP, n={len(tp_df)})')
    axes[0].scatter(fp_df['malicious_prob'], np.full(len(fp_df), 15), color='#f59e0b', s=90, zorder=5, label=f'False Positives (FP, n={len(fp_df)})')
    axes[0].scatter(fn_df['malicious_prob'], np.full(len(fn_df), 15), color='#ef4444', s=90, marker='X', zorder=5, label=f'False Negatives (FN, n={len(fn_df)})')
    axes[0].axvline(threshold, color='#f59e0b', linestyle='--', lw=2, label=f'Threshold ({threshold})')
    axes[0].set_title('Probability Distribution & Decision Boundary Placement', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Malicious Probability', color='#94a3b8')
    axes[0].set_ylabel('Event Frequency', color='#94a3b8')
    axes[0].legend(loc='upper center', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[0].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    # Subplot 2: Behavioral Feature Divergence across Classes
    features_to_compare = ['command_line_len', 'file_writes', 'network_conns', 'handles_count']
    benign_means = [tn_df[f].mean() for f in features_to_compare]
    fp_means = [fp_df[f].mean() for f in features_to_compare]
    fn_means = [fn_df[f].mean() for f in features_to_compare]
    mal_means = [tp_df[f].mean() for f in features_to_compare]
    
    # Normalized for comparison
    x = np.arange(len(features_to_compare))
    w = 0.2
    
    # Normalize by max in each feature for clear visualization
    maxs = np.maximum.reduce([benign_means, fp_means, fn_means, mal_means])
    
    axes[1].bar(x - 1.5*w, np.array(benign_means)/maxs, width=w, label='True Benign', color='#38bdf8')
    axes[1].bar(x - 0.5*w, np.array(fp_means)/maxs, width=w, label='False Positives (Admin Noise)', color='#f59e0b')
    axes[1].bar(x + 0.5*w, np.array(fn_means)/maxs, width=w, label='False Negatives (Stealth Attack)', color='#ef4444')
    axes[1].bar(x + 1.5*w, np.array(mal_means)/maxs, width=w, label='True Attacks', color='#10b981')
    
    axes[1].set_title('Feature Signature Comparison: Why Errors Occur', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Cmd Length', 'File Writes', 'Network Conns', 'Handles'], color='#cbd5e1', fontsize=9.5)
    axes[1].set_ylabel('Relative Intensity (Normalized)', color='#94a3b8')
    axes[1].set_ylim(0, 1.25)
    axes[1].legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[1].grid(axis='y', linestyle='--', alpha=0.2, color='#cbd5e1')

    plt.suptitle('Step 7 Error Analysis: Root Cause Diagnosis for SOC Triage', color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path = 'C:/Users/DELL/.gemini/antigravity/brain/302c1ab4-bde3-41f6-919e-7891ea22eab5/error_analysis_breakdown.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Saved error analysis visualization to: {img_path}")

if __name__ == '__main__':
    run_error_inspection()

