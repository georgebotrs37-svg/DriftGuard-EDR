import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.data_loader import load_and_split_edr_data

def run_explainability_analysis():
    print("=" * 70)
    print("STEP 9: MODEL EXPLAINABILITY & CYBERSECURITY CAUSALITY ANALYSIS")
    print("=" * 70)
    
    data = load_and_split_edr_data('datasets/edr_events.csv')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    feature_names = np.array(data['feature_names'])
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_train)
    
    # 1. Gini Importance (Tree MDI)
    gini_imp = rf.feature_importances_
    
    # 2. Permutation Importance on unseen Validation Set
    print("Computing Permutation Importance on Validation cohort (10 repeats)...")
    perm_res = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42, scoring='f1')
    perm_mean = perm_res.importances_mean
    perm_std = perm_res.importances_std
    
    # Sort by Permutation Importance
    sorted_idx = np.argsort(perm_mean)[::-1][:10]
    
    top_features = feature_names[sorted_idx]
    top_perm_mean = perm_mean[sorted_idx]
    top_perm_std = perm_std[sorted_idx]
    top_gini = gini_imp[sorted_idx]
    
    print("\nTop 10 Most Influential Behavioral Features (Permutation Importance):")
    for f, p_m, p_s, g in zip(top_features, top_perm_mean, top_perm_std, top_gini):
        print(f"  • {f:<32}: Permutation Impact = {p_m:+.4f} (±{p_s:.4f}) | Gini = {g:.4f}")
        
    # 3. Generate Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Panel 1: Top 10 Permutation Feature Importance Bar Chart
    y_pos = np.arange(len(top_features))[::-1]
    axes[0].barh(y_pos, top_perm_mean, xerr=top_perm_std, color='#38bdf8', ecolor='#f8fafc', capsize=4, alpha=0.9, label='Permutation F1 Drop')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_features, color='#f8fafc', fontsize=9)
    axes[0].set_xlabel('Drop in Validation F1 When Feature is Shuffled (Zero Signal)', color='#94a3b8')
    axes[0].set_title('Top Behavioral Drivers: Permutation Importance', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.2, color='#cbd5e1')
    axes[0].legend(loc='lower right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)

    # Panel 2: The Critical Cybersecurity Warning (Correlation != Causation)
    axes[1].axis('off')
    
    warning_box = plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor='#1e293b', edgecolor='#f59e0b', lw=2, transform=axes[1].transAxes)
    axes[1].add_patch(warning_box)
    
    axes[1].text(0.5, 0.88, 'CRUCIAL CYBERSECURITY WARNING:', color='#f59e0b', fontsize=12, fontweight='bold', ha='center', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.81, 'Statistical Importance != Causal Truth', color='#f8fafc', fontsize=11, fontweight='bold', ha='center', transform=axes[1].transAxes)
    
    caveat_text = (
        "1. Spurious Correlation Hazard:\n"
        "   High 'file_writes' correlated with ransomware in this cohort.\n"
        "   However, file writes do NOT CAUSE malware. A database server\n"
        "   or backup tool writes millions of files completely benignly.\n\n"
        "2. Adversarial Evasion (Tactic Modification):\n"
        "   If an adversary learns our model relies on 'command_line_len',\n"
        "   they execute payloads via DLL injection or short stagers,\n"
        "   dropping feature importance to zero while achieving full compromise.\n\n"
        "3. SOC Operational Reality:\n"
        "   Feature importance is a TRIAGE AID for the human analyst,\n"
        "   NOT a standalone proof of attack or forensic certainty."
    )
    axes[1].text(0.1, 0.44, caveat_text, color='#cbd5e1', fontsize=9.2, linespacing=1.4, transform=axes[1].transAxes)

    plt.suptitle('Step 9 Model Explainability: Feature Attribution & The Causal Boundary', color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path = 'C:/Users/DELL/.gemini/antigravity/brain/302c1ab4-bde3-41f6-919e-7891ea22eab5/explainability_importance.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Saved explainability figure to: {img_path}")

if __name__ == '__main__':
    run_explainability_analysis()

