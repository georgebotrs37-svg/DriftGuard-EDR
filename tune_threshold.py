
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from src.data_loader import load_and_split_edr_data

def evaluate_threshold_cost():
    data = load_and_split_edr_data('datasets/edr_events.csv')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_train)
    y_prob_val = rf.predict_proba(X_val)[:, 1]

    # Security Operations Center (SOC) cost matrix:
    # Cost of 1 False Positive (FP) = $20 (human triage time: ~15 mins @ $80/hr)
    # Cost of 1 False Negative (FN) = $500 (compromise risk, breach containment, lateral movement)
    cost_fp = 20
    cost_fn = 500

    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    recalls = []
    precisions = []
    fprs = []
    f1s = []
    fps = []
    fns = []

    for t in thresholds:
        y_pred = (y_prob_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        recalls.append(rec)
        precisions.append(prec)
        fprs.append(fpr)
        f1s.append(f1)
        fps.append(fp)
        fns.append(fn)

    # 1. Default Threshold (0.50)
    def_idx = int(np.argmin(np.abs(thresholds - 0.5)))
    def_t = thresholds[def_idx]

    # 2. Pure Cost-Minimization Threshold
    opt_idx = int(np.argmin(costs))
    opt_t = thresholds[opt_idx]

    # 3. SOC Alert-Budget Threshold (FPR <= 0.005 or 0.5%)
    budget_candidates = [i for i, f in enumerate(fprs) if f <= 0.005]
    budget_idx = budget_candidates[np.argmax([recalls[i] for i in budget_candidates])]
    budget_t = thresholds[budget_idx]

    print("=" * 65)
    print("THRESHOLD OPTIMIZATION RESULTS ON VALIDATION COHORT")
    print("=" * 65)
    print(f"1. Default Threshold (t = {def_t:.2f}):")
    print(f"   Recall (Catch Rate): {recalls[def_idx]:.1%} | FPR: {fprs[def_idx]:.2%} ({fps[def_idx]} FP, {fns[def_idx]} FN)")
    print(f"   Precision: {precisions[def_idx]:.1%} | F1: {f1s[def_idx]:.3f} | Total SOC Cost: ${costs[def_idx]:,}\n")

    print(f"2. Pure Cost-Optimal Threshold (t = {opt_t:.2f}):")
    print(f"   Recall (Catch Rate): {recalls[opt_idx]:.1%} | FPR: {fprs[opt_idx]:.2%} ({fps[opt_idx]} FP, {fns[opt_idx]} FN)")
    print(f"   Precision: {precisions[opt_idx]:.1%} | F1: {f1s[opt_idx]:.3f} | Total SOC Cost: ${costs[opt_idx]:,}\n")

    print(f"3. SOC Alert-Budget Justified Threshold (t = {budget_t:.2f}):")
    print(f"   Recall (Catch Rate): {recalls[budget_idx]:.1%} | FPR: {fprs[budget_idx]:.2%} ({fps[budget_idx]} FP, {fns[budget_idx]} FN)")
    print(f"   Precision: {precisions[budget_idx]:.1%} | F1: {f1s[budget_idx]:.3f} | Total SOC Cost: ${costs[budget_idx]:,}")
    print("=" * 65)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Panel 1: Precision / Recall / F1 trade-off
    axes[0].plot(thresholds, recalls, lw=2.2, color='#10b981', label='Recall (Detection Rate)')
    axes[0].plot(thresholds, precisions, lw=2.2, color='#38bdf8', label='Precision (Alert Authenticity)')
    axes[0].plot(thresholds, f1s, lw=2.0, linestyle=':', color='#a855f7', label='F1-Score')
    axes[0].axvline(budget_t, color='#f59e0b', linestyle='--', lw=1.8, label=f'Chosen SOC Threshold ({budget_t:.2f})')
    axes[0].set_title('Security Metric Trade-off vs. Threshold', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Decision Threshold (Probability Cutoff)', color='#94a3b8')
    axes[0].set_ylabel('Metric Score', color='#94a3b8')
    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower left', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[0].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    # Panel 2: Total Operational SOC Cost
    axes[1].plot(thresholds, costs, lw=2.5, color='#ef4444', label='Total Operational Cost ($)')
    axes[1].scatter([def_t], [costs[def_idx]], color='#94a3b8', s=80, zorder=5, label=f'Default 0.50 (${costs[def_idx]:,})')
    axes[1].scatter([budget_t], [costs[budget_idx]], color='#f59e0b', s=100, zorder=5, label=f'SOC Budget {budget_t:.2f} (${costs[budget_idx]:,})')
    axes[1].scatter([opt_t], [costs[opt_idx]], color='#10b981', s=100, zorder=5, label=f'Cost-Min {opt_t:.2f} (${costs[opt_idx]:,})')
    axes[1].set_title('SOC Cost Curve: FN ($500) vs FP ($20)', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Decision Threshold', color='#94a3b8')
    axes[1].set_ylabel('Total Operational Cost ($)', color='#94a3b8')
    axes[1].legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[1].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    # Panel 3: False Positive Rate vs. Catch Rate (SOC Operating Window)
    axes[2].plot(fprs, recalls, lw=2.2, color='#38bdf8', label='Operating Curve')
    axes[2].axvspan(0, 0.005, color='#10b981', alpha=0.15, label='SOC Alert Budget (FPR <= 0.5%)')
    axes[2].scatter([fprs[budget_idx]], [recalls[budget_idx]], color='#f59e0b', s=110, zorder=5, label=f'Selected Point (FPR={fprs[budget_idx]:.2%}, Rec={recalls[budget_idx]:.1%})')
    axes[2].set_title('SOC Alert Budget Operating Window', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('False Positive Rate (FPR)', color='#94a3b8')
    axes[2].set_ylabel('Recall (Threat Catch Rate)', color='#94a3b8')
    axes[2].set_xlim(-0.005, 0.08)
    axes[2].set_ylim(0.7, 1.02)
    axes[2].legend(loc='lower right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[2].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    plt.suptitle('Step 6 Threshold Optimization: Balancing Threat Catch vs. SOC Alert Fatigue', color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path = 'C:/Users/DELL/.gemini/antigravity/brain/302c1ab4-bde3-41f6-919e-7891ea22eab5/threshold_cost_analysis.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Saved threshold cost analysis figure to: {img_path}")

if __name__ == '__main__':
    evaluate_threshold_cost()
