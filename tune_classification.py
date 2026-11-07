import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data_loader import load_and_split_edr_data

def run_overfitting_analysis():
    print("=" * 70)
    print("STEP 5: DETECTING AND MITIGATING OVERFITTING IN CYBER TELEMETRY")
    print("=" * 70)
    
    # 1. Load host-exclusive data
    data = load_and_split_edr_data('datasets/edr_events.csv')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # 2. Diagnose Overfitting in Default (Unconstrained) Models
    print("\n[1/4] Diagnosing Train vs. Validation Generalization Gap...")
    models = {
        'Unconstrained Decision Tree': DecisionTreeClassifier(random_state=42),
        'Unconstrained Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    gap_records = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_f1 = f1_score(y_train, model.predict(X_train))
        val_f1 = f1_score(y_val, model.predict(X_val))
        gap = train_f1 - val_f1
        gap_records.append({
            'Model': name,
            'Train F1': train_f1,
            'Validation F1': val_f1,
            'Generalization Gap (Overfit)': gap
        })
        print(f"  {name:<30} -> Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Gap: {gap:.4f}")
    
    # 3. Hyperparameter Sweep: max_depth and min_samples_leaf
    print("\n[2/4] Sweeping Regularization Parameters (max_depth & min_samples_leaf)...")
    depths = list(range(1, 21))
    tree_train_f1 = []
    tree_val_f1 = []
    
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tree_train_f1.append(f1_score(y_train, dt.predict(X_train)))
        tree_val_f1.append(f1_score(y_val, dt.predict(X_val)))
        
    leaf_sizes = [1, 2, 4, 8, 12, 16, 24, 32]
    leaf_train_f1 = []
    leaf_val_f1 = []
    for l in leaf_sizes:
        dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=l, random_state=42)
        dt.fit(X_train, y_train)
        leaf_train_f1.append(f1_score(y_train, dt.predict(X_train)))
        leaf_val_f1.append(f1_score(y_val, dt.predict(X_val)))
        
    # 4. Host-Exclusive Cross Validation on Training Split
    print("\n[3/4] Performing 5-Fold Host-Exclusive GroupKFold Cross-Validation...")
    # Map back training host groups
    # We use data['X_train_raw'] index to extract training host IDs
    raw_df = pd.read_csv('datasets/edr_events.csv')
    train_hosts = raw_df.loc[data['X_train_raw'].index, 'host_name'].values
    
    gkf = GroupKFold(n_splits=5)
    cv_scores_unconstrained = []
    cv_scores_regularized = []
    
    for train_fold_idx, val_fold_idx in gkf.split(X_train, y_train, groups=train_hosts):
        X_tr, y_tr = X_train[train_fold_idx], y_train[train_fold_idx]
        X_va, y_va = X_train[val_fold_idx], y_train[val_fold_idx]
        
        # Unconstrained
        m1 = DecisionTreeClassifier(random_state=42)
        m1.fit(X_tr, y_tr)
        cv_scores_unconstrained.append(f1_score(y_va, m1.predict(X_va)))
        
        # Regularized
        m2 = DecisionTreeClassifier(max_depth=8, min_samples_leaf=6, random_state=42)
        m2.fit(X_tr, y_tr)
        cv_scores_regularized.append(f1_score(y_va, m2.predict(X_va)))
        
    print(f"  Unconstrained Tree 5-Fold CV F1: {np.mean(cv_scores_unconstrained):.4f} (+/- {np.std(cv_scores_unconstrained):.4f})")
    print(f"  Regularized Tree   5-Fold CV F1: {np.mean(cv_scores_regularized):.4f} (+/- {np.std(cv_scores_regularized):.4f})")
    
    # Also evaluate tuned Random Forest
    rf_tuned = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42)
    rf_tuned.fit(X_train, y_train)
    rf_val_f1 = f1_score(y_val, rf_tuned.predict(X_val))
    print(f"\n  Tuned Random Forest (max_depth=12, min_samples_leaf=3) -> Val F1: {rf_val_f1:.4f}")
    
    # 5. Visualizations
    print("\n[4/4] Generating Overfitting Analysis Visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Subplot 1: max_depth curve
    axes[0].plot(depths, tree_train_f1, marker='o', lw=2.2, color='#ef4444', label='Train F1 (Memorization)')
    axes[0].plot(depths, tree_val_f1, marker='s', lw=2.2, color='#10b981', label='Validation F1 (Generalization)')
    axes[0].axvline(x=depths[np.argmax(tree_val_f1)], color='#f59e0b', linestyle='--', label=f'Optimal Depth ({depths[np.argmax(tree_val_f1)]})')
    axes[0].set_title('Overfitting Diagnosis: max_depth Sweep', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tree Depth (max_depth)', color='#94a3b8')
    axes[0].set_ylabel('F1 Score', color='#94a3b8')
    axes[0].set_ylim(0.7, 1.02)
    axes[0].legend(loc='lower right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[0].grid(linestyle='--', alpha=0.2, color='#cbd5e1')
    
    # Subplot 2: min_samples_leaf curve
    axes[1].plot(leaf_sizes, leaf_train_f1, marker='o', lw=2.2, color='#ef4444', label='Train F1')
    axes[1].plot(leaf_sizes, leaf_val_f1, marker='s', lw=2.2, color='#38bdf8', label='Validation F1')
    axes[1].set_title('Regularization: min_samples_leaf', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Min Samples Per Leaf', color='#94a3b8')
    axes[1].set_ylabel('F1 Score', color='#94a3b8')
    axes[1].set_ylim(0.7, 1.02)
    axes[1].legend(loc='lower left', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[1].grid(linestyle='--', alpha=0.2, color='#cbd5e1')

    # Subplot 3: Cross-Validation Fold Stability
    folds = np.arange(1, 6)
    w = 0.35
    axes[2].bar(folds - w/2, cv_scores_unconstrained, width=w, color='#ef4444', alpha=0.8, label='Unconstrained (High Variance)')
    axes[2].bar(folds + w/2, cv_scores_regularized, width=w, color='#10b981', alpha=0.8, label='Regularized (Stable)')
    axes[2].set_title('5-Fold Host-Exclusive CV Stability', color='#f8fafc', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Validation Fold Index', color='#94a3b8')
    axes[2].set_ylabel('Fold F1 Score', color='#94a3b8')
    axes[2].set_ylim(0.7, 1.0)
    axes[2].legend(loc='lower right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)
    axes[2].grid(axis='y', linestyle='--', alpha=0.2, color='#cbd5e1')

    plt.suptitle('Step 5 Overfitting Analysis: Detecting and Fixing Memorization', color='#38bdf8', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    img_path = 'C:/Users/DELL/.gemini/antigravity/brain/302c1ab4-bde3-41f6-919e-7891ea22eab5/overfitting_analysis.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"Saved overfitting analysis figure to: {img_path}")

if __name__ == '__main__':
    run_overfitting_analysis()

