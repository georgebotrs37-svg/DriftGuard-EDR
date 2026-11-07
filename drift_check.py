import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def check_feature_drift(baseline_path='datasets/edr_events.csv', challenge_path='datasets/edr_events_challenge.csv'):
    print("=" * 70)
    print("FEATURE & COVARIATE DRIFT ANALYSIS: BASELINE vs CHALLENGE COHORT")
    print("=" * 70)
    
    df_base = pd.read_csv(baseline_path)
    df_chal = pd.read_csv(challenge_path)
    
    numeric_features = [
        'command_line_len', 'process_path_depth', 'cpu_pct', 'memory_mb',
        'network_conns', 'remote_ips', 'file_writes', 'file_deletes',
        'registry_writes', 'handles_count', 'thread_count'
    ]
    
    drift_records = []
    
    print("\n[1/2] Computing Kolmogorov-Smirnov (KS) Test for Numeric Telemetry Features:")
    print(f"{'Feature':<22} | {'KS Stat':<9} | {'p-value':<12} | {'Drift Status'}")
    print("-" * 65)
    
    for feat in numeric_features:
        stat, p_val = ks_2samp(df_base[feat], df_chal[feat])
        is_drifted = (p_val < 0.01) and (stat > 0.05)
        status = "⚠️ SIGNIFICANT DRIFT" if is_drifted else "✅ Stable"
        drift_records.append({
            'Feature': feat,
            'KS_Stat': stat,
            'p_value': p_val,
            'Drifted': is_drifted,
            'Base_Mean': df_base[feat].mean(),
            'Chal_Mean': df_chal[feat].mean()
        })
        print(f"{feat:<22} | {stat:<9.4f} | {p_val:<12.2e} | {status}")
        
    print("\n[2/2] Checking Categorical Process Name Distribution Shift:")
    top_base_procs = df_base['process_name'].value_counts(normalize=True).head(5)
    top_chal_procs = df_chal['process_name'].value_counts(normalize=True).head(5)
    
    print("  Baseline Cohort Top Processes:")
    for proc, pct in top_base_procs.items():
        print(f"    - {proc:<16}: {pct:.1%}")
        
    print("\n  Challenge Cohort Top Processes (New Enterprise Software):")
    for proc, pct in top_chal_procs.items():
        print(f"    - {proc:<16}: {pct:.1%}")
        
    df_drift = pd.DataFrame(drift_records)
    return df_drift

if __name__ == '__main__':
    check_feature_drift()

