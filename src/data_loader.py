import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_split_edr_data(filepath='datasets/edr_events.csv', random_state=42):
    """
    Loads EDR data, isolates identifiers, and performs a strict Host-Exclusive split.
    Guarantees that no host appears in more than one partition (Zero Data Leakage).
    
    Partitions:
      - Train: ~70% of hosts
      - Validation: ~15% of hosts
      - Final Test: ~15% of hosts (Preserved for final evaluation)
    """
    df = pd.read_csv(filepath)
    
    # 1. Strictly exclude identifier columns from features
    id_cols = ['event_id', 'process_id', 'parent_process_id']
    
    # Target and group
    target_col = 'is_malicious'
    group_col = 'host_name'
    
    # Available feature columns
    feature_cols = [c for c in df.columns if c not in id_cols + [target_col, group_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]
    
    # 2. Host-exclusive split: Step 1 (Train+Val vs Test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(X, y, groups))
    
    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    hosts_test = groups.iloc[test_idx].unique()
    
    # Host-exclusive split: Step 2 (Train vs Val)
    # 0.1765 of 85% is ~15% of the total
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=random_state)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))
    
    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    hosts_train = groups_train_val.iloc[train_idx].unique()
    
    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]
    hosts_val = groups_train_val.iloc[val_idx].unique()
    
    # 3. Setup Preprocessor (Fit ONLY on Training Data)
    categorical_cols = ['process_name', 'parent_process_name']
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Fit strictly on X_train
    preprocessor.fit(X_train)
    
    # Transform all splits using the preprocessor fitted on train
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = numeric_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    
    splits = {
        'X_train': X_train_proc, 'y_train': y_train.values,
        'X_val': X_val_proc, 'y_val': y_val.values,
        'X_test': X_test_proc, 'y_test': y_test.values,
        'X_train_raw': X_train, 'X_val_raw': X_val, 'X_test_raw': X_test,
        'hosts_train': hosts_train, 'hosts_val': hosts_val, 'hosts_test': hosts_test,
        'preprocessor': preprocessor, 'feature_names': feature_names
    }
    return splits

