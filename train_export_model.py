import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_and_split_edr_data

os.makedirs('models', exist_ok=True)

# Train the production tuned model on full training split
data = load_and_split_edr_data('datasets/edr_events.csv')
X_train, y_train = data['X_train'], data['y_train']
preprocessor = data['preprocessor']

rf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42)
rf.fit(X_train, y_train)

pipeline_bundle = {
    'preprocessor': preprocessor,
    'model': rf,
    'feature_names': data['feature_names'],
    'threshold': 0.38
}

joblib.dump(pipeline_bundle, 'models/edr_pipeline.joblib')
print('Exported models/edr_pipeline.joblib successfully!')

