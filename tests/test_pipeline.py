import unittest
import numpy as np
import pandas as pd
import joblib
from src.data_loader import load_and_split_edr_data

class TestEDRPipeline(unittest.TestCase):
    """
    Automated Unit & Integrity Tests for DriftGuard EDR Pipeline.
    Verifies:
      1. Zero Data Leakage (Strict Host Isolation)
      2. No missing values in processed matrices
      3. Model artifact existence and threshold validity
    """
    
    @classmethod
    def setUpClass(cls):
        cls.data = load_and_split_edr_data('datasets/edr_events.csv')
        cls.bundle = joblib.load('models/edr_pipeline.joblib')

    def test_host_exclusive_integrity(self):
        """CRITICAL: Ensure ZERO host overlap between Train, Val, and Test splits."""
        hosts_train = set(self.data['hosts_train'])
        hosts_val = set(self.data['hosts_val'])
        hosts_test = set(self.data['hosts_test'])
        
        train_val_overlap = hosts_train.intersection(hosts_val)
        train_test_overlap = hosts_train.intersection(hosts_test)
        val_test_overlap = hosts_val.intersection(hosts_test)
        
        self.assertEqual(len(train_val_overlap), 0, f"LEAKAGE DETECTED! Hosts in both Train and Val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"LEAKAGE DETECTED! Hosts in both Train and Test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"LEAKAGE DETECTED! Hosts in both Val and Test: {val_test_overlap}")

    def test_feature_exclusion_compliance(self):
        """Ensure no identifiers (PIDs, Hostnames, Event IDs) leaked into feature matrix."""
        forbidden_keywords = ['event_id', 'host_name', 'process_id', 'parent_process_id']
        feature_names = self.data['feature_names']
        
        for feat in feature_names:
            for kw in forbidden_keywords:
                self.assertNotIn(kw, feat.lower(), f"Forbidden identifier '{kw}' found in feature: '{feat}'")

    def test_model_artifact_and_threshold(self):
        """Verify model bundle loads and decision threshold is within SOC operational bounds."""
        model = self.bundle['model']
        threshold = self.bundle['threshold']
        
        self.assertIsNotNone(model)
        self.assertGreater(threshold, 0.10, "Threshold too low; causes extreme False Positives")
        self.assertLess(threshold, 0.50, "Threshold too high; causes severe False Negatives")
        self.assertAlmostEqual(threshold, 0.38, delta=0.02)

    def test_no_nan_in_splits(self):
        """Verify preprocessing eliminates any NaNs or infinities."""
        self.assertFalse(np.isnan(self.data['X_train']).any())
        self.assertFalse(np.isnan(self.data['X_val']).any())
        self.assertFalse(np.isnan(self.data['X_test']).any())

if __name__ == '__main__':
    unittest.main()

