#!/usr/bin/env python3
"""Quick test to verify the fixed fusion implementation works correctly."""

import sys
import numpy as np
from sklearn.impute import SimpleImputer

sys.path.append('.')
from src.utils.io import load_features
from src.models.fusion import weighted_late_fusion

print("Loading PRE features...")
feature_data = load_features('data/results/features_PRE/extracted_features_PRE.pkl', timeframe='PRE')
merged_df = feature_data['merged_df']

print("Preparing data...")
X_physio = SimpleImputer(strategy='mean').fit_transform(merged_df[feature_data['physio_cols']])
X_behavior = SimpleImputer(strategy='mean').fit_transform(merged_df[feature_data['behavior_cols']])
X_gaze = SimpleImputer(strategy='mean').fit_transform(merged_df[feature_data['gaze_cols']])
y = merged_df['outcome'].values
subjects = merged_df['subject_id'].values

X_modalities = [X_physio, X_behavior, X_gaze]
modality_names = ['Physiology', 'Behavior', 'Gaze']

print(f"\nData shapes:")
print(f"  Physiology: {X_physio.shape}")
print(f"  Behavior: {X_behavior.shape}")
print(f"  Gaze: {X_gaze.shape}")
print(f"  Samples: {len(y)}, Subjects: {len(np.unique(subjects))}")

print("\n" + "="*70)
print("Testing FIXED weighted fusion (with nested CV)...")
print("="*70)
results = weighted_late_fusion(
    X_modalities, y, subjects, modality_names,
    fusion_method='weighted',
    random_state=42
)

print(f"\nResults:")
print(f"  Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_sem']:.3f}")
print(f"  F1-Score: {results['f1_mean']:.3f} ± {results['f1_sem']:.3f}")
print(f"\nModality weights:")
for name, weight in zip(modality_names, results['weights']):
    print(f"  {name:12s}: {weight*100:5.1f}%")

print("\n" + "="*70)
print("Expected: Behavior should have higher weight than before")
print("Previous (buggy): Gaze=99.7%, Behavior=0.3%, Physio=0.0%")
print("="*70)
