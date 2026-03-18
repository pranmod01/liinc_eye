#!/usr/bin/env python3
"""
Analyze per-subject accuracy to identify subjects dragging down overall model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Load subject accuracies
data_dir = Path('/Users/pranmodu/Projects/columbia/liinc/data/results/fusion_model_results_PRE')
subject_acc = pd.read_csv(data_dir / 'late_fusion_model_PRE_subject_accuracies.csv')

print("="*80)
print("SUBJECT-LEVEL ACCURACY ANALYSIS")
print("="*80)

# Basic stats
print(f"\nTotal subjects: {len(subject_acc)}")
print(f"Mean accuracy: {subject_acc['accuracy'].mean():.4f}")
print(f"Median accuracy: {subject_acc['accuracy'].median():.4f}")
print(f"SD accuracy: {subject_acc['accuracy'].std():.4f}")
print(f"Min accuracy: {subject_acc['accuracy'].min():.4f}")
print(f"Max accuracy: {subject_acc['accuracy'].max():.4f}")

# Identify low performers (below chance or significantly below mean)
below_chance = subject_acc[subject_acc['accuracy'] < 0.5]
below_mean_1sd = subject_acc[subject_acc['accuracy'] < (subject_acc['accuracy'].mean() - subject_acc['accuracy'].std())]

print(f"\n" + "-"*80)
print("LOW-PERFORMING SUBJECTS")
print("-"*80)

print(f"\nSubjects below chance (< 50%): {len(below_chance)} ({100*len(below_chance)/len(subject_acc):.1f}%)")
if len(below_chance) > 0:
    print("\nBelow-chance subjects:")
    print(below_chance.sort_values('accuracy')[['subject_id', 'accuracy']].to_string(index=False))

print(f"\nSubjects more than 1 SD below mean (< {subject_acc['accuracy'].mean() - subject_acc['accuracy'].std():.4f}): {len(below_mean_1sd)}")

# Identify high performers
above_80 = subject_acc[subject_acc['accuracy'] >= 0.80]
above_90 = subject_acc[subject_acc['accuracy'] >= 0.90]

print(f"\n" + "-"*80)
print("HIGH-PERFORMING SUBJECTS")
print("-"*80)

print(f"\nSubjects above 80% accuracy: {len(above_80)} ({100*len(above_80)/len(subject_acc):.1f}%)")
print(f"Subjects above 90% accuracy: {len(above_90)} ({100*len(above_90)/len(subject_acc):.1f}%)")

if len(above_90) > 0:
    print("\nTop 10 performers:")
    print(subject_acc.nlargest(10, 'accuracy')[['subject_id', 'accuracy']].to_string(index=False))

# Distribution analysis
print(f"\n" + "-"*80)
print("DISTRIBUTION ANALYSIS")
print("-"*80)

percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    print(f"  {p}th percentile: {subject_acc['accuracy'].quantile(p/100):.4f}")

# Calculate what happens if we remove low performers
print(f"\n" + "-"*80)
print("IMPACT OF REMOVING LOW PERFORMERS")
print("-"*80)

for threshold in [0.40, 0.45, 0.50]:
    filtered = subject_acc[subject_acc['accuracy'] >= threshold]
    n_removed = len(subject_acc) - len(filtered)
    new_mean = filtered['accuracy'].mean()
    print(f"\nRemoving subjects with acc < {threshold:.0%}:")
    print(f"  Subjects removed: {n_removed}")
    print(f"  New mean accuracy: {new_mean:.4f} (was {subject_acc['accuracy'].mean():.4f})")
    print(f"  Improvement: {new_mean - subject_acc['accuracy'].mean():.4f}")

# Extract subject codes to identify patterns
print(f"\n" + "-"*80)
print("SUBJECT CODE ANALYSIS")
print("-"*80)

# Parse subject IDs to extract device/session info
subject_acc['device'] = subject_acc['subject_id'].apply(lambda x: x.split('_')[-1])

device_stats = subject_acc.groupby('device')['accuracy'].agg(['mean', 'std', 'count']).round(4)
print("\nAccuracy by device code:")
print(device_stats.to_string())

# Check for subjects appearing multiple times (same device, different dates)
subject_acc['date_time'] = subject_acc['subject_id'].apply(lambda x: '_'.join(x.split('_')[:2]))

# Find subjects with lowest accuracy
print(f"\n" + "-"*80)
print("BOTTOM 20 SUBJECTS (DRAGGING DOWN ACCURACY)")
print("-"*80)
bottom_20 = subject_acc.nsmallest(20, 'accuracy')
print(bottom_20[['subject_id', 'accuracy']].to_string(index=False))

# Summary
print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Key findings:
1. {len(below_chance)} subjects perform BELOW CHANCE (< 50%), suggesting:
   - Possible data quality issues
   - Highly atypical decision patterns
   - Or model predictions are anti-correlated with their choices

2. If we removed the {len(below_chance)} below-chance subjects:
   - Mean accuracy would increase from {subject_acc['accuracy'].mean():.4f} to {subject_acc[subject_acc['accuracy'] >= 0.5]['accuracy'].mean():.4f}

3. The {len(above_90)} subjects above 90% demonstrate the ceiling potential.

4. Device codes show {'different' if device_stats['mean'].std() > 0.05 else 'similar'} patterns across groups.
""")

# Save analysis
output = {
    'below_chance_subjects': below_chance['subject_id'].tolist(),
    'above_90_subjects': above_90['subject_id'].tolist(),
    'bottom_20_subjects': bottom_20['subject_id'].tolist(),
    'mean_accuracy': subject_acc['accuracy'].mean(),
    'mean_without_below_chance': subject_acc[subject_acc['accuracy'] >= 0.5]['accuracy'].mean(),
}

print("\nAnalysis complete.")
