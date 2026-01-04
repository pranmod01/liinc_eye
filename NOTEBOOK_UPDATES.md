# Notebook Updates for POST Condition (No Gaze Data)

## Summary
All fusion model notebooks have been updated to handle the POST condition where gaze data is not available (empty gaze_cols).

## Changes Made

### 1. **Fixed `multi_seed_late_fusion.ipynb` Save Cell**
- Updated the final save cell to properly use `save_results()` utility function
- Now correctly saves the `summary_df` to CSV

### 2. **Updated All Fusion Notebooks for Empty Gaze Data**

The following notebooks were updated to conditionally handle gaze features:

- `late_fusion_model.ipynb` ✓
- `multi_seed_late_fusion.ipynb` ✓
- `ambiguity_group_late_fusion.ipynb` ✓
- `ambiguity_group_late_fusion_balanced.ipynb` ✓
- `late_fusion_by_visit.ipynb` ✓
- `reaction_time_group_late_fusion.ipynb` ✓
- `reaction_time_group_late_fusion_balanced.ipynb` ✓

### 3. **Specific Changes in Each Notebook**

#### **Feature Preparation Cell**
```python
# OLD: Always tried to impute gaze features
X_gaze = SimpleImputer(strategy='mean').fit_transform(merged_df[gaze_cols])

# NEW: Handles empty gaze_cols
if len(gaze_cols) > 0:
    X_gaze = SimpleImputer(strategy='mean').fit_transform(merged_df[gaze_cols])
else:
    # Create placeholder array for POST condition (no gaze data)
    X_gaze = np.zeros((len(merged_df), 1))
```

#### **Model Training Cell** (late_fusion_model.ipynb only)
```python
# NEW: Only train gaze model if data exists
if len(gaze_cols) > 0:
    model_gaze, acc_gaze, f1_gaze, pred_gaze, _ = train_evaluate_modality(
        X_gaze, y, subjects, "Gaze"
    )
else:
    # Placeholder for POST (no gaze data)
    model_gaze = None
    acc_gaze = 0.5
    f1_gaze = 0.5
    pred_gaze = np.zeros(len(y))
    print("Skipping - no gaze data available (POST condition)")
```

#### **Fusion Methods Cell**
```python
# OLD: Always included gaze
X_modalities = [X_physio, X_behavior, X_gaze]
modality_names = ['Physiology', 'Behavior', 'Gaze']

# NEW: Conditionally include gaze
if len(gaze_cols) > 0:
    X_modalities = [X_physio, X_behavior, X_gaze]
    modality_names = ['Physiology', 'Behavior', 'Gaze']
else:
    # POST condition: no gaze data
    X_modalities = [X_physio, X_behavior]
    modality_names = ['Physiology', 'Behavior']
```

#### **Comparison DataFrame** (late_fusion_model.ipynb only)
```python
# NEW: Dynamically build comparison data
comparison_data = {
    'Method': ['Physiology Only', 'Behavior Only'],
    'Accuracy': [acc_physio, acc_behavior],
    'F1-Score': [f1_physio, f1_behavior]
}

# Add gaze if available
if len(gaze_cols) > 0:
    comparison_data['Method'].append('Gaze Only')
    comparison_data['Accuracy'].append(acc_gaze)
    comparison_data['F1-Score'].append(f1_gaze)

# Add fusion methods...
```

#### **Summary Print** (late_fusion_model.ipynb only)
```python
# NEW: Conditionally print gaze results
if len(gaze_cols) > 0:
    print(f"   Gaze:        Acc={acc_gaze:.3f}, F1={f1_gaze:.3f}")
else:
    print(f"   Gaze:        Not available (POST condition)")
```

#### **Multi-Seed Aggregate Results** (multi_seed_late_fusion.ipynb)
```python
# NEW: Dynamically add weight columns based on available modalities
for r in results:
    row = {...}
    # Add weights dynamically
    for i, mod_name in enumerate(modality_names):
        row[f'{mod_name}_Weight'] = r['weights'][i]
    summary_data.append(row)
```

#### **Weight Visualization** (multi_seed_late_fusion.ipynb)
```python
# NEW: Build weight data dynamically
weight_data = []
for mod_name in modality_names:
    col_name = f'{mod_name}_Weight'
    weight_data.append(summary_df[col_name].values)

bp = ax.boxplot(weight_data, labels=modality_names, patch_artist=True)
colors = ['steelblue', 'coral', 'mediumseagreen'][:len(modality_names)]
```

## How to Use

1. **For PRE condition** (has gaze data):
   - Set `TIMEFRAME = 'PRE'` in the first cell
   - All three modalities will be used: Physiology, Behavior, Gaze

2. **For POST condition** (no gaze data):
   - Set `TIMEFRAME = 'POST'` in the first cell
   - Only two modalities will be used: Physiology, Behavior
   - Gaze-related outputs will show "Not available (POST condition)"

## Testing

To verify the changes work:

```bash
# Test with POST condition
jupyter nbconvert --to notebook --execute notebooks/fusion_models/late_fusion_model.ipynb
```

All notebooks should now run correctly for both PRE and POST timeframes without errors.
