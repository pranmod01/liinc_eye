# LIINC Analysis Package

Shared utilities and models for multimodal physiological analysis.

## Installation

From a notebook in `notebooks/`, add the src directory to your Python path:

```python
import sys
sys.path.append('../..')  # Add project root to path
```

## Modules

### `src.models.fusion`

Late fusion implementations for combining multiple modalities.

**Example:**
```python
from src.models.fusion import weighted_late_fusion

# Prepare modality features
X_modalities = [X_physio, X_behavior, X_gaze]
modality_names = ['Physiology', 'Behavior', 'Gaze']

# Run weighted late fusion
results = weighted_late_fusion(
    X_modalities, y, subjects, modality_names,
    fusion_method='weighted'  # or 'average', 'stacking'
)

# Access results
print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_sem']:.3f}")
for name, weight in zip(results['modality_names'], results['weights']):
    print(f"{name}: {weight:.3f}")
```

### `src.utils.io`

File loading/saving utilities with error handling.

**Example:**
```python
from src.utils.io import load_features, save_results

# Load extracted features
feature_data = load_features(
    '../../data/results/features_PRE/extracted_features_PRE.pkl',
    timeframe='PRE'
)
merged_df = feature_data['merged_df']
physio_cols = feature_data['physio_cols']

# Save results
save_results(comparison_df, '../../data/results/model_comparison.csv')
```

### `src.utils.config`

Configuration file loading.

**Example:**
```python
from src.utils.config import load_config, get_model_params
from sklearn.ensemble import RandomForestClassifier

# Load model parameters from config
rf_params = get_model_params('random_forest')
model = RandomForestClassifier(**rf_params)

# Or load entire config
config = load_config('model_params')
time_window = config['time_windows']['pre_decision']
```

### `src.utils.validation`

Data validation utilities.

**Example:**
```python
from src.utils.validation import validate_features, validate_modality_features

# Validate loaded features
validate_features(merged_df, timeframe='PRE')

# Validate modality arrays before fusion
validate_modality_features(
    X_modalities, y, subjects, modality_names
)
```

### `src.visualization.plots`

Reusable plotting functions to eliminate visualization code duplication.

**Example:**
```python
from src.visualization.plots import (
    plot_method_comparison,
    plot_modality_weights,
    plot_group_comparison_dual,
    set_style
)

# Set consistent style
set_style('whitegrid')

# Plot method comparison
fig = plot_method_comparison(comparison_df, timeframe='PRE')
plt.show()

# Plot modality weights across fusion methods
fig = plot_modality_weights(weights_df, timeframe='PRE')
plt.show()

# Plot performance across groups (e.g., ambiguity levels)
fig = plot_group_comparison_dual(comparison_df, group_column='Group')
plt.show()
```

**Available plotting functions:**
- `plot_method_comparison()` - Compare accuracy/F1 across methods
- `plot_modality_weights()` - Show modality weights for fusion methods
- `plot_group_comparison()` - Single metric across groups with error bars
- `plot_group_comparison_dual()` - Accuracy + F1 across groups
- `plot_subject_distribution()` - Subject-level accuracy histograms
- `plot_modality_weights_by_group()` - Modality weights by group
- `plot_multi_seed_convergence()` - Performance across random seeds
- `plot_visit_performance()` - Performance across experimental visits
- `set_style()` - Set consistent plotting style

## Configuration Files

### `config/model_params.yaml`

Contains all hyperparameters:
- Random Forest parameters
- Logistic Regression parameters
- XGBoost/LightGBM parameters
- Time windows
- Cross-validation settings
- Bootstrap/permutation test parameters

### `config/paths.yaml`

Contains all data paths:
- Feature file locations
- Output directories
- Session mapping path

## Benefits

✅ **No code duplication** - Shared functions used across all notebooks
✅ **Consistent parameters** - All models use same config
✅ **Error handling** - Helpful error messages if files missing
✅ **Easy maintenance** - Change once, applies everywhere
✅ **Type hints & docs** - Clear API with examples

## Migration Guide

### Before (duplicated code):
```python
# In every notebook:
model = RandomForestClassifier(n_estimators=100, max_depth=5,
                               min_samples_split=10, min_samples_leaf=5,
                               class_weight='balanced', random_state=42)

with open('../../data/results/features_PRE/extracted_features_PRE.pkl', 'rb') as f:
    feature_data = pickle.load(f)
```

### After (using shared utilities):
```python
import sys
sys.path.append('../..')

from src.utils.io import load_features
from src.utils.config import get_model_params
from sklearn.ensemble import RandomForestClassifier

# Load features with error handling
feature_data = load_features(
    '../../data/results/features_PRE/extracted_features_PRE.pkl',
    timeframe='PRE'
)

# Use consistent parameters from config
rf_params = get_model_params('random_forest')
model = RandomForestClassifier(**rf_params)
```

## Testing

Test that imports work:

```python
import sys
sys.path.append('../..')

# Test imports
from src.models.fusion import weighted_late_fusion
from src.utils.io import load_features
from src.utils.config import get_model_params
from src.utils.validation import validate_features

print("✓ All imports successful!")
```

## Future Additions

Planned additions:
- `src.preprocessing` - Feature extraction utilities
- `src.evaluation` - Metrics and statistical tests
