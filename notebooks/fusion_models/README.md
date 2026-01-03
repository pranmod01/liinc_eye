# Consolidated Fusion Model Notebooks

This directory contains **parameterized** notebooks that work for both PRE and POST analyses.

## Why Consolidate?

**Before**: 16 separate notebooks (8 PRE + 8 POST) with 95% duplicate code
**After**: 8 consolidated notebooks with a `TIMEFRAME` parameter

## Benefits

✅ **Less duplication**: Change logic once, applies to both PRE and POST
✅ **Easier maintenance**: Fix bugs in one place
✅ **Consistent results**: Same code ensures fair comparison
✅ **Clearer organization**: Related analyses grouped together

---

## How to Use

### 1. Set the `TIMEFRAME` parameter

At the top of each notebook, you'll see:

```python
# ============================================================================
# CONFIGURATION: Set timeframe for analysis
# ============================================================================
TIMEFRAME = 'PRE'  # Change to 'POST' for post-decision analysis
# ============================================================================
```

Change `'PRE'` to `'POST'` to switch between:
- **PRE**: -2 to 0 seconds (anticipatory/deliberation period before submit)
- **POST**: 0 to 2 seconds (response period after submit)

### 2. Run the notebook

The notebook automatically:
- Loads the correct feature file (`features_PRE` or `features_POST`)
- Uses the correct time window from config
- Saves results to the correct output directory

---

## Available Notebooks

| Notebook | Description | Old Notebooks Replaced |
|----------|-------------|----------------------|
| `late_fusion_model.ipynb` | Main late fusion analysis | `late_fusion_model_PRE.ipynb`<br>`late_fusion_model_POST.ipynb` |
| `late_fusion_by_visit.ipynb` | Performance across experimental visits | `late_fusion_by_visit_PRE.ipynb`<br>`late_fusion_by_visit_POST.ipynb` |
| `ambiguity_group_late_fusion.ipynb` | Analysis by ambiguity level | `ambiguity_group_late_fusion_PRE.ipynb`<br>`ambiguity_group_late_fusion_POST.ipynb` |
| `ambiguity_group_late_fusion_balanced.ipynb` | Balanced sampling by ambiguity | `ambiguity_group_late_fusion_balanced_PRE.ipynb`<br>`ambiguity_group_late_fusion_balanced_POST.ipynb` |
| `reaction_time_group_late_fusion.ipynb` | Analysis by reaction time | `reaction_time_group_late_fusion_PRE.ipynb`<br>`reaction_time_group_late_fusion_POST.ipynb` |
| `reaction_time_group_late_fusion_balanced.ipynb` | Balanced sampling by RT | `reaction_time_group_late_fusion_balanced_PRE.ipynb`<br>`reaction_time_group_late_fusion_balanced_POST.ipynb` |
| `multi_seed_late_fusion.ipynb` | Robustness testing | `multi_seed_late_fusion_PRE.ipynb`<br>`multi_seed_late_fusion_POST.ipynb` |
| `subject_repeat_analysis.ipynb` | Subject-level repeatability | `subject_repeat_analysis_PRE.ipynb`<br>`subject_repeat_analysis_POST.ipynb` |

---

## Migration Guide

### Running PRE Analysis

1. Open `late_fusion_model.ipynb`
2. Verify `TIMEFRAME = 'PRE'` (should be default)
3. Run all cells
4. Results saved to: `data/results/fusion_model_results_PRE/`

### Running POST Analysis

1. Open `late_fusion_model.ipynb`
2. Change to `TIMEFRAME = 'POST'`
3. Run all cells
4. Results saved to: `data/results/fusion_model_results_POST/`

### Running Both

To generate both PRE and POST results:
1. Run notebook with `TIMEFRAME = 'PRE'`
2. Change to `TIMEFRAME = 'POST'`
3. Run notebook again
4. Both sets of results now available

---

## Old Notebooks

The original separate PRE/POST notebooks are kept in:
- `notebooks/fusion_models_PRE/` - Can be deleted after validation
- `notebooks/fusion_models_POST/` - Can be deleted after validation

**Recommended**: After confirming the consolidated notebooks work correctly, archive or delete the old separate versions.

---

## Technical Details

### How It Works

The notebooks use:
1. **`TIMEFRAME` variable**: Set to `'PRE'` or `'POST'`
2. **Dynamic paths**: `f'../../data/results/features_{TIMEFRAME}/...'`
3. **Config-based time windows**: Loaded from `config/model_params.yaml`
4. **Shared utilities**: From `src/` package

### Example Code Pattern

```python
# Set timeframe
TIMEFRAME = 'PRE'

# Load features (path changes based on TIMEFRAME)
features_path = f'../../data/results/features_{TIMEFRAME}/extracted_features_{TIMEFRAME}.pkl'
feature_data = load_features(features_path, timeframe=TIMEFRAME)

# Get time window from config
config = load_config('model_params')
time_window = config['time_windows'][f'{TIMEFRAME.lower()}_decision']
print(f"Analyzing: {time_window['start']}s to {time_window['end']}s")

# Results save to timeframe-specific directory
output_dir = f'../../data/results/fusion_model_results_{TIMEFRAME}'
```

---

## Benefits Over Separate Notebooks

| Aspect | Separate (Old) | Consolidated (New) |
|--------|---------------|-------------------|
| Code duplication | 95% duplicate | 0% duplicate |
| Maintenance | Fix bugs twice | Fix bugs once |
| Consistency | Can drift apart | Always consistent |
| Lines of code | ~8000 lines × 2 | ~8000 lines × 1 |
| Easy to compare | Need to run both | Change 1 line |

---

**Questions?** See the main project README or check `src/README.md` for utility documentation.
