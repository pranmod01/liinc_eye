# Notebooks

This directory contains Jupyter notebooks for the multimodal late fusion analysis pipeline. Notebooks are organized by analysis stage with numerical prefixes indicating execution order.

**Active notebooks:** 33
**Archived notebooks:** 16 (in `archive/`)

---

## Execution Order

Notebooks are prefixed with numbers to indicate their logical execution sequence:

- **01_** - Preprocessing and feature extraction
- **02_** - Core late fusion experiments (main results)
- **03_** - EEG modality integration
- **04_** - Baseline model comparisons
- **05_** - Statistical analyses and further investigations
- **06_** - Publication figure generation
- **07_** - Visualization utilities
- **08_** - Results consolidation

---

## Directory Structure

```
notebooks/
├── preprocessing/              # 01: Data preprocessing and feature extraction
├── fusion_models/             # 02: Core late fusion experiments
├── fusion_models_eeg/         # 03: EEG integration analyses
├── other_models/              # 04: Baseline comparisons
├── further_analysis/          # 05-06: Statistical tests and publication figures
├── visualization/             # 07: Gaze visualization utilities
├── archive/                   # Archived experimental variants
└── 08_consolidate_all_results.ipynb  # Aggregate all analyses
```

---

## Notebook Categories

### 01: Preprocessing (`preprocessing/`)

**3 notebooks** - Feature extraction from raw physiological signals

| Notebook | Description |
|----------|-------------|
| `01_feature_extraction.ipynb` | Extract behavior, physiology, and gaze features from raw data |
| `01_pupil_preprocessing.ipynb` | Pupil diameter preprocessing (filtering, blink removal) |
| `01_eeg_preprocessing_visualization.ipynb` | EEG preprocessing pipeline visualization |

**Input:** Raw eye-tracking and EEG data (not in repo)
**Output:** Extracted feature CSVs for modeling

---

### 02: Core Late Fusion (`fusion_models/`)

**6 notebooks** - Main late fusion experiments with behavior + physiology + gaze

| Notebook | Description | Key Results |
|----------|-------------|-------------|
| `02_late_fusion_model.ipynb` | **PRIMARY** - Base late fusion with LOSO-CV | 69.3% accuracy (weighted fusion) |
| `02_ambiguity_group_late_fusion.ipynb` | Performance by ambiguity level (Low/Med/High) | 80.7% (Low) → 60.8% (High) |
| `02_reaction_time_group_late_fusion.ipynb` | Performance by reaction time (Fast/Slow) | 72.2% (Fast) → 63.6% (Slow) |
| `02_late_fusion_by_visit.ipynb` | Longitudinal analysis across experimental visits | Consistent performance across visits |
| `02_multi_seed_late_fusion.ipynb` | Robustness validation across random seeds | Low variance across seeds |
| `02_subject_repeat_analysis.ipynb` | Subject-level repeat reliability | Individual differences in predictability |

**Method:** Random Forest classifiers per modality + logistic regression meta-learner
**Evaluation:** Leave-One-Subject-Out Cross-Validation

---

### 03: EEG Integration (`fusion_models_eeg/`)

**5 notebooks** - Adding EEG as 4th modality

| Notebook | Description | Key Finding |
|----------|-------------|-------------|
| `03_late_fusion_with_eeg.ipynb` | 4-modality fusion (behavior + physio + gaze + EEG) | Minimal improvement (~1-7% weight) |
| `03_late_fusion_eeg_feature_comparison.ipynb` | Compare 3 EEG extraction methods | Regional (16), Regional+Channels (96), Non-temporal (96) |
| `03_late_fusion_eeg_multimodal_comparison.ipynb` | Systematic comparison of EEG contribution | EEG underperforms other modalities |
| `03_ambiguity_group_late_fusion_eeg.ipynb` | EEG fusion by ambiguity level | Minimal EEG benefit across ambiguity levels |
| `03_reaction_time_group_late_fusion_eeg.ipynb` | EEG fusion by reaction time | Minimal EEG benefit across RT groups |

**Conclusion:** EEG provides minimal predictive value despite extensive feature engineering.

---

### 04: Baseline Comparisons (`other_models/`)

**3 notebooks** - Alternative modeling approaches

| Notebook | Description |
|----------|-------------|
| `04_gradient_boosting_baseline.ipynb` | XGBoost and LightGBM models |
| `04_ensemble_baseline.ipynb` | Alternative ensemble strategies |
| `04_feature_engineering_baseline.ipynb` | Engineered feature experiments |

**Purpose:** Validate that late fusion Random Forest approach is competitive with alternatives.

---

### 05-06: Statistical Analyses (`further_analysis/`)

**14 notebooks** - Validation, feature importance, and supplementary analyses

#### Statistical Testing
| Notebook | Description |
|----------|-------------|
| `05_statistical_testing.ipynb` | **Permutation tests, bootstrap CIs, McNemar's tests** |

#### Feature Importance
| Notebook | Description |
|----------|-------------|
| `05_feature_importance_shap.ipynb` | **SHAP values for all features** |
| `05_feature_engineering_experiments.ipynb` | Feature ablation studies |

#### Temporal Dynamics
| Notebook | Description |
|----------|-------------|
| `05_temporal_dynamics.ipynb` | **Temporal evolution of predictive signals** |
| `05_temporal_dynamics_within_window.ipynb` | Fine-grained within-window dynamics |
| `05_rolling_window_post_decision.ipynb` | Post-decision rolling window analysis |
| `05_rolling_window_post_decision_dynamic.ipynb` | Dynamic post-decision trajectories |

#### Cross-Modality Analyses
| Notebook | Description |
|----------|-------------|
| `05_modality_correlations_analysis.ipynb` | **Inter-modality correlations** (0.02-0.07, low) |
| `05_cross_modality_temporal_alignment.ipynb` | Temporal alignment across modalities |

#### Behavioral Analyses
| Notebook | Description |
|----------|-------------|
| `05_behavior_choice_analysis.ipynb` | Investment choice patterns by condition |
| `05_gaze_position_by_choice.ipynb` | Eye position differences by decision |

#### Subject-Level Analyses
| Notebook | Description |
|----------|-------------|
| `05_comprehensive_subject_accuracy_analysis.ipynb` | Per-subject performance distributions |

#### Results Aggregation
| Notebook | Description |
|----------|-------------|
| `05_combine_all_results.ipynb` | Combine results across all experiments |
| `06_final_pub_figures.ipynb` | **Generate 12 publication-ready figures** |

---

### 07: Visualization (`visualization/`)

**1 notebook** - Visualization utilities

| Notebook | Description |
|----------|-------------|
| `07_gaze_data_visualization.ipynb` | Gaze trajectory visualization tools |

---

### 08: Consolidation

**1 top-level notebook** - Final results aggregation

| Notebook | Description |
|----------|-------------|
| `08_consolidate_all_results.ipynb` | Consolidate all analyses into summary CSVs |

**Output:** `data/results/combined_all_results.csv` (383 KB, 207 experiments)

---

## Archived Notebooks (`archive/`)

**16 notebooks** - Exploratory analyses and experimental variants

See [archive/README.md](archive/README.md) for details.

**Categories:**
- `exploration/` (1) - Initial data checks
- `normalized_variants/` (3) - Feature normalization experiments
- `previous_trial_effects/` (3) - Sequential dependency analyses
- `balanced_variants/` (2) - Class balancing experiments
- `no_temporal_features/` (6) - Non-temporal feature experiments
- `session_level/` (1) - Session mapping

**Reason:** Negative/null results, duplicate analyses, or not central to main publication.

---

## Running Notebooks

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

### Notebook Execution Notes

1. **Path setup:** Each notebook includes:
   ```python
   import sys
   sys.path.append('../..')  # Add project root to path
   ```

2. **Config files:** Notebooks load hyperparameters from `config/model_params.yaml`

3. **Data paths:** Set in `config/paths.yaml` (not in repo, create from template)

4. **Random seed:** All notebooks use `random_state=42` for reproducibility

5. **Outputs:** Results saved to `data/results/` with descriptive filenames

### Recommended Execution Order

For reproducing main results:

1. **Preprocessing** (if raw data available):
   - `01_feature_extraction.ipynb`

2. **Core analyses**:
   - `02_late_fusion_model.ipynb` ← **Start here for main results**
   - `02_ambiguity_group_late_fusion.ipynb`
   - `02_reaction_time_group_late_fusion.ipynb`

3. **Statistical validation**:
   - `05_statistical_testing.ipynb`
   - `05_feature_importance_shap.ipynb`

4. **Publication figures**:
   - `06_final_pub_figures.ipynb`

**Note:** Raw data not included. Feature CSVs and result outputs are preserved in `data/results/`.

---

## Notebook Standards

All notebooks follow these conventions:

1. **Markdown headers:** Clearly structured sections
2. **Imports:** All imports at top of notebook
3. **Config loading:** Use `src.utils.config` for parameters
4. **Reproducibility:** Set random seeds explicitly
5. **Output saving:** Save results to `data/results/` with descriptive names
6. **Inline documentation:** Explain key analysis decisions
7. **Visualization:** Use consistent color schemes and figure sizes

---

## Key Findings Summary

Across all 33 active notebooks:

| Finding | Evidence |
|---------|----------|
| **Weighted late fusion achieves 69.3% accuracy** | `02_late_fusion_model.ipynb` |
| **Behavioral features dominate (96.7% weight)** | All fusion notebooks |
| **Performance varies by ambiguity: 80.7% → 60.8%** | `02_ambiguity_group_late_fusion.ipynb` |
| **All models significantly > chance (p<0.0001)** | `05_statistical_testing.ipynb` |
| **Top predictor: Ambiguity level (SHAP=0.064)** | `05_feature_importance_shap.ipynb` |
| **Modalities weakly correlated (0.02-0.07)** | `05_modality_correlations_analysis.ipynb` |
| **EEG contributes minimally (1-7% weight)** | `fusion_models_eeg/` notebooks |
| **Temporal features outperform non-temporal** | `archive/no_temporal_features/` |

---

## Citation

If using these notebooks, please cite the associated publication (add citation once published).

For questions or issues, refer to project [README.md](../README.md) or contact the LIINC lab.
