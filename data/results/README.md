# Results Directory

This directory contains all analysis outputs, including CSV result files, statistical summaries, and publication figures from the multimodal late fusion project.

**Total:** 207 CSV files + 63 figures organized into hierarchical structure

---

## Directory Structure

```
results/
├── main/                           # Core publication results
│   ├── late_fusion/               # Main late fusion experiments
│   ├── eeg_integration/           # EEG modality integration
│   ├── statistical_analyses/      # Statistical tests and validations
│   └── feature_analyses/          # Feature importance and selection
├── publication/                    # Publication-ready figures (12 files)
├── archive/                        # Archived experimental variants
│   ├── normalized_variants/       # Feature normalization experiments
│   ├── no_temporal_features/      # Non-temporal feature experiments
│   └── previous_trial_effects/    # Sequential dependency analyses
├── combined_all_results.csv        # Consolidated results from all experiments
├── combined_results_summary.csv    # Summary statistics across analyses
├── consolidated_all_analyses.csv   # Aggregated analysis metrics
└── session_mapping.csv             # Participant session/team mappings
```

---

## Main Results (`main/`)

### `late_fusion/` (3 subdirectories, ~68 files)

Core late fusion model results using behavior, physiology, and gaze modalities.

**Subdirectories:**
- `fusion_model_results_PRE/` - Pre-decision window (-2s to 0s) results
- `fusion_model_results_POST/` - Post-decision window (0s to +2s) results
- `fusion_feature_selection_PRE/` - Feature selection experiments

**File naming convention:**
- `late_fusion_model_{WINDOW}_*.csv` - Base late fusion results
- `ambiguity_group_late_fusion_{WINDOW}_{LEVEL}_*.csv` - Results by ambiguity level (Low/Medium/High)
- `reaction_time_group_late_fusion_{WINDOW}_{GROUP}_*.csv` - Results by RT group (Fast/Slow)
- `*_method_comparison.csv` - Comparison of fusion methods (average/weighted/stacking)
- `*_modality_weights.csv` - Learned weights for each modality
- `*_subject_accuracies.csv` - Per-subject LOSO-CV accuracies
- `*_weighted_fusion_summary.csv` - Summary metrics for weighted fusion

**Key results:** Main paper findings showing 69.3% accuracy with weighted late fusion.

### `eeg_integration/` (2 subdirectories, ~33 files)

EEG modality integration with multimodal fusion (behavior + physiology + gaze + EEG).

**Subdirectories:**
- `fusion_models_eeg_PRE/` - Full EEG integration results
- `fusion_model_with_eeg_PRE/` - EEG contribution analysis

**File naming convention:**
- `late_fusion_with_eeg_{WINDOW}_*.csv` - 4-modality fusion results
- `ambiguity_group_eeg_{WINDOW}_{LEVEL}_*.csv` - EEG fusion by ambiguity
- `reaction_time_group_eeg_{WINDOW}_{GROUP}_*.csv` - EEG fusion by RT
- `*_modality_info.csv` - EEG feature extraction method details

**Key finding:** EEG contributes 1-7% to fusion weights despite multiple extraction methods tested.

### `statistical_analyses/` (2 subdirectories, ~37 files)

Statistical validation, feature importance, and supplementary analyses.

**Subdirectories:**
- `analysis_outputs_PRE/` - Pre-decision window analyses
- `analysis_outputs_POST/` - Post-decision window analyses

**Key files:**
- `statistical_testing_summary_{WINDOW}.csv` - Permutation tests, bootstrap CIs, McNemar's tests
- `mcnemar_test_summary_{WINDOW}.csv` - Pairwise model comparisons
- `behavior_choice_analysis_summary.csv` - Investment choice patterns
- `comprehensive_subject_accuracy_*.csv` - Subject-level performance distributions
- `modality_correlation_matrix_{WINDOW}.csv` - Cross-modality correlations
- `shap_importance_*.csv` - SHAP feature importance values
- `temporal_dynamics_*.csv` - Temporal evolution of predictive signals
- `gaze_position_by_choice_*.csv` - Eye position differences by decision

**Key figures (PNG/PDF):**
- Statistical test visualizations
- SHAP importance plots
- Temporal dynamics plots
- Subject accuracy distributions

### `feature_analyses/` (currently empty)

Reserved for future feature engineering and selection analyses.

---

## Publication Figures (`publication/`)

**12 publication-ready figures** (PDF + PNG formats)

| Figure | Description |
|--------|-------------|
| `figure1_late_fusion_performance.*` | Overall model performance comparison |
| `figure2_ambiguity_performance.*` | Performance by ambiguity level |
| `figure2_ambiguity_rt_combined.*` | Combined ambiguity and RT analysis |
| `figure3_reaction_time_performance.*` | Performance by reaction time group |
| `figure3_feature_importance.*` / `figure4_feature_importance.*` | SHAP feature importance rankings |

All figures are vector-quality PDFs with corresponding PNGs for presentations.

---

## Archived Results (`archive/`)

Experimental variants that did not contribute to main publication but are preserved for reproducibility.

### `normalized_variants/` (2 subdirectories, ~10 files)

Feature normalization experiments (z-score standardization across subjects).

**Subdirectories:**
- `fusion_model_normalized_PRE/` - Pre-decision normalized results
- `fusion_model_normalized_POST/` - Post-decision normalized results

**Finding:** Normalization did not improve performance over unnormalized features.

### `no_temporal_features/` (2 subdirectories, ~61 files)

Experiments removing temporal dynamics from features (using only aggregate statistics).

**Subdirectories:**
- `fusion_model_results_no_temporal_PRE/` - Base no-temporal results
- `fusion_model_results_no_temporal_eeg_PRE/` - No-temporal + EEG

**File naming:**
- `*_no_temporal_*` - Non-temporal feature results
- `*_excluded_features.csv` - Lists temporal features that were removed

**Finding:** Removing temporal dynamics significantly reduced performance, confirming temporal information is critical.

### `previous_trial_effects/` (1 subdirectory, ~15 files)

Sequential dependency and trial-to-trial effects analyses.

**Subdirectory:**
- `previous_trial_effects_PRE/`

**File naming:**
- `feature_importance_*.csv` - Feature importance for different trial contexts
- `previous_trial_*.csv` - Sequential dependency metrics

**Finding:** Sequential effects exist but were not central to main research question.

---

## Top-Level Summary Files

### `combined_all_results.csv` (383 KB)
Comprehensive dataset combining results from all experiments across notebooks. Each row represents one experimental configuration with:
- Model parameters
- Performance metrics (accuracy, F1, precision, recall)
- Modality weights
- Experimental conditions (window, ambiguity, RT group)

### `combined_results_summary.csv`
Summary statistics across all analyses (means, SDs, CIs).

### `consolidated_all_analyses.csv`
Aggregated metrics for quick reference and meta-analysis.

### `session_mapping.csv`
Maps participants to experimental sessions, teams, and visits for longitudinal analysis.

---

## File Naming Conventions

### Window Suffixes
- `*_PRE` - Pre-decision window (-2s to 0s before submit button)
- `*_POST` - Post-decision window (0s to +2s after submit button)

### Group Suffixes
- `*_Low` / `*_Medium` / `*_High` - Ambiguity level
- `*_Fast` / `*_Slow` - Reaction time group (median split)

### File Types
- `*_method_comparison.csv` - Performance across fusion methods
- `*_modality_weights.csv` - Learned modality contribution weights
- `*_subject_accuracies.csv` - Per-subject LOSO-CV performance
- `*_summary.csv` - Aggregated summary statistics
- `*_comparison.csv` - Cross-condition comparisons
- `*_feature_importance.csv` - Feature importance rankings
- `*.png` / `*.pdf` - Figures in raster/vector formats

---

## Usage

### Reproducing Main Results
1. Primary results are in `main/late_fusion/fusion_model_results_PRE/`
2. Key file: `late_fusion_model_PRE_weighted_fusion_summary.csv`
3. Publication figures: `publication/figure*.pdf`

### Statistical Validation
1. Permutation tests: `main/statistical_analyses/analysis_outputs_PRE/statistical_testing_summary_PRE.csv`
2. Feature importance: `main/statistical_analyses/analysis_outputs_PRE/shap_importance_*.csv`

### Understanding Experimental Variants
1. See `archive/*/README.md` (if created) or notebook archive
2. Archived results document negative/null findings

---

## Notes

- All CSV files use standard format (comma-separated, UTF-8 encoding)
- Subject IDs are anonymized
- LOSO-CV = Leave-One-Subject-Out Cross-Validation
- PRE-decision window performed better than POST-decision (main focus of paper)
- Results generated using Python 3.13 with scikit-learn Random Forests
- Random seed = 42 for reproducibility

---

## Citation

If using these results, please cite the associated publication (add citation once published).

For questions about specific files or analyses, refer to the corresponding notebooks in `notebooks/`.
