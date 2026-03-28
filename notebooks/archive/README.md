# Archived Notebooks

This directory contains notebooks that represent exploratory analyses, duplicate experiments, or variants that did not contribute to the main publication. They are preserved for historical reference and reproducibility but are not part of the core analysis pipeline.

## Archive Categories

### `exploration/`
**1 notebook** - Initial data exploration and quality checks
- `check_raw_eye_data.ipynb` - Raw eye-tracking data validation

### `normalized_variants/`
**3 notebooks** - Feature normalization experiments
- `late_fusion_normalized.ipynb` - Late fusion with normalized features
- `late_fusion_normalized_feature_selection.ipynb` - Normalized + feature selection
- `late_fusion_normalized_with_eeg.ipynb` - Normalized features + EEG

**Finding:** Feature normalization did not significantly improve model performance over unnormalized features.

### `previous_trial_effects/`
**3 notebooks** - Sequential dependency analyses
- `previous_trial_effects.ipynb` - General previous trial effects
- `previous_trial_effects_by_ambiguity.ipynb` - Previous trial effects by ambiguity level
- `previous_trial_effects_by_rt.ipynb` - Previous trial effects by reaction time

**Finding:** Sequential dependencies exist but were not central to the main research question.

### `balanced_variants/`
**2 notebooks** - Class balancing experiments
- `ambiguity_group_late_fusion_balanced.ipynb` - Balanced sampling by ambiguity
- `reaction_time_group_late_fusion_balanced.ipynb` - Balanced sampling by RT

**Finding:** Class balancing did not improve performance over the natural class distribution approach used in main analyses.

### `no_temporal_features/`
**6 notebooks** - Non-temporal feature experiments
- `late_fusion_model_no_temporal.ipynb` - Base late fusion without temporal features
- `late_fusion_model_no_temporal_eeg.ipynb` - No temporal + EEG
- `ambiguity_group_late_fusion_no_temporal.ipynb` - By ambiguity, no temporal
- `ambiguity_group_late_fusion_no_temporal_eeg.ipynb` - By ambiguity, no temporal + EEG
- `reaction_time_group_late_fusion_no_temporal.ipynb` - By RT, no temporal
- `reaction_time_group_late_fusion_no_temporal_eeg.ipynb` - By RT, no temporal + EEG

**Finding:** Removing temporal dynamics from features (using only aggregate statistics) significantly reduced model performance, confirming that temporal information is critical for prediction.

### `session_level/`
**1 notebook** - Session mapping preprocessing
- `session_mapping.ipynb` - Maps participants to experimental sessions/teams

**Note:** Preprocessing artifact for data organization.

---

## Summary

**Total archived:** 16 notebooks
**Reason for archiving:** Exploratory analyses, negative results, or duplicate variants not included in main publication

These notebooks remain available for:
1. Reproducing exploratory work
2. Understanding design decisions (e.g., why normalization wasn't used)
3. Historical reference
4. Potential follow-up studies

To restore any archived notebook to active use, simply move it back to the appropriate directory under `notebooks/`.
