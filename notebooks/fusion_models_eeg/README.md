# EEG Late Fusion Analysis

This directory contains notebooks for analyzing EEG features in multimodal late fusion.

## Notebooks

### 1. `late_fusion_with_eeg.ipynb` (Original)
Standard late fusion with regional EEG features (16 features: 4 bands × 4 regions).

**Purpose**: Baseline analysis with original regional EEG approach

**Features**:
- Regional band power averages (Frontal, Central, Parietal, Occipital)
- Late fusion with physiology, behavior, gaze, and EEG
- Modality weight analysis

**Key Finding**: EEG contributes only 1.1% to weighted fusion

---

### 2. `late_fusion_eeg_feature_comparison.ipynb` (New - **USE THIS**)
**Comprehensive comparison of 3 EEG feature extraction approaches in multimodal late fusion.**

**EEG Feature Sets Compared**:
1. **Regional** (16 features): Original regional averages
2. **Regional + Channels** (96 features): Regional averages + individual channel band powers
3. **Non-temporal** (96 features): Power + temporal dynamics + lateralization indices

**Analysis Pipeline**:
1. **Feature Listings**: Display all features for each approach
2. **Single-Modality Evaluation**: Test EEG features alone
3. **Multimodal Late Fusion**: Full fusion with all modalities (physio, behavior, gaze, EEG)
4. **Modality Contributions**: Compare EEG weights across feature sets
5. **Feature Importance**: Identify most predictive EEG features

**Key Questions Answered**:
- Which EEG feature approach gives best overall fusion accuracy?
- Which EEG approach has highest modality contribution?
- Which specific EEG features are most important?
- Does adding channel-level or temporal information improve beyond regional averages?

---

### 3. `late_fusion_eeg_multimodal_comparison.ipynb` (Alternative)
Similar comprehensive analysis, but in a separate notebook.

**Note**: Content is now integrated into `late_fusion_eeg_feature_comparison.ipynb`. Use that instead.

---

## Running the Analysis

### Step 1: Extract EEG Features (All 3 Types)
```bash
# Run all extraction scripts
./scripts/test_eeg_extraction.sh

# OR run individually
python3 scripts/extract_eeg_features.py
python3 scripts/extract_eeg_features_with_channels.py
python3 scripts/extract_eeg_features_non_temporal.py
```

### Step 2: Run Comparison Notebook
```bash
jupyter notebook notebooks/fusion_models_eeg/late_fusion_eeg_feature_comparison.ipynb
```

---

## Expected Outputs

### Files Generated

In `data/results/main/eeg_integration/fusion_models_eeg_PRE/`:

1. **Single-Modality Results**:
   - `eeg_feature_set_comparison_single_modality.csv` - EEG-only performance
   - `eeg_feature_importances_regional.csv` - Regional feature importances
   - `eeg_feature_importances_regional_channels.csv` - Regional+Channels importances
   - `eeg_feature_importances_non-temporal.csv` - Non-temporal importances

2. **Multimodal Fusion Results**:
   - `eeg_feature_set_comparison_multimodal.csv` - Fusion performance by EEG type
   - `eeg_modality_weights_by_feature_set.csv` - Modality contributions
   - `multimodal_regional_subject_accuracies.csv` - Subject-level results (Regional)
   - `multimodal_regional_channels_subject_accuracies.csv` - Subject-level (Regional+Channels)
   - `multimodal_non-temporal_subject_accuracies.csv` - Subject-level (Non-temporal)

---

## Interpretation Guide

### Comparing EEG Feature Sets

**1. Overall Fusion Accuracy**
- Look at `eeg_feature_set_comparison_multimodal.csv`
- Which EEG feature set gives highest weighted fusion accuracy?
- Is the difference significant (> 2 × SEM)?

**2. EEG Modality Contribution**
- Check `eeg_modality_weights_by_feature_set.csv`
- Which EEG approach has highest weight in weighted fusion?
- Does EEG contribution increase beyond baseline 1.1%?

**3. Feature Importance**
- Review `eeg_feature_importances_*.csv` files
- Which specific features are most predictive?
- For Regional+Channels: Are specific channels more important than regional averages?
- For Non-temporal: Do temporal dynamics or lateralization improve prediction?

### Key Metrics

- **Baseline (Regional EEG)**: 69.8% accuracy, 1.1% EEG weight
- **Target**: Improve EEG contribution to >2% while maintaining/improving fusion accuracy
- **Behavioral dominance**: Behavior alone achieves 65% accuracy, dominates fusion (95.8%)

---

## Next Steps Based on Results

### If EEG Contribution Increases
✅ Use best EEG feature set for final analyses
✅ Investigate which specific features drive improvement
✅ Consider feature engineering based on important features

### If EEG Contribution Remains Low (<2%)
⚠️ EEG signal may not capture decision-making information
⚠️ Possible issues:
  - Signal quality (check preprocessing)
  - Time alignment (verify EEG window matches decision period)
  - Behavioral dominance (RT/confidence already capture decision info)
⚠️ Consider:
  - Connectivity features (phase-locking, coherence)
  - Event-related potentials (ERPs)
  - Time-frequency decomposition
  - Source localization

---

## Feature Set Details

### Regional (16 features)
- Delta_Frontal, Delta_Central, Delta_Parietal, Delta_Occipital
- Theta_Frontal, Theta_Central, Theta_Parietal, Theta_Occipital
- Alpha_Frontal, Alpha_Central, Alpha_Parietal, Alpha_Occipital
- Beta_Frontal, Beta_Central, Beta_Parietal, Beta_Occipital

### Regional + Channels (96 features)
- All 16 regional features above
- Plus 80 individual channel features (4 bands × 20 channels):
  - Channels: Fp1, Fp2, F3, Fz, F4, F7, F8, C3, Cz, C4, T3, T4, P3, Pz, P4, T5, T6, O1, O2, POz

### Non-temporal (96 features)
- **16 power features**: Band power averaged across full epoch
- **48 temporal dynamics**: Mean, std, slope for each band/region
- **32 lateralization indices**: L-R asymmetry for 8 electrode pairs × 4 bands
  - Pairs: F7-F8, F3-F4, Fp1-Fp2, C3-C4, T3-T4, T5-T6, P3-P4, O1-O2

---

## Contact

For questions about this analysis, see:
- `scripts/README_EEG_FEATURES.md` - EEG feature extraction documentation
- Original EEG notebook: `late_fusion_with_eeg.ipynb`
