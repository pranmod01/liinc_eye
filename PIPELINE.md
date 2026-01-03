# LIINC Analysis Pipeline

Complete guide to replicating the multimodal physiological analysis from raw data to final results.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Structure](#data-structure)
4. [Pipeline Steps](#pipeline-steps)
   - [Step 1: Feature Extraction](#step-1-feature-extraction)
   - [Step 2: Main Late Fusion Analysis](#step-2-main-late-fusion-analysis)
   - [Step 3: By-Visit Analysis](#step-3-by-visit-analysis)
   - [Step 4: Group Analyses](#step-4-group-analyses)
   - [Step 5: Robustness Testing](#step-5-robustness-testing)
5. [Configuration](#configuration)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline analyzes multimodal physiological data to predict investment decisions using late fusion of:
- **Physiological signals** (pupillometry: pupil diameter, velocity)
- **Behavioral data** (reaction time, decision characteristics)
- **Gaze patterns** (eye position, movements, fixations)

The analysis compares two time windows:
- **PRE-decision** (-2 to 0 seconds before submit button press): Deliberation period
- **POST-decision** (0 to 2 seconds after submit button press): Response period

---

## Prerequisites

### 1. Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Packages

See [requirements.txt](requirements.txt):
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- pyyaml
- xgboost, lightgbm

### 3. Data Files

Required input files (provided by preprocessing pipeline):
```
data/results/
├── pupil_PRE/              # Pupillometry features (PRE window)
├── pupil_POST/             # Pupillometry features (POST window)
├── eeg_PRE/                # EEG features (PRE window) [optional]
├── eeg_POST/               # EEG features (POST window) [optional]
├── behavioral/             # Behavioral data (trial-level)
├── gaze/                   # Gaze tracking data
└── session_mapping.csv     # Subject/session metadata
```

---

## Data Structure

### Feature Files

Each modality provides trial-level features:

**Pupillometry** (`pupil_PRE/`, `pupil_POST/`):
- Baseline-corrected pupil diameter (mean, std, min, max)
- Pupil velocity statistics
- Pupil acceleration features
- 13 features total per trial

**Behavioral** (`behavioral/`):
- `reaction_time`: Time from stimulus to response
- `decision_time`: Time spent deliberating
- `ev_difference`: Expected value difference between options
- `invest_variance`: Variance in investment amounts
- `ambiguity`: Ambiguity level (0, 3, 6)
- `condition_social`: Social vs non-social condition
- `risk_premium`: Risk preference measure
- 7 features total per trial

**Gaze** (`gaze/`):
- Mean/std gaze position (x, y)
- Gaze velocity and acceleration
- Fixation counts and durations
- Saccade characteristics
- 20 features total per trial

### Session Mapping

`session_mapping.csv` contains:
- `subject_id`: Unique identifier (mmdd_hhmm_userid)
- `desktop_id`: User identifier (same user across sessions)
- `team`: Team assignment
- `session`: Session number
- `date`: Session date
- `visit_number`: Experimental visit (1-4)

---

## Pipeline Steps

### Step 1: Feature Extraction

**Purpose**: Combine multimodal data into unified feature matrices for PRE and POST time windows.

#### For PRE-decision analysis:

```bash
cd notebooks/feature_extraction
jupyter notebook feature_extraction_PRE.ipynb
```

**What it does**:
1. Loads pupil features from `data/results/pupil_PRE/`
2. Loads behavioral features from `data/results/behavioral/`
3. Loads gaze features from `data/results/gaze/`
4. Applies baseline correction to physiological signals
5. Merges modalities by trial ID
6. Saves to `data/results/features_PRE/extracted_features_PRE.pkl`

**Key configuration**:
- Time window: -2.0s to 0.0s (PRE-decision)
- Baseline method: Trial-start baseline (configurable)
- Missing data: Mean imputation

#### For POST-decision analysis:

```bash
jupyter notebook feature_extraction_POST.ipynb
```

**What it does**:
- Same as PRE but uses POST time window (0.0s to 2.0s)
- Saves to `data/results/features_POST/extracted_features_POST.pkl`

**Output files**:
```
data/results/features_PRE/
└── extracted_features_PRE.pkl    # Contains:
                                   #   - merged_df (full feature matrix)
                                   #   - physio_cols (13 features)
                                   #   - behavior_cols (7 features)
                                   #   - gaze_cols (20 features)
                                   #   - metadata (extraction info)

data/results/features_POST/
└── extracted_features_POST.pkl
```

---

### Step 2: Main Late Fusion Analysis

**Purpose**: Train individual modality models and combine via late fusion to predict investment decisions.

#### Using consolidated notebooks:

```bash
cd notebooks/fusion_models
jupyter notebook late_fusion_model.ipynb
```

**Configuration** (first cell):
```python
TIMEFRAME = 'PRE'  # Change to 'POST' for post-decision analysis
```

**What it does**:
1. **Loads features** from `features_{TIMEFRAME}/`
2. **Trains individual models** (LOSO cross-validation):
   - Physiology-only Random Forest
   - Behavior-only Random Forest
   - Gaze-only Random Forest
3. **Implements late fusion**:
   - **Average Fusion**: Equal weight to all modalities
   - **Weighted Fusion**: Logistic regression meta-learner learns optimal weights
   - **Stacking**: Random Forest meta-learner
4. **Extracts modality weights** to quantify influence
5. **Saves results** to `fusion_model_results_{TIMEFRAME}/`

**Model configuration** (from `config/model_params.yaml`):
- Random Forest: 100 trees, max_depth=5, class_weight='balanced'
- LOSO CV: Leave-One-Subject-Out for subject-independent evaluation

**Output files**:
```
data/results/fusion_model_results_PRE/
├── late_fusion_model_PRE_method_comparison.csv     # Performance across methods
├── late_fusion_model_PRE_modality_weights.csv      # Weights for each fusion type
├── late_fusion_model_PRE_weighted_fusion_summary.csv
└── late_fusion_model_PRE_subject_accuracies.csv    # Per-subject results
```

**Key metrics**:
- Accuracy (mean ± SEM across subjects)
- F1-Score (weighted average)
- Modality weights (physiological, behavioral, gaze contributions)

**To run both timeframes**:
1. Set `TIMEFRAME = 'PRE'`, run all cells
2. Change to `TIMEFRAME = 'POST'`, run all cells again
3. Compare results from both output directories

---

### Step 3: By-Visit Analysis

**Purpose**: Examine how model performance changes across experimental visits.

```bash
cd notebooks/fusion_models
jupyter notebook late_fusion_by_visit.ipynb
```

**Configuration**:
```python
TIMEFRAME = 'PRE'  # or 'POST'
```

**What it does**:
1. Splits trials by `visit_number` (1-4)
2. Runs late fusion separately for each visit
3. Compares performance across visits
4. Tests for learning effects or fatigue

**Output files**:
```
data/results/fusion_model_results_PRE/
├── late_fusion_by_visit_PRE_comparison.csv         # Performance by visit
└── late_fusion_by_visit_PRE_visit{N}_*.csv         # Per-visit details
```

**Use cases**:
- Identify optimal experimental session timing
- Detect practice effects
- Assess model stability across visits

---

### Step 4: Group Analyses

#### 4a. By Ambiguity Level

**Purpose**: Analyze performance for different decision ambiguity levels.

```bash
cd notebooks/fusion_models
jupyter notebook ambiguity_group_late_fusion.ipynb
```

**Configuration**:
```python
TIMEFRAME = 'PRE'  # or 'POST'
```

**Ambiguity groups**:
- **Low** (ambiguity = 0): Clear risk distribution
- **Medium** (ambiguity = 3): Moderate uncertainty
- **High** (ambiguity = 6): Maximum uncertainty

**What it does**:
1. Splits trials into Low/Medium/High ambiguity
2. Runs late fusion for each group
3. Compares modality weights across ambiguity levels
4. Plots subject-level accuracy distributions

**Hypothesis**: Physiological signals may be more predictive under high ambiguity (emotional involvement).

**Output files**:
```
data/results/fusion_model_results_PRE/
├── ambiguity_group_late_fusion_PRE_comparison.csv
├── ambiguity_group_late_fusion_PRE_Low_weights.csv
├── ambiguity_group_late_fusion_PRE_Medium_weights.csv
└── ambiguity_group_late_fusion_PRE_High_weights.csv
```

#### 4b. By Reaction Time

**Purpose**: Examine how performance varies with decision speed.

```bash
jupyter notebook reaction_time_group_late_fusion.ipynb
```

**RT groups** (tertiles):
- **Fast**: Bottom 33% of reaction times
- **Medium**: Middle 33%
- **Slow**: Top 33%

**What it does**:
1. Splits trials into Fast/Medium/Slow RT groups
2. Runs late fusion for each group
3. Compares modality contributions by decision speed

**Hypothesis**: Fast decisions may rely more on gaze/behavior, slow on deliberative physiology.

#### 4c. Balanced Sampling

For both ambiguity and RT analyses, balanced versions ensure equal subject representation:

```bash
jupyter notebook ambiguity_group_late_fusion_balanced.ipynb
jupyter notebook reaction_time_group_late_fusion_balanced.ipynb
```

**Balancing strategy**:
- Randomly sample equal trials per subject within each group
- Prevents high-trial subjects from dominating LOSO folds
- **Trade-off**: Reduces sample size (~93% data loss) but ensures valid SEM

**When to use**:
- Standard version: Maximum power, compare group means
- Balanced version: Fair cross-subject comparison, assess individual differences

---

### Step 5: Robustness Testing

#### 5a. Multi-Seed Analysis

**Purpose**: Verify results are stable across random initializations.

```bash
cd notebooks/fusion_models
jupyter notebook multi_seed_late_fusion.ipynb
```

**What it does**:
1. Runs late fusion with 10 different random seeds (42, 43, ..., 51)
2. Measures variability in accuracy and modality weights
3. Computes convergence statistics

**Good robustness**: Low variance across seeds (<0.01 accuracy SD)

**Output files**:
```
data/results/fusion_model_results_PRE/
├── multi_seed_late_fusion_PRE_summary.csv          # Mean/SD across seeds
└── multi_seed_late_fusion_PRE_all_seeds.csv        # Per-seed results
```

#### 5b. Subject Repeatability

**Purpose**: Identify subjects with consistent vs. variable predictions.

```bash
jupyter notebook subject_repeat_analysis.ipynb
```

**What it does**:
1. Runs late fusion 50 times with different random splits
2. Calculates per-subject accuracy variance
3. Identifies "reliable" vs "noisy" subjects

**Applications**:
- Quality control (exclude unreliable subjects)
- Individual differences analysis
- Identify subjects for targeted follow-up

**Output files**:
```
data/results/fusion_model_results_PRE/
├── subject_repeat_analysis_PRE_summary.csv
└── subject_repeat_analysis_PRE_subject_stability.csv
```

---

## Configuration

All parameters centralized in `config/`:

### Model Parameters (`config/model_params.yaml`)

```yaml
random_forest:
  n_estimators: 100          # Number of trees
  max_depth: 5               # Maximum tree depth
  min_samples_split: 10      # Min samples to split node
  min_samples_leaf: 5        # Min samples per leaf
  class_weight: 'balanced'   # Handle imbalanced outcomes
  random_state: 42           # Reproducibility
  n_jobs: -1                 # Parallel processing

logistic_regression:
  max_iter: 1000
  random_state: 42
  class_weight: 'balanced'

time_windows:
  pre_decision:
    start: -2.0
    end: 0.0
    description: "Anticipatory period before submit button"
  post_decision:
    start: 0.0
    end: 2.0
    description: "Response period after submit button"
```

**To modify**:
1. Edit `config/model_params.yaml`
2. Re-run notebooks (automatically load new config)
3. No code changes needed!

### Data Paths (`config/paths.yaml`)

```yaml
pre_paths:
  features: "data/results/features_PRE/extracted_features_PRE.pkl"
  fusion_results: "data/results/fusion_model_results_PRE"

post_paths:
  features: "data/results/features_POST/extracted_features_POST.pkl"
  fusion_results: "data/results/fusion_model_results_POST"

shared:
  session_mapping: "data/results/session_mapping.csv"
  behavioral: "data/results/behavioral/"
  gaze: "data/results/gaze/"
```

---

## Output Files

### Directory Structure

After running full pipeline:

```
data/results/
├── features_PRE/
│   └── extracted_features_PRE.pkl
├── features_POST/
│   └── extracted_features_POST.pkl
├── fusion_model_results_PRE/
│   ├── late_fusion_model_PRE_*.csv                    # Main analysis
│   ├── late_fusion_by_visit_PRE_*.csv                 # By-visit
│   ├── ambiguity_group_late_fusion_PRE_*.csv          # By-ambiguity
│   ├── reaction_time_group_late_fusion_PRE_*.csv      # By-RT
│   ├── multi_seed_late_fusion_PRE_*.csv               # Robustness
│   └── subject_repeat_analysis_PRE_*.csv              # Repeatability
└── fusion_model_results_POST/
    └── [same structure as PRE]
```

### File Naming Convention

```
{analysis_name}_{TIMEFRAME}_{detail}.csv

Examples:
- late_fusion_model_PRE_method_comparison.csv
- ambiguity_group_late_fusion_POST_High_weights.csv
- multi_seed_late_fusion_PRE_summary.csv
```

### Key Results Files

**Performance summaries**:
- `*_method_comparison.csv`: Accuracy/F1 for all methods
- `*_modality_weights.csv`: Contribution of each modality
- `*_subject_accuracies.csv`: Per-subject performance

**Group analyses**:
- `*_comparison.csv`: Performance across groups
- `*_{GroupName}_weights.csv`: Modality weights for specific group

---

## Troubleshooting

### Missing Feature Files

**Error**: `FileNotFoundError: features_PRE/extracted_features_PRE.pkl`

**Solution**: Run feature extraction first:
```bash
cd notebooks/feature_extraction
jupyter notebook feature_extraction_PRE.ipynb
```

### Inconsistent Sample Sizes

**Error**: `ValueError: X and y have inconsistent sample sizes`

**Cause**: Modalities have different numbers of trials (missing data)

**Solution**: Feature extraction handles merging with inner join. Check:
```python
print(f"Physio trials: {len(physio_df)}")
print(f"Behavior trials: {len(behavior_df)}")
print(f"Gaze trials: {len(gaze_df)}")
print(f"Merged trials: {len(merged_df)}")
```

### Low Cross-Validation Scores

**Symptom**: Accuracy near chance (50%) or low F1-scores

**Possible causes**:
1. **Imbalanced classes**: Check outcome distribution
   ```python
   print(merged_df['outcome'].value_counts())
   ```
   Solution: `class_weight='balanced'` (already enabled)

2. **Insufficient subjects**: LOSO requires 10+ subjects
   ```python
   print(f"Subjects: {merged_df['subject_id'].nunique()}")
   ```

3. **High-variance features**: Check for outliers/scaling issues
   ```python
   merged_df[physio_cols].describe()
   ```

### Memory Issues

**Error**: `MemoryError` or kernel crash

**Solutions**:
1. Close other notebooks
2. Run one timeframe at a time
3. Reduce `n_estimators` in config (temporarily)
4. Use `n_jobs=1` instead of `-1`

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Add project root to path (already in notebooks):
```python
import sys
sys.path.append('../..')
```

**Error**: `ImportError: cannot import name 'weighted_late_fusion'`

**Solution**: Check file exists:
```bash
ls -la src/models/fusion.py
ls -la src/utils/
ls -la src/visualization/
```

### Visualization Issues

**Error**: Plots not displaying in Jupyter

**Solution**:
```python
%matplotlib inline  # Add to first cell
```

**Error**: `UserWarning: Tight layout not applied`

**Solution**: Already handled in plot functions, safe to ignore.

---

## Analysis Checklist

Use this to verify pipeline completion:

### Data Preparation
- [ ] Feature extraction PRE completed
- [ ] Feature extraction POST completed
- [ ] Feature files exist and load correctly
- [ ] Session mapping file loaded

### Core Analysis
- [ ] Main late fusion PRE completed
- [ ] Main late fusion POST completed
- [ ] Results saved to correct directories

### Extended Analyses
- [ ] By-visit analysis (PRE and POST)
- [ ] Ambiguity group analysis (PRE and POST)
- [ ] RT group analysis (PRE and POST)
- [ ] Balanced versions (if needed)

### Robustness
- [ ] Multi-seed analysis (PRE and POST)
- [ ] Subject repeatability (PRE and POST)

### Outputs
- [ ] All CSV files generated
- [ ] Visualizations reviewed
- [ ] Results summarized for publication

---

## Citation

If using this pipeline, please cite:

```
[Add citation when published]
```

---

## Questions?

- **Technical issues**: Check [troubleshooting](#troubleshooting) or open GitHub issue
- **Methodological questions**: See main README or paper
- **Configuration help**: Review `config/model_params.yaml` comments

**Last updated**: 2026-01-01
