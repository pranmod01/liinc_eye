# Multimodal Late Fusion for Predicting Human Decision-Making Under Risk and Ambiguity

Research conducted at the [Laboratory for Intelligent Imaging and Neural Computing (LIINC)](https://liinc.bme.columbia.edu/), Columbia University.

## Overview

This project investigates whether multimodal physiological signals -- EEG, eye tracking (gaze and pupil dynamics), and behavioral measures -- can predict human investment decisions in a lottery-based task. We develop a **late fusion framework** that combines modality-specific classifiers to leverage complementary information across signal types.

Participants made binary investment decisions (invest or not) in trials varying in risk and ambiguity levels. We extract features from pre-decision and post-decision time windows and evaluate how different modalities and their combinations contribute to prediction accuracy.

## Key Results

- **Weighted late fusion** achieves **69.3% accuracy** (F1: 0.679) on leave-one-subject-out cross-validation, outperforming any single modality
- Behavioral features (reaction time, trial parameters) are the strongest individual predictors (64.7% accuracy)
- Fusion with physiological and gaze signals provides a consistent boost over behavior alone
- Model performance varies systematically with **ambiguity level** and **reaction time**, suggesting the model captures meaningful decision-process differences

## Repository Structure

```
├── config/                  # Model hyperparameters and data paths (YAML)
├── data/results/            # Output CSVs and publication figures
├── notebooks/
│   ├── preprocessing/       # EEG and pupil data preprocessing
│   ├── fusion_models/       # Late fusion experiments (main results)
│   ├── further_analysis/    # SHAP, statistical tests, temporal dynamics
│   ├── other_models/        # Baseline comparisons (XGBoost, ensemble)
│   ├── exploration/         # Exploratory data checks
│   └── visualization/       # Gaze data visualization
├── scripts/                 # Feature extraction scripts
├── src/                     # Shared library code
│   ├── models/              # Late fusion implementations
│   ├── utils/               # I/O, config loading, validation
│   └── visualization/       # Reusable plotting functions
└── requirements.txt
```

## Tech Stack

**Languages:** Python 3.13, Jupyter notebooks

**Core libraries:** scikit-learn, PyTorch, XGBoost, LightGBM, SHAP, NumPy, pandas, SciPy, statsmodels, matplotlib, seaborn

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notebooks expect the project root on the Python path. Each notebook includes:
```python
import sys
sys.path.append('../..')
```

## Data

Raw EEG and eye-tracking data are not included in this repository due to size. Extracted features and model outputs (CSVs and figures) are included under `data/results/` so that analysis notebooks can be reviewed with their outputs intact.

## Notebooks Guide

| Notebook | Description |
|---|---|
| `preprocessing/feature_extraction.ipynb` | Feature extraction from raw physiological signals |
| `fusion_models/late_fusion_model.ipynb` | Main late fusion experiment with LOSO-CV |
| `fusion_models/ambiguity_group_late_fusion.ipynb` | Performance breakdown by ambiguity level |
| `fusion_models/reaction_time_group_late_fusion.ipynb` | Performance breakdown by reaction time |
| `fusion_models/normalized/late_fusion_normalized.ipynb` | Fusion with normalized features |
| `further_analysis/feature_importance_shap.ipynb` | SHAP feature importance analysis |
| `further_analysis/statistical_testing.ipynb` | Bootstrap CIs, permutation tests, McNemar's test |
| `further_analysis/temporal_dynamics.ipynb` | Temporal evolution of predictive signals |
| `further_analysis/final_pub_figures.ipynb` | Publication-ready figures |
