"""
Data validation utilities.
"""

import numpy as np
import pandas as pd


def validate_features(merged_df, timeframe='PRE', verbose=True):
    """
    Validate loaded feature dataframe.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        Feature dataframe to validate.
    timeframe : str, default='PRE'
        Either 'PRE' or 'POST', for informative messages.
    verbose : bool, default=True
        Whether to print validation results.

    Raises
    ------
    AssertionError
        If validation fails.

    Examples
    --------
    >>> validate_features(merged_df, 'PRE')
    ✓ Validation passed: 12511 trials, 97 subjects
    """
    # Check required columns
    assert 'subject_id' in merged_df.columns, "Missing 'subject_id' column"
    assert 'outcome' in merged_df.columns, "Missing 'outcome' column"

    # Check for NaN in subject_id
    assert merged_df['subject_id'].notna().all(), "NaN values found in subject_id"

    # Check outcome values
    unique_outcomes = merged_df['outcome'].unique()
    assert set(unique_outcomes).issubset({0, 1}), \
        f"Outcome must be 0 or 1, found: {unique_outcomes}"

    # Check sufficient subjects for LOSO CV
    n_subjects = merged_df['subject_id'].nunique()
    assert n_subjects >= 10, \
        f"Need at least 10 subjects for LOSO CV, got {n_subjects}"

    # Check sufficient trials per subject
    trials_per_subject = merged_df.groupby('subject_id').size()
    min_trials = trials_per_subject.min()
    assert min_trials >= 1, \
        f"All subjects must have at least 1 trial, minimum found: {min_trials}"

    if verbose:
        print(f"✓ Validation passed ({timeframe}): "
              f"{len(merged_df)} trials, {n_subjects} subjects")
        print(f"  Trials per subject: min={min_trials}, "
              f"max={trials_per_subject.max()}, "
              f"mean={trials_per_subject.mean():.1f}")
        print(f"  Outcome balance: "
              f"{(merged_df['outcome']==0).sum()} keep / "
              f"{(merged_df['outcome']==1).sum()} invest")


def validate_modality_features(X_modalities, y, subjects, modality_names):
    """
    Validate modality feature arrays before fusion.

    Parameters
    ----------
    X_modalities : list of np.ndarray
        List of feature matrices.
    y : np.ndarray
        Target labels.
    subjects : np.ndarray
        Subject IDs.
    modality_names : list of str
        Names of modalities.

    Raises
    ------
    AssertionError
        If validation fails.
    """
    n_samples = len(y)

    # Check all modalities have same number of samples
    for i, (X, name) in enumerate(zip(X_modalities, modality_names)):
        assert len(X) == n_samples, \
            f"Modality '{name}' has {len(X)} samples, expected {n_samples}"

        # Check no NaN values
        assert not np.isnan(X).any(), \
            f"Modality '{name}' contains NaN values"

        # Check no infinite values
        assert not np.isinf(X).any(), \
            f"Modality '{name}' contains infinite values"

    # Check subjects array
    assert len(subjects) == n_samples, \
        f"subjects has {len(subjects)} elements, expected {n_samples}"

    print(f"✓ Modality validation passed:")
    for name, X in zip(modality_names, X_modalities):
        print(f"  {name}: {X.shape[0]} samples × {X.shape[1]} features")


def validate_loso_split(X, y, subjects, min_subjects=10):
    """
    Validate that data is suitable for LOSO cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    subjects : np.ndarray
        Subject IDs.
    min_subjects : int, default=10
        Minimum number of subjects required.

    Raises
    ------
    AssertionError
        If validation fails.
    """
    n_subjects = len(np.unique(subjects))
    assert n_subjects >= min_subjects, \
        f"Need at least {min_subjects} subjects for LOSO CV, got {n_subjects}"

    # Check that each subject has both classes
    subjects_with_single_class = []
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        subject_outcomes = y[subject_mask]
        if len(np.unique(subject_outcomes)) < 2:
            subjects_with_single_class.append(subject)

    if subjects_with_single_class:
        print(f"⚠️  Warning: {len(subjects_with_single_class)} subjects have only one class:")
        print(f"   This may cause issues in some folds")
        if len(subjects_with_single_class) <= 5:
            print(f"   Subjects: {subjects_with_single_class}")

    print(f"✓ LOSO validation passed: {n_subjects} subjects")
