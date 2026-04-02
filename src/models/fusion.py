"""
Late fusion implementations for multimodal classification.

This module provides functions for combining predictions from multiple
modalities using different fusion strategies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats


def _cohens_d_two_sample(group1, group2):
    """Cohen's d for two independent samples."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def _optimal_bins_from_train(bin_df, y, feature_names, n_bins, train_idx,
                             min_per_class=10):
    """
    For each feature, choose the bin with largest |Cohen's d| using only train_idx rows.
    Falls back to bin 0 if no bin has enough data in that class.
    """
    train_idx = np.asarray(train_idx)
    y_tr = y[train_idx]
    invest_idx = train_idx[y_tr == 1]
    keep_idx = train_idx[y_tr == 0]
    optimal = {}
    for feat in feature_names:
        best_bin = 0
        best_abs_d = -1.0
        for b in range(n_bins):
            col = f'{feat}_bin{b}'
            if col not in bin_df.columns:
                continue
            g1 = bin_df.iloc[invest_idx][col].dropna().to_numpy(dtype=float)
            g2 = bin_df.iloc[keep_idx][col].dropna().to_numpy(dtype=float)
            if len(g1) < min_per_class or len(g2) < min_per_class:
                continue
            d = _cohens_d_two_sample(g1, g2)
            if abs(d) > best_abs_d:
                best_abs_d = abs(d)
                best_bin = b
        optimal[feat] = best_bin
    return optimal


def _physio_matrix_from_bins(bin_df, feature_names, optimal_map):
    """Stack selected-bin columns into (n_samples, n_features)."""
    n = len(bin_df)
    X = np.zeros((n, len(feature_names)), dtype=float)
    for j, feat in enumerate(feature_names):
        b = optimal_map[feat]
        col = f'{feat}_bin{b}'
        if col in bin_df.columns:
            X[:, j] = pd.to_numeric(bin_df[col], errors='coerce').to_numpy()
        else:
            X[:, j] = np.nan
    return X


def weighted_late_fusion(X_modalities, y, subjects, modality_names,
                         n_estimators=100, max_depth=5,
                         min_samples_split=10, min_samples_leaf=5,
                         class_weight='balanced', random_state=42,
                         fusion_method='weighted'):
    """
    Weighted late fusion using LOSO cross-validation.

    Trains separate Random Forest models for each modality, then combines their
    predictions using a meta-learner (logistic regression or Random Forest).

    Parameters
    ----------
    X_modalities : list of np.ndarray
        List of feature matrices, one per modality. Each has shape (n_samples, n_features).
    y : np.ndarray
        Target labels, shape (n_samples,).
    subjects : np.ndarray
        Subject IDs for each sample, shape (n_samples,).
    modality_names : list of str
        Names of modalities (e.g., ['Physiology', 'Behavior', 'Gaze']).
    n_estimators : int, default=100
        Number of trees in Random Forest base models.
    max_depth : int, default=5
        Maximum depth of Random Forest trees.
    min_samples_split : int, default=10
        Minimum samples required to split a node.
    min_samples_leaf : int, default=5
        Minimum samples required in a leaf node.
    class_weight : str or dict, default='balanced'
        Class weighting strategy for Random Forest models.
    random_state : int, default=42
        Random seed for reproducibility.
    fusion_method : str, default='weighted'
        Fusion strategy:
        - 'average': Simple average of probabilities
        - 'weighted': Learn weights via logistic regression meta-learner
        - 'stacking': Random Forest meta-learner on probability predictions

    Returns
    -------
    dict
        Dictionary containing:
        - 'accuracy_mean' : float, mean accuracy across subjects
        - 'accuracy_sem' : float, standard error of the mean for accuracy
        - 'accuracy_std' : float, standard deviation of accuracy
        - 'accuracy_per_subject' : np.ndarray, accuracy for each subject
        - 'f1_mean' : float, mean F1-score across subjects
        - 'f1_sem' : float, standard error of the mean for F1-score
        - 'f1_std' : float, standard deviation of F1-score
        - 'f1_per_subject' : np.ndarray, F1-score for each subject
        - 'weights' : np.ndarray, learned fusion weights (normalized)
        - 'modality_names' : list, names of modalities
        - 'n_trials' : int, total number of trials
        - 'n_subjects' : int, number of unique subjects
        - 'predictions' : list, all predictions
        - 'y_true' : list, all true labels
        - 'subject_accs' : dict, accuracy per subject ID
        - 'subject_f1s' : dict, F1-score per subject ID

    Examples
    --------
    >>> X_physio = np.random.rand(1000, 13)
    >>> X_behavior = np.random.rand(1000, 7)
    >>> X_gaze = np.random.rand(1000, 20)
    >>> y = np.random.randint(0, 2, 1000)
    >>> subjects = np.repeat(np.arange(100), 10)
    >>>
    >>> results = weighted_late_fusion(
    ...     [X_physio, X_behavior, X_gaze],
    ...     y,
    ...     subjects,
    ...     ['Physiology', 'Behavior', 'Gaze']
    ... )
    >>> print(f"Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_sem']:.3f}")
    >>> for name, weight in zip(results['modality_names'], results['weights']):
    ...     print(f"{name}: {weight:.3f}")
    """

    logo = LeaveOneGroupOut()

    # Create base models for each modality
    base_models = [
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1  # Use all cores
        )
        for _ in X_modalities
    ]

    # Store subject-level results
    subject_accs = {}
    subject_f1s = {}
    all_weights = []
    preds_all = []
    y_true_all = []

    # LOSO cross-validation
    for train_idx, test_idx in logo.split(X_modalities[0], y, subjects):
        # Get training and test data
        y_train, y_test = y[train_idx], y[test_idx]
        train_subjects = subjects[train_idx]

        # For meta-learner training, we need unbiased predictions on training data
        # Use nested 5-fold CV to get out-of-fold predictions for train_idx
        # (using k-fold instead of LOSO for computational efficiency)
        if fusion_method in ['weighted', 'stacking']:
            train_probs = np.zeros((len(train_idx), len(X_modalities)))

            # Nested 5-fold within training set
            gkf_inner = GroupKFold(n_splits=5)
            for inner_train_idx, inner_val_idx in gkf_inner.split(
                X_modalities[0][train_idx], y_train, train_subjects
            ):
                # Convert to absolute indices
                abs_inner_train = train_idx[inner_train_idx]
                abs_inner_val = train_idx[inner_val_idx]

                # Train base models on inner training set, predict on inner validation
                # Use fewer estimators for inner loop for computational efficiency
                for mod_i, X in enumerate(X_modalities):
                    model = RandomForestClassifier(
                        n_estimators=min(50, n_estimators),  # Reduce for inner loop
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=-1  # Use all cores
                    )
                    model.fit(X[abs_inner_train], y[abs_inner_train])
                    # Store out-of-fold predictions for this modality
                    train_probs[inner_val_idx, mod_i] = model.predict_proba(
                        X[abs_inner_val]
                    )[:, 1]
        else:
            # For average fusion, we don't need special handling
            train_probs = []

        # Train base models on full training set and get test predictions
        test_probs = []
        for X, model in zip(X_modalities, base_models):
            X_train, X_test = X[train_idx], X[test_idx]
            model.fit(X_train, y_train)
            test_probs.append(model.predict_proba(X_test)[:, 1])

        test_probs = np.column_stack(test_probs)

        # Fusion strategy
        if fusion_method == 'average':
            # Simple average of probabilities
            y_pred = (np.mean(test_probs, axis=1) > 0.5).astype(int)
            weights = np.ones(len(X_modalities)) / len(X_modalities)

        elif fusion_method == 'weighted':
            # Logistic regression meta-learner trained on unbiased predictions
            meta = LogisticRegression(random_state=random_state, max_iter=1000)
            meta.fit(train_probs, y_train)
            weights = meta.coef_[0]
            y_pred = meta.predict(test_probs)

        elif fusion_method == 'stacking':
            # Random Forest meta-learner trained on unbiased predictions
            meta = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                class_weight=class_weight,
                random_state=random_state
            )
            meta.fit(train_probs, y_train)
            weights = meta.feature_importances_
            y_pred = meta.predict(test_probs)

        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}. "
                           f"Use 'average', 'weighted', or 'stacking'.")

        # Store subject-level metrics
        test_subject = subjects[test_idx][0]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        subject_accs[test_subject] = acc
        subject_f1s[test_subject] = f1
        all_weights.append(weights)
        preds_all.extend(y_pred)
        y_true_all.extend(y_test)

    # Convert to arrays
    subject_acc_values = np.array(list(subject_accs.values()))
    subject_f1_values = np.array(list(subject_f1s.values()))

    # Average weights across folds
    avg_weights = np.mean(all_weights, axis=0)

    # Normalize weights
    if fusion_method == 'weighted':
        # Softmax normalization for logistic regression coefficients
        norm_weights = np.exp(avg_weights) / np.sum(np.exp(avg_weights))
    else:
        # L1 normalization for other methods
        norm_weights = avg_weights / np.sum(avg_weights)

    return {
        'accuracy_mean': np.mean(subject_acc_values),
        'accuracy_sem': stats.sem(subject_acc_values),
        'accuracy_std': np.std(subject_acc_values),
        'accuracy_per_subject': subject_acc_values,
        'f1_mean': np.mean(subject_f1_values),
        'f1_sem': stats.sem(subject_f1_values),
        'f1_std': np.std(subject_f1_values),
        'f1_per_subject': subject_f1_values,
        'weights': norm_weights,
        'modality_names': modality_names,
        'n_trials': len(y),
        'n_subjects': len(subject_accs),
        'predictions': preds_all,
        'y_true': y_true_all,
        'subject_accs': subject_accs,
        'subject_f1s': subject_f1s
    }


def weighted_late_fusion_train_optimal_bins(
        bin_df,
        physio_feature_names,
        n_bins,
        X_behavior,
        X_gaze,
        y,
        subjects,
        modality_names,
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        fusion_method='weighted',
        min_per_class=10):
    """
    Late fusion with the same LOSO + nested GroupKFold stacking as weighted_late_fusion.

    The first modality is built from wide columns ``{feature}_bin{b}`` in ``bin_df``.
    For each LOSO fold, the bin with largest |Cohen's d| (INVEST vs KEEP) per feature
    is chosen using **training subjects only**, then applied to all trials in that fold.
    Missing physiology values are imputed (mean) fit on training rows only.

    Parameters
    ----------
    bin_df : pandas.DataFrame
        One row per trial, same order as ``y`` / ``subjects`` / behavior / gaze arrays.
    physio_feature_names : sequence of str
        Base feature names (e.g. pupil_mean); columns ``name_bin0`` ... must exist.
    n_bins : int
        Number of time bins (used when scanning ``{name}_bin{b}``).
    X_behavior, X_gaze : np.ndarray
        Shape (n_samples, n_features); typically already imputed to match the notebook
        baseline for behavior and gaze.
    y, subjects : np.ndarray
        Labels and group ids per row.

    Returns
    -------
    dict
        Same keys as ``weighted_late_fusion``.
    """
    if len(modality_names) != 3:
        raise ValueError('modality_names must have length 3 '
                         '(physio from bins, behavior, gaze).')

    bin_df = bin_df.reset_index(drop=True)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    n_samples = len(y)
    physio_feature_names = list(physio_feature_names)

    if len(bin_df) != n_samples:
        raise ValueError(
            f'bin_df length ({len(bin_df)}) must match y ({n_samples}).')
    if X_behavior.shape[0] != n_samples or X_gaze.shape[0] != n_samples:
        raise ValueError('X_behavior and X_gaze must have n_samples rows.')

    logo = LeaveOneGroupOut()
    base_models = [
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1)
        for _ in range(3)
    ]

    subject_accs = {}
    subject_f1s = {}
    all_weights = []
    preds_all = []
    y_true_all = []

    for train_idx, test_idx in logo.split(np.zeros(n_samples), y, subjects):
        y_train, y_test = y[train_idx], y[test_idx]
        train_subjects = subjects[train_idx]

        optimal_map = _optimal_bins_from_train(
            bin_df, y, physio_feature_names, n_bins, train_idx,
            min_per_class=min_per_class)
        X_phys_raw = _physio_matrix_from_bins(
            bin_df, physio_feature_names, optimal_map)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_phys_raw[train_idx])
        X_phys = imputer.transform(X_phys_raw)

        X_modalities = [X_phys, X_behavior, X_gaze]

        if fusion_method in ['weighted', 'stacking']:
            train_probs = np.zeros((len(train_idx), len(X_modalities)))
            gkf_inner = GroupKFold(n_splits=5)
            for inner_train_idx, inner_val_idx in gkf_inner.split(
                    X_modalities[0][train_idx], y_train, train_subjects):
                abs_inner_train = train_idx[inner_train_idx]
                abs_inner_val = train_idx[inner_val_idx]
                for mod_i, X in enumerate(X_modalities):
                    model = RandomForestClassifier(
                        n_estimators=min(50, n_estimators),
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=-1)
                    model.fit(X[abs_inner_train], y[abs_inner_train])
                    train_probs[inner_val_idx, mod_i] = model.predict_proba(
                        X[abs_inner_val])[:, 1]
        else:
            train_probs = []

        test_probs = []
        for X, model in zip(X_modalities, base_models):
            X_tr, X_te = X[train_idx], X[test_idx]
            model.fit(X_tr, y_train)
            test_probs.append(model.predict_proba(X_te)[:, 1])

        test_probs = np.column_stack(test_probs)

        if fusion_method == 'average':
            y_pred = (np.mean(test_probs, axis=1) > 0.5).astype(int)
            weights = np.ones(len(X_modalities)) / len(X_modalities)
        elif fusion_method == 'weighted':
            meta = LogisticRegression(random_state=random_state, max_iter=1000)
            meta.fit(train_probs, y_train)
            weights = meta.coef_[0]
            y_pred = meta.predict(test_probs)
        elif fusion_method == 'stacking':
            meta = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                class_weight=class_weight,
                random_state=random_state)
            meta.fit(train_probs, y_train)
            weights = meta.feature_importances_
            y_pred = meta.predict(test_probs)
        else:
            raise ValueError(
                f"Unknown fusion_method: {fusion_method}. "
                f"Use 'average', 'weighted', or 'stacking'.")

        test_subject = subjects[test_idx][0]
        subject_accs[test_subject] = accuracy_score(y_test, y_pred)
        subject_f1s[test_subject] = f1_score(
            y_test, y_pred, average='weighted', zero_division=0)
        all_weights.append(weights)
        preds_all.extend(y_pred)
        y_true_all.extend(y_test)

    subject_acc_values = np.array(list(subject_accs.values()))
    subject_f1_values = np.array(list(subject_f1s.values()))
    avg_weights = np.mean(all_weights, axis=0)

    if fusion_method == 'weighted':
        norm_weights = np.exp(avg_weights) / np.sum(np.exp(avg_weights))
    else:
        norm_weights = avg_weights / np.sum(avg_weights)

    return {
        'accuracy_mean': np.mean(subject_acc_values),
        'accuracy_sem': stats.sem(subject_acc_values),
        'accuracy_std': np.std(subject_acc_values),
        'accuracy_per_subject': subject_acc_values,
        'f1_mean': np.mean(subject_f1_values),
        'f1_sem': stats.sem(subject_f1_values),
        'f1_std': np.std(subject_f1_values),
        'f1_per_subject': subject_f1_values,
        'weights': norm_weights,
        'modality_names': modality_names,
        'n_trials': len(y),
        'n_subjects': len(subject_accs),
        'predictions': preds_all,
        'y_true': y_true_all,
        'subject_accs': subject_accs,
        'subject_f1s': subject_f1s
    }
