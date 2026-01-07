"""
Late fusion implementations for multimodal classification.

This module provides functions for combining predictions from multiple
modalities using different fusion strategies.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats


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
