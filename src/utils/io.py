"""
Input/Output utilities for loading and saving data.
"""

import pickle
import pandas as pd
from pathlib import Path


def load_features(filepath, timeframe='PRE'):
    """
    Load extracted features with error handling and validation.

    Parameters
    ----------
    filepath : str or Path
        Path to the pickle file containing extracted features.
    timeframe : str, default='PRE'
        Timeframe identifier ('PRE' or 'POST') for error messages.

    Returns
    -------
    dict
        Dictionary containing:
        - 'merged_df' : pandas DataFrame with all features
        - 'physio_cols' : list of physiology feature column names
        - 'behavior_cols' : list of behavior feature column names
        - 'gaze_cols' : list of gaze feature column names
        - 'metadata' : dict with extraction metadata

    Raises
    ------
    FileNotFoundError
        If the feature file doesn't exist.
    ValueError
        If required keys are missing from the loaded data.

    Examples
    --------
    >>> feature_data = load_features(
    ...     '../../data/results/features_PRE/extracted_features_PRE.pkl',
    ...     timeframe='PRE'
    ... )
    >>> merged_df = feature_data['merged_df']
    >>> print(f"Loaded {len(merged_df)} trials")
    """
    filepath = Path(filepath)

    try:
        with open(filepath, 'rb') as f:
            feature_data = pickle.load(f)

        # Validate expected keys
        required_keys = ['merged_df', 'physio_cols', 'behavior_cols', 'gaze_cols']
        missing = [k for k in required_keys if k not in feature_data]
        if missing:
            raise ValueError(f"Missing required keys in feature file: {missing}")

        return feature_data

    except FileNotFoundError:
        print(f"ERROR: Feature file not found: {filepath}")
        print(f"  → Run notebooks/preprocessing/feature_extraction_{timeframe}.ipynb first")
        raise

    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        raise


def save_results(data, filepath, index=False, verbose=True):
    """
    Save results to CSV with automatic directory creation.

    Parameters
    ----------
    data : pandas.DataFrame or dict
        Data to save. If dict, converts to DataFrame first.
    filepath : str or Path
        Output file path.
    index : bool, default=False
        Whether to save DataFrame index.
    verbose : bool, default=True
        Whether to print confirmation message.

    Examples
    --------
    >>> results_df = pd.DataFrame({'accuracy': [0.68, 0.72], 'model': ['A', 'B']})
    >>> save_results(results_df, '../../data/results/model_comparison.csv')
    ✓ Saved results to: ../../data/results/model_comparison.csv
    """
    filepath = Path(filepath)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    # Save
    data.to_csv(filepath, index=index)

    if verbose:
        print(f"✓ Saved results to: {filepath}")


def load_pickle(filepath, description="data"):
    """
    Load a pickle file with error handling.

    Parameters
    ----------
    filepath : str or Path
        Path to pickle file.
    description : str, default="data"
        Description of what's being loaded (for error messages).

    Returns
    -------
    object
        Unpickled data.

    Examples
    --------
    >>> model = load_pickle('../../data/models/trained_model.pkl', 'trained model')
    """
    filepath = Path(filepath)

    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {description.capitalize()} file not found: {filepath}")
        raise
    except Exception as e:
        print(f"ERROR loading {description} from {filepath}: {e}")
        raise


def save_pickle(data, filepath, description="data", verbose=True):
    """
    Save data to pickle file with error handling.

    Parameters
    ----------
    data : object
        Data to pickle.
    filepath : str or Path
        Output file path.
    description : str, default="data"
        Description of what's being saved (for messages).
    verbose : bool, default=True
        Whether to print confirmation message.

    Examples
    --------
    >>> save_pickle(trained_model, '../../data/models/model.pkl', 'trained model')
    ✓ Saved trained model to: ../../data/models/model.pkl
    """
    filepath = Path(filepath)

    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    if verbose:
        print(f"✓ Saved {description} to: {filepath}")
