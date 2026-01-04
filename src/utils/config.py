"""
Configuration loading utilities.
"""

import yaml
from pathlib import Path


def load_config(config_name='model_params'):
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_name : str, default='model_params'
        Name of config file (without .yaml extension).

    Returns
    -------
    dict
        Configuration dictionary.

    Examples
    --------
    >>> config = load_config('model_params')
    >>> rf_params = config['random_forest']
    >>> model = RandomForestClassifier(**rf_params)
    """
    # Determine config file path (relative to project root)
    config_path = Path(__file__).parent.parent.parent / 'config' / f'{config_name}.yaml'

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"  → Create config/{config_name}.yaml first"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_model_params(model_name='random_forest', config_name='model_params'):
    """
    Get model parameters from config file.

    Parameters
    ----------
    model_name : str, default='random_forest'
        Name of the model in config file.
    config_name : str, default='model_params'
        Name of config file.

    Returns
    -------
    dict
        Model parameters.

    Examples
    --------
    >>> rf_params = get_model_params('random_forest')
    >>> model = RandomForestClassifier(**rf_params)
    """
    config = load_config(config_name)

    if model_name not in config:
        available = list(config.keys())
        raise KeyError(
            f"Model '{model_name}' not found in config. "
            f"Available models: {available}"
        )

    return config[model_name]


def get_paths(timeframe='PRE', config_name='paths'):
    """
    Get data paths for a specific timeframe.

    Parameters
    ----------
    timeframe : str, default='PRE'
        Either 'PRE' or 'POST'.
    config_name : str, default='paths'
        Name of paths config file.

    Returns
    -------
    dict
        Dictionary of paths for the timeframe.

    Examples
    --------
    >>> paths = get_paths('PRE')
    >>> features_path = paths['features']
    """
    config = load_config(config_name)

    timeframe = timeframe.upper()
    if timeframe not in ['PRE', 'POST']:
        raise ValueError(f"timeframe must be 'PRE' or 'POST', got '{timeframe}'")

    # Get timeframe-specific paths
    timeframe_key = f'{timeframe.lower()}_paths'
    if timeframe_key in config:
        paths = config[timeframe_key].copy()
    else:
        # Fallback: use generic paths and substitute
        paths = {}
        for key, value in config.get('paths', {}).items():
            paths[key] = value.replace('{timeframe}', timeframe)

    return paths
