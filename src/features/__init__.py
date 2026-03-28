"""Feature extraction modules for multimodal data."""

from .eeg_features import (
    load_channel_locations,
    EEGFeatureExtractor,
    extract_eeg_features
)

__all__ = [
    'load_channel_locations',
    'EEGFeatureExtractor',
    'extract_eeg_features',
]
