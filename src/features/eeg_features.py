"""
EEG Feature Extraction Module

Unified feature extraction from preprocessed EEG data supporting multiple
extraction strategies and time windows (PRE/POST decision).

This module follows the same structure as the main preprocessing pipeline
(01_feature_extraction.ipynb) for consistency.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional, Literal
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CHANNEL CONFIGURATION (from chan_locs.sfp)
# =============================================================================

def load_channel_locations(chan_locs_path: str = 'data/eeg/chan_locs.sfp') -> pd.DataFrame:
    """
    Load EEG channel locations from .sfp file.

    Parameters
    ----------
    chan_locs_path : str
        Path to chan_locs.sfp file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['channel', 'x', 'y', 'z']
    """
    chan_locs = pd.read_csv(
        chan_locs_path,
        sep='\t',
        names=['channel', 'x', 'y', 'z'],
        skipinitialspace=True
    )
    chan_locs['channel'] = chan_locs['channel'].str.strip()
    return chan_locs


# Channel names matching chan_locs.sfp (10-20 system, 20 channels)
CHANNEL_NAMES = [
    'Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2', 'O1', 'P3',
    'Pz', 'F3', 'Fz', 'F4', 'C4', 'P4', 'POz', 'C3', 'Cz', 'O2'
]

# Channel regions for grouped analysis
CHANNEL_REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Central': ['T3', 'C3', 'Cz', 'C4', 'T4'],
    'Parietal': ['T5', 'P3', 'Pz', 'P4', 'T6'],
    'Occipital': ['O1', 'POz', 'O2']
}

# Left-right channel pairs for lateralization
LATERALIZATION_PAIRS = {
    'Frontal_lateral': ('F7', 'F8'),
    'Frontal_medial': ('F3', 'F4'),
    'Frontal_pole': ('Fp1', 'Fp2'),
    'Central': ('C3', 'C4'),
    'Temporal_anterior': ('T3', 'T4'),
    'Temporal_posterior': ('T5', 'T6'),
    'Parietal': ('P3', 'P4'),
    'Occipital': ('O1', 'O2'),
}

# Frequency bands (Gamma excluded due to noise)
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
}


# =============================================================================
# CORE SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def compute_psd(eeg_data: np.ndarray, fs: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density for all channels using Welch's method.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data with shape (n_channels, n_samples)
    fs : int
        Sampling frequency in Hz

    Returns
    -------
    freqs : np.ndarray
        Frequency bins
    psd : np.ndarray
        Power spectral density with shape (n_channels, n_freqs)
    """
    freqs, psd = signal.welch(
        eeg_data,
        fs=fs,
        nperseg=min(256, eeg_data.shape[1])
    )
    return freqs, psd


def compute_band_power(
    eeg_data: np.ndarray,
    fs: int = 256,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute band power for each channel and frequency band.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data with shape (n_channels, n_samples)
    fs : int
        Sampling frequency in Hz
    bands : dict, optional
        Dictionary of frequency bands {name: (low_freq, high_freq)}
        If None, uses default FREQ_BANDS

    Returns
    -------
    dict
        Dictionary mapping band name to band power array (n_channels,)
    """
    if bands is None:
        bands = FREQ_BANDS

    freqs, psd = compute_psd(eeg_data, fs)

    band_powers = {}
    for band_name, (f_low, f_high) in bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        # Integrate PSD over frequency band using trapezoidal rule
        # Use trapezoid (numpy>=2.0) or trapz (numpy<2.0) for compatibility
        try:
            band_power = np.trapezoid(psd[:, freq_mask], freqs[freq_mask], axis=1)
        except AttributeError:
            band_power = np.trapz(psd[:, freq_mask], freqs[freq_mask], axis=1)
        band_powers[band_name] = band_power

    return band_powers


# =============================================================================
# FEATURE EXTRACTION STRATEGIES
# =============================================================================

class EEGFeatureExtractor:
    """
    Unified EEG feature extractor supporting multiple extraction strategies.

    Strategies:
    - 'regional': Regional averages only (16 features)
    - 'regional_with_channels': Regional + individual channels (96 features)
    - 'non_temporal': Regional + temporal dynamics + lateralization (96 features)

    Parameters
    ----------
    strategy : str
        Feature extraction strategy
    fs : int
        Sampling frequency in Hz
    channel_names : list, optional
        List of channel names (defaults to CHANNEL_NAMES)
    channel_regions : dict, optional
        Dictionary mapping regions to channel lists
    freq_bands : dict, optional
        Dictionary of frequency bands
    """

    def __init__(
        self,
        strategy: Literal['regional', 'regional_with_channels', 'non_temporal'] = 'regional',
        fs: int = 256,
        channel_names: Optional[List[str]] = None,
        channel_regions: Optional[Dict[str, List[str]]] = None,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.strategy = strategy
        self.fs = fs
        self.channel_names = channel_names or CHANNEL_NAMES
        self.channel_regions = channel_regions or CHANNEL_REGIONS
        self.freq_bands = freq_bands or FREQ_BANDS

        # Validate strategy
        valid_strategies = ['regional', 'regional_with_channels', 'non_temporal']
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")

    def extract_trial_features(
        self,
        eeg_data: np.ndarray,
        subject_id: str,
        trial_id: str
    ) -> Dict[str, float]:
        """
        Extract EEG features for a single trial.

        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data with shape (n_channels, n_samples)
        subject_id : str
            Subject identifier
        trial_id : str
            Trial identifier

        Returns
        -------
        dict
            Dictionary of features
        """
        features = {
            'subject_id': subject_id,
            'trial_id': trial_id
        }

        # Compute band powers
        band_powers = compute_band_power(eeg_data, self.fs, self.freq_bands)

        if self.strategy == 'regional':
            features.update(self._extract_regional_features(band_powers))

        elif self.strategy == 'regional_with_channels':
            features.update(self._extract_regional_features(band_powers))
            features.update(self._extract_channel_features(band_powers))

        elif self.strategy == 'non_temporal':
            features.update(self._extract_regional_features(band_powers))
            features.update(self._extract_temporal_features(eeg_data))
            features.update(self._extract_lateralization_features(band_powers))

        return features

    def _extract_regional_features(
        self,
        band_powers: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Extract regional average band power features (16 features)."""
        features = {}

        for band_name in self.freq_bands.keys():
            for region, channels in self.channel_regions.items():
                ch_indices = [self.channel_names.index(ch) for ch in channels]
                avg_power = np.mean(band_powers[band_name][ch_indices])
                features[f'eeg_{band_name}_{region}'] = avg_power

        return features

    def _extract_channel_features(
        self,
        band_powers: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Extract individual channel band power features (80 features)."""
        features = {}

        for band_name in self.freq_bands.keys():
            for ch_idx, ch_name in enumerate(self.channel_names):
                features[f'eeg_{band_name}_{ch_name}'] = band_powers[band_name][ch_idx]

        return features

    def _extract_temporal_features(
        self,
        eeg_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract temporal dynamics features using sliding windows (48 features).

        Computes mean, std, and slope of band power over time for each region.
        """
        features = {}

        # Sliding window parameters
        window_size = int(0.5 * self.fs)  # 500ms windows
        step_size = int(0.25 * self.fs)   # 250ms steps
        n_channels, n_samples = eeg_data.shape

        for band_name, (f_low, f_high) in self.freq_bands.items():
            # Compute band power in sliding windows
            band_powers_over_time = []

            for start in range(0, n_samples - window_size, step_size):
                window_data = eeg_data[:, start:start+window_size]
                bp = compute_band_power(
                    window_data,
                    self.fs,
                    {band_name: (f_low, f_high)}
                )
                band_powers_over_time.append(bp[band_name])

            if len(band_powers_over_time) == 0:
                continue

            band_powers_over_time = np.array(band_powers_over_time)  # (n_windows, n_channels)

            # Compute statistics across time for each region
            for region, channels in self.channel_regions.items():
                ch_indices = [self.channel_names.index(ch) for ch in channels]
                region_power = band_powers_over_time[:, ch_indices].mean(axis=1)

                # Temporal statistics
                features[f'eeg_{band_name}_{region}_mean'] = np.mean(region_power)
                features[f'eeg_{band_name}_{region}_std'] = np.std(region_power)

                # Temporal slope (linear trend)
                if len(region_power) > 1:
                    slope = stats.linregress(
                        np.arange(len(region_power)),
                        region_power
                    ).slope
                    features[f'eeg_{band_name}_{region}_slope'] = slope
                else:
                    features[f'eeg_{band_name}_{region}_slope'] = 0.0

        return features

    def _extract_lateralization_features(
        self,
        band_powers: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Extract left-right asymmetry indices (32 features).

        Lateralization index: (Left - Right) / (Left + Right)
        Positive = left dominance, Negative = right dominance
        """
        features = {}

        for band_name in self.freq_bands.keys():
            for pair_name, (left_ch, right_ch) in LATERALIZATION_PAIRS.items():
                left_idx = self.channel_names.index(left_ch)
                right_idx = self.channel_names.index(right_ch)

                left_power = band_powers[band_name][left_idx]
                right_power = band_powers[band_name][right_idx]

                # Asymmetry index with small epsilon to avoid division by zero
                lat_index = (left_power - right_power) / (left_power + right_power + 1e-10)
                features[f'eeg_{band_name}_{pair_name}_lateralization'] = lat_index

        return features


# =============================================================================
# HIGH-LEVEL EXTRACTION FUNCTION
# =============================================================================

def extract_eeg_features(
    eeg_df: pd.DataFrame,
    strategy: Literal['regional', 'regional_with_channels', 'non_temporal'] = 'regional',
    fs: int = 256,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract EEG features from preprocessed EEG DataFrame.

    Parameters
    ----------
    eeg_df : pd.DataFrame
        DataFrame with columns:
        - 'subject_date_id': Subject identifier
        - 'trial_id': Trial number
        - 'display_eeg': EEG data array (time x channels)
    strategy : str
        Feature extraction strategy:
        - 'regional': Regional averages (16 features)
        - 'regional_with_channels': Regional + channels (96 features)
        - 'non_temporal': Regional + temporal + lateralization (96 features)
    fs : int
        Sampling frequency in Hz
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted features
    """
    extractor = EEGFeatureExtractor(strategy=strategy, fs=fs)

    if verbose:
        print(f"Extracting EEG features using '{strategy}' strategy...")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  Trials: {len(eeg_df)}")

    features_list = []

    for idx, row in eeg_df.iterrows():
        # display_eeg has shape (time, channels)
        # Transpose to (channels, time) for processing
        display_eeg = np.array(row['display_eeg']).T

        # Extract features
        trial_features = extractor.extract_trial_features(
            eeg_data=display_eeg,
            subject_id=row['subject_date_id'],
            trial_id=f"{row['trial_id']}_{row['subject_date_id']}"
        )

        features_list.append(trial_features)

    features_df = pd.DataFrame(features_list)

    if verbose:
        eeg_cols = [c for c in features_df.columns if c.startswith('eeg_')]
        print(f"✓ Extracted {len(eeg_cols)} EEG features")

        # Categorize features by type
        if strategy == 'regional':
            print(f"  Regional power: {len(eeg_cols)} features")
        elif strategy == 'regional_with_channels':
            regional_cols = [c for c in eeg_cols if any(r in c for r in CHANNEL_REGIONS.keys())]
            channel_cols = [c for c in eeg_cols if c not in regional_cols]
            print(f"  Regional power: {len(regional_cols)} features")
            print(f"  Individual channels: {len(channel_cols)} features")
        elif strategy == 'non_temporal':
            power_cols = [c for c in eeg_cols if '_power' in c or
                         (not any(x in c for x in ['_mean', '_std', '_slope', '_lateralization']))]
            temporal_cols = [c for c in eeg_cols if any(x in c for x in ['_mean', '_std', '_slope'])]
            lat_cols = [c for c in eeg_cols if '_lateralization' in c]
            print(f"  Regional power: {len(power_cols)} features")
            print(f"  Temporal dynamics: {len(temporal_cols)} features")
            print(f"  Lateralization: {len(lat_cols)} features")

    return features_df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_metadata(strategy: str) -> Dict:
    """
    Get metadata about features for a given strategy.

    Parameters
    ----------
    strategy : str
        Feature extraction strategy

    Returns
    -------
    dict
        Metadata dictionary
    """
    metadata = {
        'strategy': strategy,
        'sampling_rate': 256,
        'frequency_bands': FREQ_BANDS,
        'regions': list(CHANNEL_REGIONS.keys()),
        'channels': CHANNEL_NAMES,
        'n_channels': len(CHANNEL_NAMES),
        'n_regions': len(CHANNEL_REGIONS),
    }

    if strategy == 'regional':
        metadata['n_features'] = len(FREQ_BANDS) * len(CHANNEL_REGIONS)
        metadata['feature_types'] = ['regional_power']
    elif strategy == 'regional_with_channels':
        metadata['n_features'] = (len(FREQ_BANDS) * len(CHANNEL_REGIONS) +
                                 len(FREQ_BANDS) * len(CHANNEL_NAMES))
        metadata['feature_types'] = ['regional_power', 'channel_power']
    elif strategy == 'non_temporal':
        metadata['n_features'] = (len(FREQ_BANDS) * len(CHANNEL_REGIONS) +     # Regional power
                                 len(FREQ_BANDS) * len(CHANNEL_REGIONS) * 3 +  # Temporal (mean, std, slope)
                                 len(FREQ_BANDS) * len(LATERALIZATION_PAIRS))  # Lateralization
        metadata['feature_types'] = ['regional_power', 'temporal_dynamics', 'lateralization']
        metadata['lateralization_pairs'] = LATERALIZATION_PAIRS

    return metadata
