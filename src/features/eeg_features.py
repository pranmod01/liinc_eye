"""
EEG Feature Extraction Module

Unified feature extraction from preprocessed EEG data supporting multiple
extraction strategies with a principled granularity hierarchy.

GRANULARITY HIERARCHY (from finest to coarsest):
- Level 0: channels_raw     → 20 features (total power per electrode)
- Level 1: regional_raw     → 4 features (regional averages of total power)
- Level 2: channels_bands   → 80 features (20 channels × 4 bands)
- Level 3: regional_bands   → 16 features (4 regions × 4 bands)
- Level 4: extended         → 112+ features (channels_bands + lateralization + temporal)

Each level builds on the previous, allowing systematic ablation studies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional, Literal
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Repo root and preprocessed EEG paths (2s epochs at 256 Hz; channels × time)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Pre-decision (display window): stimulus onset to decision (-2s to 0s relative to decision)
PREPROCESSED_EEG_PRE = _REPO_ROOT / "data" / "eeg" / "display_window_2s.pkl"

# Post-decision (review window): decision to feedback (0s to +2s relative to decision)
PREPROCESSED_EEG_POST = _REPO_ROOT / "data" / "eeg" / "review_window_2s.pkl"

# Default to pre-decision for backwards compatibility
PREPROCESSED_EEG_PKL = PREPROCESSED_EEG_PRE

EEG_EPOCH_T0 = -2.0  # seconds relative to decision (start of pre-decision epoch)
EEG_EPOCH_T1 = 0.0   # seconds relative to decision (end of pre-decision epoch)


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

# Strategy hierarchy for reference
STRATEGY_HIERARCHY = {
    'channels_raw': {'level': 0, 'n_features': 20, 'builds_on': None},
    'regional_raw': {'level': 1, 'n_features': 4, 'builds_on': 'channels_raw'},
    'channels_bands': {'level': 2, 'n_features': 80, 'builds_on': 'channels_raw'},
    'regional_bands': {'level': 3, 'n_features': 16, 'builds_on': 'channels_bands'},
    'extended': {'level': 4, 'n_features': 112, 'builds_on': 'channels_bands'},
}


# =============================================================================
# CORE SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def compute_total_power(eeg_data: np.ndarray) -> np.ndarray:
    """
    Compute total (broadband) power for each channel using variance.

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data with shape (n_channels, n_samples)

    Returns
    -------
    np.ndarray
        Total power for each channel (n_channels,)
    """
    return np.var(eeg_data, axis=1)


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
    Unified EEG feature extractor supporting hierarchical extraction strategies.

    GRANULARITY HIERARCHY:
    - Level 0: channels_raw     → 20 features (total power per electrode)
    - Level 1: regional_raw     → 4 features (regional averages of total power)
    - Level 2: channels_bands   → 80 features (20 channels × 4 bands)
    - Level 3: regional_bands   → 16 features (4 regions × 4 bands)
    - Level 4: extended         → 112+ features (channels_bands + lateralization + temporal)

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

    VALID_STRATEGIES = ['channels_raw', 'regional_raw', 'channels_bands', 'regional_bands', 'extended']

    def __init__(
        self,
        strategy: Literal['channels_raw', 'regional_raw', 'channels_bands', 'regional_bands', 'extended'] = 'channels_raw',
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
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self.VALID_STRATEGIES}")

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

        # Level 0: channels_raw (base level)
        if self.strategy == 'channels_raw':
            features.update(self._extract_channels_raw(eeg_data))

        # Level 1: regional_raw (aggregates channels_raw)
        elif self.strategy == 'regional_raw':
            channel_powers = compute_total_power(eeg_data)
            features.update(self._extract_regional_raw(channel_powers))

        # Level 2: channels_bands (adds band decomposition to channels)
        elif self.strategy == 'channels_bands':
            band_powers = compute_band_power(eeg_data, self.fs, self.freq_bands)
            features.update(self._extract_channels_bands(band_powers))

        # Level 3: regional_bands (aggregates channels_bands by region)
        elif self.strategy == 'regional_bands':
            band_powers = compute_band_power(eeg_data, self.fs, self.freq_bands)
            features.update(self._extract_regional_bands(band_powers))

        # Level 4: extended (channels_bands + lateralization + temporal)
        elif self.strategy == 'extended':
            band_powers = compute_band_power(eeg_data, self.fs, self.freq_bands)
            features.update(self._extract_channels_bands(band_powers))
            features.update(self._extract_lateralization_features(band_powers))
            features.update(self._extract_temporal_features(eeg_data))

        return features

    def _extract_channels_raw(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Level 0: Extract total power per channel (20 features).

        This is the base level - raw broadband power at each electrode.
        """
        features = {}
        channel_powers = compute_total_power(eeg_data)

        for ch_idx, ch_name in enumerate(self.channel_names):
            features[f'eeg_{ch_name}'] = channel_powers[ch_idx]

        return features

    def _extract_regional_raw(self, channel_powers: np.ndarray) -> Dict[str, float]:
        """
        Level 1: Extract regional average total power (4 features).

        Aggregates channels_raw by averaging within each region.
        """
        features = {}

        for region, channels in self.channel_regions.items():
            ch_indices = [self.channel_names.index(ch) for ch in channels]
            avg_power = np.mean(channel_powers[ch_indices])
            features[f'eeg_{region}'] = avg_power

        return features

    def _extract_channels_bands(self, band_powers: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Level 2: Extract band power per channel (80 features).

        20 channels × 4 frequency bands.
        """
        features = {}

        for band_name in self.freq_bands.keys():
            for ch_idx, ch_name in enumerate(self.channel_names):
                features[f'eeg_{band_name}_{ch_name}'] = band_powers[band_name][ch_idx]

        return features

    def _extract_regional_bands(self, band_powers: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Level 3: Extract regional average band power (16 features).

        Aggregates channels_bands by averaging within each region.
        4 regions × 4 frequency bands.
        """
        features = {}

        for band_name in self.freq_bands.keys():
            for region, channels in self.channel_regions.items():
                ch_indices = [self.channel_names.index(ch) for ch in channels]
                avg_power = np.mean(band_powers[band_name][ch_indices])
                features[f'eeg_{band_name}_{region}'] = avg_power

        return features

    def _extract_temporal_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal dynamics features using sliding windows (48 features).

        Computes mean, std, and slope of band power over time for each region.
        Part of Level 4 (extended).
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

    def _extract_lateralization_features(self, band_powers: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract left-right asymmetry indices (32 features).

        Lateralization index: (Left - Right) / (Left + Right)
        Positive = left dominance, Negative = right dominance

        Part of Level 4 (extended).
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
    strategy: Literal['channels_raw', 'regional_raw', 'channels_bands', 'regional_bands', 'extended'] = 'channels_raw',
    eeg_column: str = 'eeg',
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
        - eeg_column: EEG data array (channels × time), 2s epochs at `fs` Hz
    strategy : str
        Feature extraction strategy (hierarchical):
        - 'channels_raw': Total power per channel (20 features) - Level 0
        - 'regional_raw': Regional average total power (4 features) - Level 1
        - 'channels_bands': Band power per channel (80 features) - Level 2
        - 'regional_bands': Regional band power (16 features) - Level 3
        - 'extended': Channels + lateralization + temporal (112+ features) - Level 4
    eeg_column : str
        Name of the column containing EEG data arrays (default: 'eeg')
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
        print(f"Extracting EEG features using '{strategy}' strategy (Level {STRATEGY_HIERARCHY[strategy]['level']})...")
        print(f"  EEG column: {eeg_column}")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  Trials: {len(eeg_df)}")

    features_list = []

    for idx, row in eeg_df.iterrows():
        # review_eeg has shape (channels, time) - no transpose needed
        eeg_data = np.array(row[eeg_column])

        # Extract features
        trial_features = extractor.extract_trial_features(
            eeg_data=eeg_data,
            subject_id=row['subject_date_id'],
            trial_id=f"{row['trial_id']}_{row['subject_date_id']}"
        )

        features_list.append(trial_features)

    features_df = pd.DataFrame(features_list)

    if verbose:
        eeg_cols = [c for c in features_df.columns if c.startswith('eeg_')]
        print(f"✓ Extracted {len(eeg_cols)} EEG features")
        _print_feature_summary(strategy, eeg_cols)

    return features_df


def _print_feature_summary(strategy: str, eeg_cols: List[str]) -> None:
    """Print a summary of extracted features based on strategy."""
    if strategy == 'channels_raw':
        print(f"  Channel power: {len(eeg_cols)} features (20 electrodes)")

    elif strategy == 'regional_raw':
        print(f"  Regional power: {len(eeg_cols)} features (4 regions)")

    elif strategy == 'channels_bands':
        print(f"  Channel-band power: {len(eeg_cols)} features (20 channels × 4 bands)")

    elif strategy == 'regional_bands':
        print(f"  Regional-band power: {len(eeg_cols)} features (4 regions × 4 bands)")

    elif strategy == 'extended':
        channel_band_cols = [c for c in eeg_cols if not any(x in c for x in
                            ['_mean', '_std', '_slope', '_lateralization'])]
        temporal_cols = [c for c in eeg_cols if any(x in c for x in
                        ['_mean', '_std', '_slope'])]
        lat_cols = [c for c in eeg_cols if '_lateralization' in c]

        print(f"  Channel-band power: {len(channel_band_cols)} features")
        print(f"  Temporal dynamics: {len(temporal_cols)} features")
        print(f"  Lateralization: {len(lat_cols)} features")


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
        'level': STRATEGY_HIERARCHY[strategy]['level'],
        'builds_on': STRATEGY_HIERARCHY[strategy]['builds_on'],
        'sampling_rate': 256,
        'frequency_bands': FREQ_BANDS,
        'regions': list(CHANNEL_REGIONS.keys()),
        'channels': CHANNEL_NAMES,
        'n_channels': len(CHANNEL_NAMES),
        'n_regions': len(CHANNEL_REGIONS),
    }

    if strategy == 'channels_raw':
        metadata['n_features'] = len(CHANNEL_NAMES)
        metadata['feature_types'] = ['channel_power']

    elif strategy == 'regional_raw':
        metadata['n_features'] = len(CHANNEL_REGIONS)
        metadata['feature_types'] = ['regional_power']

    elif strategy == 'channels_bands':
        metadata['n_features'] = len(FREQ_BANDS) * len(CHANNEL_NAMES)
        metadata['feature_types'] = ['channel_band_power']

    elif strategy == 'regional_bands':
        metadata['n_features'] = len(FREQ_BANDS) * len(CHANNEL_REGIONS)
        metadata['feature_types'] = ['regional_band_power']

    elif strategy == 'extended':
        n_channel_band = len(FREQ_BANDS) * len(CHANNEL_NAMES)  # 80
        n_temporal = len(FREQ_BANDS) * len(CHANNEL_REGIONS) * 3  # 48
        n_lateralization = len(FREQ_BANDS) * len(LATERALIZATION_PAIRS)  # 32
        metadata['n_features'] = n_channel_band + n_temporal + n_lateralization
        metadata['feature_types'] = ['channel_band_power', 'temporal_dynamics', 'lateralization']
        metadata['lateralization_pairs'] = LATERALIZATION_PAIRS

    return metadata


def get_strategy_hierarchy() -> Dict:
    """
    Get the full strategy hierarchy for reference.

    Returns
    -------
    dict
        Strategy hierarchy with level, feature count, and dependencies
    """
    return STRATEGY_HIERARCHY.copy()
