#!/usr/bin/env python3
"""
Extract NON-TEMPORAL EEG features from preprocessed EEG data.
This extracts features from the ENTIRE display epoch without time-windowing,
similar to the non-temporal pupil/gaze approach.

Features extracted:
- Band power statistics (mean, std, min, max) across the full epoch
- Lateralization indices (left-right asymmetry)
- Inter-regional connectivity (phase-locking, coherence)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats

# Define frequency bands (Gamma excluded due to noise)
freq_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
}

# Channel names matching chan_locs.sfp (10-20 system, 20 channels)
channel_names = [
    'Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2', 'O1', 'P3',
    'Pz', 'F3', 'Fz', 'F4', 'C4', 'P4', 'POz', 'C3', 'Cz', 'O2'
]

# Channel regions for grouped analysis
channel_regions = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Central': ['T3', 'C3', 'Cz', 'C4', 'T4'],
    'Parietal': ['T5', 'P3', 'Pz', 'P4', 'T6'],
    'Occipital': ['O1', 'POz', 'O2']
}

# Left-right channel pairs for lateralization
lateralization_pairs = {
    'Frontal_lateral': ('F7', 'F8'),
    'Frontal_medial': ('F3', 'F4'),
    'Frontal_pole': ('Fp1', 'Fp2'),
    'Central': ('C3', 'C4'),
    'Temporal_anterior': ('T3', 'T4'),
    'Temporal_posterior': ('T5', 'T6'),
    'Parietal': ('P3', 'P4'),
    'Occipital': ('O1', 'O2'),
}

def compute_psd(eeg_data, fs=256):
    """Compute Power Spectral Density for all channels."""
    freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=min(256, eeg_data.shape[1]))
    return freqs, psd

def compute_band_power(eeg_data, fs=256, bands=None):
    """Compute band power for each channel and frequency band."""
    if bands is None:
        bands = freq_bands

    freqs, psd = compute_psd(eeg_data, fs)

    band_powers = {}
    for band_name, (f_low, f_high) in bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        band_power = np.trapz(psd[:, freq_mask], freqs[freq_mask], axis=1)
        band_powers[band_name] = band_power

    return band_powers

def compute_temporal_features(eeg_data, fs=256, bands=None):
    """
    Compute temporal dynamics of band power over the entire epoch.

    Returns statistics (mean, std, skew, kurt) for each band/region.
    """
    if bands is None:
        bands = freq_bands

    # Compute band power in sliding windows
    window_size = int(0.5 * fs)  # 500ms windows
    step_size = int(0.25 * fs)   # 250ms steps
    n_channels, n_samples = eeg_data.shape

    temporal_features = {}

    for band_name, (f_low, f_high) in bands.items():
        # Sliding window band power
        band_powers_over_time = []

        for start in range(0, n_samples - window_size, step_size):
            window_data = eeg_data[:, start:start+window_size]
            bp = compute_band_power(window_data, fs, {band_name: (f_low, f_high)})
            band_powers_over_time.append(bp[band_name])

        band_powers_over_time = np.array(band_powers_over_time)  # (n_windows, n_channels)

        # Compute statistics across time for each region
        for region, channels in channel_regions.items():
            ch_indices = [channel_names.index(ch) for ch in channels]
            region_power = band_powers_over_time[:, ch_indices].mean(axis=1)  # Average across channels

            # Temporal statistics
            temporal_features[f'eeg_{band_name}_{region}_mean'] = np.mean(region_power)
            temporal_features[f'eeg_{band_name}_{region}_std'] = np.std(region_power)
            temporal_features[f'eeg_{band_name}_{region}_slope'] = stats.linregress(
                np.arange(len(region_power)), region_power
            ).slope

    return temporal_features

def compute_lateralization(bp, band_name):
    """
    Compute left-right asymmetry indices.

    Lateralization index: (Left - Right) / (Left + Right)
    Positive = left dominance, Negative = right dominance
    """
    lat_features = {}

    for pair_name, (left_ch, right_ch) in lateralization_pairs.items():
        left_idx = channel_names.index(left_ch)
        right_idx = channel_names.index(right_ch)

        left_power = bp[band_name][left_idx]
        right_power = bp[band_name][right_idx]

        # Asymmetry index
        lat_index = (left_power - right_power) / (left_power + right_power + 1e-10)
        lat_features[f'eeg_{band_name}_{pair_name}_lateralization'] = lat_index

    return lat_features

def extract_eeg_features_non_temporal(eeg_df, fs=256):
    """
    Extract non-temporal EEG features from the ENTIRE display epoch.

    Features include:
    - Regional band power (mean across full epoch)
    - Temporal dynamics (std, slope)
    - Lateralization indices
    """
    features_list = []

    for idx, row in eeg_df.iterrows():
        # display_eeg has shape (320, 20) = (time, channels)
        # Transpose to (20, 320) = (channels, time)
        display_eeg = np.array(row['display_eeg']).T

        # Compute overall band powers (full epoch)
        bp = compute_band_power(display_eeg, fs)

        # Initialize trial features
        trial_features = {
            'subject_id': row['subject_date_id'],
            'trial_id': f"{row['trial_id']}_{row['subject_date_id']}"
        }

        # 1. Regional band power (mean across full epoch) - 16 features
        for band_name in freq_bands.keys():
            for region, channels in channel_regions.items():
                ch_indices = [channel_names.index(ch) for ch in channels]
                avg_power = np.mean(bp[band_name][ch_indices])
                trial_features[f'eeg_{band_name}_{region}_power'] = avg_power

        # 2. Temporal dynamics - 48 features (4 bands × 4 regions × 3 stats)
        temporal_feats = compute_temporal_features(display_eeg, fs)
        trial_features.update(temporal_feats)

        # 3. Lateralization indices - 32 features (4 bands × 8 pairs)
        for band_name in freq_bands.keys():
            lat_feats = compute_lateralization(bp, band_name)
            trial_features.update(lat_feats)

        features_list.append(trial_features)

    return pd.DataFrame(features_list)

def main():
    # Load EEG data
    print("Loading EEG data...")
    with open('data/eeg/Copy of preprocessed_eeg.pkl', 'rb') as f:
        eeg_df = pickle.load(f)

    print(f"✓ Loaded {len(eeg_df)} trials")
    print(f"  Unique subjects: {eeg_df['subject_date_id'].nunique()}")

    # Extract non-temporal features
    print("\nExtracting NON-TEMPORAL EEG features...")
    print("  - Regional band power (mean across full epoch)")
    print("  - Temporal dynamics (std, slope)")
    print("  - Lateralization indices (left-right asymmetry)")

    eeg_features_df = extract_eeg_features_non_temporal(eeg_df, fs=256)

    eeg_cols = [c for c in eeg_features_df.columns if c.startswith('eeg_')]

    # Categorize feature types
    power_cols = [c for c in eeg_cols if '_power' in c]
    temporal_cols = [c for c in eeg_cols if any(x in c for x in ['_mean', '_std', '_slope'])]
    lat_cols = [c for c in eeg_cols if '_lateralization' in c]

    print(f"\n✓ Extracted {len(eeg_cols)} total EEG features:")
    print(f"  - {len(power_cols)} regional power features")
    print(f"  - {len(temporal_cols)} temporal dynamics features")
    print(f"  - {len(lat_cols)} lateralization features")

    # Show examples
    print(f"\n  Power features: {power_cols[:3]}...")
    print(f"  Temporal features: {temporal_cols[:3]}...")
    print(f"  Lateralization features: {lat_cols[:3]}...")

    # Save features
    output_path = 'data/results/eeg_features_non_temporal.pkl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'eeg_features_df': eeg_features_df,
            'feature_columns': eeg_cols,
            'power_columns': power_cols,
            'temporal_columns': temporal_cols,
            'lateralization_columns': lat_cols,
            'metadata': {
                'n_trials': len(eeg_features_df),
                'n_features': len(eeg_cols),
                'n_power_features': len(power_cols),
                'n_temporal_features': len(temporal_cols),
                'n_lateralization_features': len(lat_cols),
                'frequency_bands': freq_bands,
                'regions': list(channel_regions.keys()),
                'lateralization_pairs': lateralization_pairs,
                'sampling_rate': 256,
                'description': 'Non-temporal EEG features from full display epoch'
            }
        }, f)

    print(f"\n✓ Saved NON-TEMPORAL EEG features to {output_path}")
    print(f"  {len(eeg_features_df)} trials")
    print(f"  {len(eeg_cols)} features")

if __name__ == '__main__':
    main()
