#!/usr/bin/env python3
"""
Extract EEG features with BOTH regional averages AND individual channels.
This script creates eeg_features_with_channels.pkl for comparison with regional-only approach.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

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

def extract_eeg_features_all(eeg_df, fs=256, include_channels=True):
    """
    Extract band power features for all trials.

    Parameters
    ----------
    eeg_df : pd.DataFrame
        DataFrame with display_eeg data (shape: time x channels)
    fs : int
        Sampling frequency
    include_channels : bool
        If True, include individual channel features in addition to regional averages

    Returns
    -------
    pd.DataFrame
        Features with both regional averages and individual channels
    """
    features_list = []

    for idx, row in eeg_df.iterrows():
        # display_eeg has shape (320, 20) = (time, channels)
        # Transpose to (20, 320) = (channels, time) for band power computation
        display_eeg = np.array(row['display_eeg']).T

        # Compute band powers
        bp = compute_band_power(display_eeg, fs)

        # Create composite trial_id to match extracted_features.pkl format
        trial_features = {
            'subject_id': row['subject_date_id'],
            'trial_id': f"{row['trial_id']}_{row['subject_date_id']}"
        }

        for band_name in freq_bands.keys():
            band_power = bp[band_name]

            # 1. Regional averages (16 features)
            for region, channels in channel_regions.items():
                ch_indices = [channel_names.index(ch) for ch in channels]
                avg_power = np.mean(band_power[ch_indices])
                trial_features[f'eeg_{band_name}_{region}'] = avg_power

            # 2. Individual channels (80 features)
            if include_channels:
                for ch_idx, ch_name in enumerate(channel_names):
                    trial_features[f'eeg_{band_name}_{ch_name}'] = band_power[ch_idx]

        features_list.append(trial_features)

    return pd.DataFrame(features_list)

def main():
    # Load EEG data
    print("Loading EEG data...")
    with open('data/eeg/Copy of preprocessed_eeg.pkl', 'rb') as f:
        eeg_df = pickle.load(f)

    print(f"✓ Loaded {len(eeg_df)} trials")
    print(f"  Unique subjects: {eeg_df['subject_date_id'].nunique()}")

    # Extract features WITH individual channels
    print("\nExtracting EEG features (regional + individual channels)...")
    eeg_features_df = extract_eeg_features_all(eeg_df, fs=256, include_channels=True)

    eeg_cols = [c for c in eeg_features_df.columns if c.startswith('eeg_')]
    regional_cols = [c for c in eeg_cols if any(region in c for region in channel_regions.keys())]
    channel_cols = [c for c in eeg_cols if c not in regional_cols]

    print(f"\n✓ Extracted {len(eeg_cols)} total EEG features:")
    print(f"  - {len(regional_cols)} regional features (4 bands × 4 regions)")
    print(f"  - {len(channel_cols)} individual channel features (4 bands × 20 channels)")

    # Show example features
    print(f"\n  Regional features: {regional_cols[:3]}...")
    print(f"  Channel features: {channel_cols[:3]}...")

    # Save features
    output_path = 'data/results/eeg_features_with_channels.pkl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'eeg_features_df': eeg_features_df,
            'feature_columns': eeg_cols,
            'regional_columns': regional_cols,
            'channel_columns': channel_cols,
            'metadata': {
                'n_trials': len(eeg_features_df),
                'n_features': len(eeg_cols),
                'n_regional_features': len(regional_cols),
                'n_channel_features': len(channel_cols),
                'frequency_bands': freq_bands,
                'regions': list(channel_regions.keys()),
                'channels': channel_names,
                'sampling_rate': 256,
                'description': 'EEG features with both regional averages and individual channels'
            }
        }, f)

    print(f"\n✓ Saved EEG features to {output_path}")
    print(f"  {len(eeg_features_df)} trials")
    print(f"  {len(eeg_cols)} features ({len(regional_cols)} regional + {len(channel_cols)} channels)")

if __name__ == '__main__':
    main()
