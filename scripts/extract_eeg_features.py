#!/usr/bin/env python3
"""
Extract EEG features from preprocessed EEG data.
This script creates the eeg_features.pkl file needed for late fusion analysis.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

# Define frequency bands
freq_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# Channel names (10-20 system, 20 channels)
channel_names = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'Oz', 'O2'
]

# Channel regions
channel_regions = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Central': ['T3', 'C3', 'Cz', 'C4', 'T4'],
    'Parietal': ['T5', 'P3', 'Pz', 'P4', 'T6'],
    'Occipital': ['O1', 'Oz', 'O2']
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

def extract_eeg_features_all(eeg_df, fs=256):
    """Extract band power features for all trials."""
    features_list = []

    for idx, row in eeg_df.iterrows():
        review_eeg = np.array(row['review_eeg'])
        baseline_eeg = np.array(row['baseline_eeg'])

        # Compute band powers
        bp_review = compute_band_power(review_eeg, fs)
        bp_baseline = compute_band_power(baseline_eeg, fs)

        # Baseline-correct
        # Create composite trial_id to match extracted_features.pkl format: "trialnum_subject_id"
        trial_features = {
            'subject_id': row['subject_date_id'],  # Rename to match extracted_features.pkl
            'trial_id': f"{row['trial_id']}_{row['subject_date_id']}"  # Composite key
        }

        for band_name in freq_bands.keys():
            bp_corr = (bp_review[band_name] - bp_baseline[band_name]) / (bp_baseline[band_name] + 1e-10)

            # Average by region
            for region, channels in channel_regions.items():
                ch_indices = [channel_names.index(ch) for ch in channels]
                avg_power = np.mean(bp_corr[ch_indices])
                trial_features[f'eeg_{band_name}_{region}'] = avg_power

        features_list.append(trial_features)

    return pd.DataFrame(features_list)

def main():
    # Load EEG data
    print("Loading EEG data...")
    with open('data/raw/preprocessed_eeg_10_subjects.pkl', 'rb') as f:
        eeg_df = pickle.load(f)

    print(f"✓ Loaded {len(eeg_df)} trials")
    print(f"  Unique subjects: {eeg_df['subject_date_id'].nunique()}")

    # Extract features
    print("\nExtracting EEG features...")
    eeg_features_df = extract_eeg_features_all(eeg_df, fs=256)

    eeg_cols = [c for c in eeg_features_df.columns if c.startswith('eeg_')]

    print(f"✓ Extracted {len(eeg_cols)} EEG features")
    print(f"  Features: {eeg_cols[:5]}... (showing first 5)")

    # Save features
    output_path = 'data/results/eeg_features.pkl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'eeg_features_df': eeg_features_df,
            'feature_columns': eeg_cols,
            'metadata': {
                'n_trials': len(eeg_features_df),
                'n_features': len(eeg_cols),
                'frequency_bands': freq_bands,
                'regions': list(channel_regions.keys()),
                'sampling_rate': 256,
                'channel_names': channel_names
            }
        }, f)

    print(f"\n✓ Saved EEG features to {output_path}")
    print(f"  {len(eeg_features_df)} trials")
    print(f"  {len(eeg_cols)} features")
    print(f"\nColumns: {eeg_features_df.columns.tolist()}")

if __name__ == '__main__':
    main()
