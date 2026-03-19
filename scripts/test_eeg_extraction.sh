#!/bin/bash
# Test script to run all EEG feature extraction scripts

echo "=========================================="
echo "Testing EEG Feature Extraction Scripts"
echo "=========================================="
echo ""

# Check if preprocessed EEG file exists
if [ ! -f "data/eeg/Copy of preprocessed_eeg.pkl" ]; then
    echo "ERROR: EEG file not found at data/eeg/Copy of preprocessed_eeg.pkl"
    exit 1
fi

echo "✓ Found EEG data file"
echo ""

# Test 1: Regional features
echo "=========================================="
echo "Test 1: Regional Features (Original)"
echo "=========================================="
python3 scripts/extract_eeg_features.py
if [ $? -eq 0 ]; then
    echo "✓ Regional features extracted successfully"
    echo ""
else
    echo "✗ Regional features extraction FAILED"
    exit 1
fi

# Test 2: Regional + Channels
echo "=========================================="
echo "Test 2: Regional + Individual Channels"
echo "=========================================="
python3 scripts/extract_eeg_features_with_channels.py
if [ $? -eq 0 ]; then
    echo "✓ Regional+Channels features extracted successfully"
    echo ""
else
    echo "✗ Regional+Channels features extraction FAILED"
    exit 1
fi

# Test 3: Non-temporal
echo "=========================================="
echo "Test 3: Non-Temporal Features"
echo "=========================================="
python3 scripts/extract_eeg_features_non_temporal.py
if [ $? -eq 0 ]; then
    echo "✓ Non-temporal features extracted successfully"
    echo ""
else
    echo "✗ Non-temporal features extraction FAILED"
    exit 1
fi

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "All extraction scripts completed successfully!"
echo ""
echo "Generated files:"
echo "  - data/results/eeg_features.pkl (16 features)"
echo "  - data/results/eeg_features_with_channels.pkl (96 features)"
echo "  - data/results/eeg_features_non_temporal.pkl (~96 features)"
echo ""
echo "Next step: Run the comparison notebook"
echo "  jupyter notebook notebooks/fusion_models_eeg/late_fusion_eeg_feature_comparison.ipynb"
