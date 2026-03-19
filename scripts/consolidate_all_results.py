"""
Consolidate all analysis results into a single comprehensive CSV file.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

# Base directory
BASE_DIR = Path('/Users/pranmodu/Projects/columbia/liinc')
RESULTS_DIR = BASE_DIR / 'data' / 'results'

# Initialize results list
all_results = []

def extract_top_features(feature_file, n=10):
    """Extract top N features from a feature importance file."""
    if not os.path.exists(feature_file):
        return None

    df = pd.read_csv(feature_file)

    # Try different column names for feature importance
    if 'Feature' in df.columns and 'Importance' in df.columns:
        top_features = df.nlargest(n, 'Importance')['Feature'].tolist()
    elif 'feature' in df.columns and 'importance' in df.columns:
        top_features = df.nlargest(n, 'importance')['feature'].tolist()
    else:
        # Just take first n features
        top_features = df.iloc[:n, 0].tolist() if len(df) >= n else df.iloc[:, 0].tolist()

    return '; '.join(top_features)


def get_modality_weights(weights_dict, modality_order=None):
    """Format modality weights as a string."""
    if not weights_dict:
        return None

    if modality_order:
        return '; '.join([f"{mod}: {weights_dict.get(mod, 'N/A'):.4f}" for mod in modality_order])
    else:
        return '; '.join([f"{k}: {v:.4f}" for k, v in weights_dict.items()])


def process_overall_analysis(analysis_type, feature_type, has_eeg=False):
    """Process overall (non-grouped) analysis results."""

    # Determine result directory
    if feature_type == 'temporal' and not has_eeg:
        result_dir = RESULTS_DIR / 'fusion_model_results_PRE'
        comparison_file = result_dir / 'late_fusion_model_PRE_method_comparison.csv'
        weights_file = result_dir / 'late_fusion_model_PRE_modality_weights.csv'
        summary_file = result_dir / 'late_fusion_model_PRE_weighted_fusion_summary.csv'
        notebook = 'notebooks/fusion_models/late_fusion_model.ipynb'
    elif feature_type == 'temporal' and has_eeg:
        result_dir = RESULTS_DIR / 'fusion_models_eeg_PRE'
        comparison_file = result_dir / 'late_fusion_with_eeg_PRE_method_comparison.csv'
        weights_file = result_dir / 'late_fusion_with_eeg_PRE_modality_weights.csv'
        summary_file = result_dir / 'late_fusion_with_eeg_PRE_weighted_fusion_summary.csv'
        feature_file = result_dir / 'late_fusion_with_eeg_PRE_eeg_feature_importances.csv'
        notebook = 'notebooks/fusion_models_eeg/late_fusion_with_eeg.ipynb'
    elif feature_type == 'non_temporal' and not has_eeg:
        result_dir = RESULTS_DIR / 'fusion_model_results_no_temporal_PRE'
        comparison_file = result_dir / 'late_fusion_model_no_temporal_PRE_method_comparison.csv'
        weights_file = result_dir / 'late_fusion_model_no_temporal_PRE_modality_weights.csv'
        summary_file = result_dir / 'late_fusion_model_no_temporal_PRE_weighted_fusion_summary.csv'
        notebook = 'notebooks/fusion_models_no_temporal/late_fusion_model_no_temporal.ipynb'
    elif feature_type == 'non_temporal' and has_eeg:
        result_dir = RESULTS_DIR / 'fusion_model_results_no_temporal_eeg_PRE'
        comparison_file = result_dir / 'late_fusion_model_no_temporal_eeg_PRE_method_comparison.csv'
        weights_file = result_dir / 'late_fusion_model_no_temporal_eeg_PRE_modality_weights.csv'
        summary_file = result_dir / 'late_fusion_model_no_temporal_eeg_PRE_weighted_fusion_summary.csv'
        notebook = 'notebooks/fusion_models_no_temporal/late_fusion_model_no_temporal_eeg.ipynb'
    else:
        return []

    results = []

    if not comparison_file.exists():
        return results

    # Read comparison file
    comparison_df = pd.read_csv(comparison_file)

    # Read weights if available
    weights_dict = {}
    if weights_file.exists():
        weights_df = pd.read_csv(weights_file)
        if 'Weighted' in weights_df.columns:
            weights_dict = dict(zip(weights_df['Modality'], weights_df['Weighted']))

    # Read summary if available
    n_subjects = None
    n_trials = None
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        if 'N_Subjects' in summary_df.columns:
            n_subjects = summary_df['N_Subjects'].iloc[0]
        if 'N_Trials' in summary_df.columns:
            n_trials = summary_df['N_Trials'].iloc[0]

    # Get top features if EEG
    top_features = None
    if has_eeg and feature_type == 'temporal':
        feature_file = result_dir / 'late_fusion_with_eeg_PRE_eeg_feature_importances.csv'
        if feature_file.exists():
            top_features = extract_top_features(feature_file)

    # Process weighted fusion row
    weighted_row = comparison_df[comparison_df['Method'] == 'Weighted Fusion']
    if len(weighted_row) > 0:
        weighted_row = weighted_row.iloc[0]

        result = {
            'Analysis_Type': analysis_type,
            'Feature_Type': feature_type,
            'Has_EEG': 'Yes' if has_eeg else 'No',
            'Grouping_Variable': 'None',
            'Group_Name': 'Overall',
            'Accuracy': weighted_row.get('Accuracy', np.nan),
            'Accuracy_SEM': np.nan,
            'Accuracy_SD': np.nan,
            'F1_Score': weighted_row.get('F1-Score', np.nan),
            'F1_SEM': np.nan,
            'Modality_Weights': get_modality_weights(weights_dict),
            'N_Subjects': n_subjects,
            'N_Trials': n_trials,
            'Top_10_Features': top_features,
            'Notebook': notebook,
            'Notes': 'Overall late fusion analysis'
        }
        results.append(result)

    return results


def process_grouped_analysis(analysis_type, grouping_var, feature_type, has_eeg=False):
    """Process grouped analysis results (RT or Ambiguity)."""

    # Determine result directory and file patterns
    if feature_type == 'temporal' and not has_eeg:
        result_dir = RESULTS_DIR / 'fusion_model_results_PRE'
        if grouping_var == 'Reaction_Time':
            comparison_file = result_dir / 'reaction_time_group_late_fusion_PRE_comparison.csv'
            weights_pattern = 'reaction_time_group_late_fusion_PRE_{}_weights.csv'
            subject_acc_pattern = 'reaction_time_group_late_fusion_PRE_{}_subject_accuracies.csv'
            shap_pattern = None
            notebook = 'notebooks/fusion_models/reaction_time_group_late_fusion.ipynb'
        else:  # Ambiguity
            comparison_file = result_dir / 'ambiguity_group_late_fusion_PRE_comparison.csv'
            weights_pattern = 'ambiguity_group_late_fusion_PRE_{}_weights.csv'
            subject_acc_pattern = 'ambiguity_group_late_fusion_PRE_{}_subject_accuracies.csv'
            shap_pattern = None
            notebook = 'notebooks/fusion_models/ambiguity_group_late_fusion.ipynb'

    elif feature_type == 'temporal' and has_eeg:
        result_dir = RESULTS_DIR / 'fusion_models_eeg_PRE'
        if grouping_var == 'Reaction_Time':
            comparison_file = result_dir / 'rt_group_eeg_PRE_comparison.csv'
            weights_pattern = 'rt_group_eeg_PRE_{}_weights.csv'
            subject_acc_pattern = 'rt_group_eeg_PRE_{}_subject_accuracies.csv'
            shap_pattern = None
            notebook = 'notebooks/fusion_models_eeg/reaction_time_group_late_fusion_eeg.ipynb'
        else:  # Ambiguity
            comparison_file = result_dir / 'ambiguity_group_eeg_PRE_comparison.csv'
            weights_pattern = 'ambiguity_group_eeg_PRE_{}_weights.csv'
            subject_acc_pattern = 'ambiguity_group_eeg_PRE_{}_subject_accuracies.csv'
            shap_pattern = None
            notebook = 'notebooks/fusion_models_eeg/ambiguity_group_late_fusion_eeg.ipynb'

    elif feature_type == 'non_temporal' and not has_eeg:
        result_dir = RESULTS_DIR / 'fusion_model_results_no_temporal_PRE'
        if grouping_var == 'Reaction_Time':
            comparison_file = result_dir / 'reaction_time_group_no_temporal_PRE_comparison.csv'
            weights_pattern = 'reaction_time_group_no_temporal_PRE_{}_weights.csv'
            subject_acc_pattern = 'reaction_time_group_no_temporal_PRE_{}_subject_accuracies.csv'
            shap_pattern = 'shap_importance_{}_rt_PRE.csv'
            notebook = 'notebooks/fusion_models_no_temporal/reaction_time_group_late_fusion_no_temporal.ipynb'
        else:  # Ambiguity
            comparison_file = result_dir / 'ambiguity_group_no_temporal_PRE_comparison.csv'
            weights_pattern = 'ambiguity_group_no_temporal_PRE_{}_weights.csv'
            subject_acc_pattern = 'ambiguity_group_no_temporal_PRE_{}_subject_accuracies.csv'
            shap_pattern = 'shap_importance_{}_ambiguity_PRE.csv'
            notebook = 'notebooks/fusion_models_no_temporal/ambiguity_group_late_fusion_no_temporal.ipynb'

    elif feature_type == 'non_temporal' and has_eeg:
        result_dir = RESULTS_DIR / 'fusion_model_results_no_temporal_eeg_PRE'
        if grouping_var == 'Reaction_Time':
            comparison_file = result_dir / 'reaction_time_group_no_temporal_eeg_PRE_comparison.csv'
            weights_pattern = 'reaction_time_group_no_temporal_eeg_PRE_{}_weights.csv'
            subject_acc_pattern = 'reaction_time_group_no_temporal_eeg_PRE_{}_subject_accuracies.csv'
            shap_pattern = None
            notebook = 'notebooks/fusion_models_no_temporal/reaction_time_group_late_fusion_no_temporal_eeg.ipynb'
        else:  # Ambiguity
            comparison_file = result_dir / 'ambiguity_group_no_temporal_eeg_PRE_comparison.csv'
            weights_pattern = 'ambiguity_group_no_temporal_eeg_PRE_{}_weights.csv'
            subject_acc_pattern = 'ambiguity_group_no_temporal_eeg_PRE_{}_subject_accuracies.csv'
            shap_pattern = None
            notebook = 'notebooks/fusion_models_no_temporal/ambiguity_group_late_fusion_no_temporal_eeg.ipynb'
    else:
        return []

    results = []

    if not comparison_file.exists():
        return results

    # Read comparison file
    comparison_df = pd.read_csv(comparison_file)

    # Process each group
    for _, row in comparison_df.iterrows():
        group_name = row['Group']

        # Read weights for this group
        weights_file = result_dir / weights_pattern.format(group_name)
        weights_dict = {}
        if weights_file.exists():
            weights_df = pd.read_csv(weights_file)
            if 'Weight' in weights_df.columns:
                weights_dict = dict(zip(weights_df['Modality'], weights_df['Weight']))
            elif 'Weighted' in weights_df.columns:
                weights_dict = dict(zip(weights_df['Modality'], weights_df['Weighted']))

        # Read subject accuracies for this group
        subject_acc_file = result_dir / subject_acc_pattern.format(group_name)
        n_subjects = None
        if subject_acc_file.exists():
            subject_df = pd.read_csv(subject_acc_file)
            n_subjects = len(subject_df)

        # Get top features if SHAP available
        top_features = None
        if shap_pattern:
            shap_file = result_dir / shap_pattern.format(group_name)
            if shap_file.exists():
                top_features = extract_top_features(shap_file)

        result = {
            'Analysis_Type': analysis_type,
            'Feature_Type': feature_type,
            'Has_EEG': 'Yes' if has_eeg else 'No',
            'Grouping_Variable': grouping_var,
            'Group_Name': group_name,
            'Accuracy': row.get('Accuracy', np.nan),
            'Accuracy_SEM': row.get('Accuracy_SEM', np.nan),
            'Accuracy_SD': row.get('Accuracy_SD', np.nan),
            'F1_Score': row.get('F1-Score', np.nan),
            'F1_SEM': row.get('F1_SEM', np.nan),
            'Modality_Weights': get_modality_weights(weights_dict),
            'N_Subjects': n_subjects if n_subjects else row.get('N_Subjects', np.nan),
            'N_Trials': row.get('N_Trials', np.nan),
            'Top_10_Features': top_features,
            'Notebook': notebook,
            'Notes': f'{grouping_var} group: {group_name}'
        }
        results.append(result)

    return results


def main():
    """Main function to consolidate all results."""

    print("Starting consolidation of all analysis results...")

    # Process overall analyses
    print("\n1. Processing overall analyses...")

    # Temporal features, no EEG
    all_results.extend(process_overall_analysis('Late Fusion', 'temporal', has_eeg=False))

    # Temporal features, with EEG
    all_results.extend(process_overall_analysis('Late Fusion with EEG', 'temporal', has_eeg=True))

    # Non-temporal features, no EEG
    all_results.extend(process_overall_analysis('Late Fusion (No Temporal)', 'non_temporal', has_eeg=False))

    # Non-temporal features, with EEG
    all_results.extend(process_overall_analysis('Late Fusion (No Temporal) with EEG', 'non_temporal', has_eeg=True))

    # Process grouped analyses
    print("\n2. Processing reaction time grouped analyses...")

    # RT groups - temporal, no EEG
    all_results.extend(process_grouped_analysis('Late Fusion by RT', 'Reaction_Time', 'temporal', has_eeg=False))

    # RT groups - temporal, with EEG
    all_results.extend(process_grouped_analysis('Late Fusion by RT with EEG', 'Reaction_Time', 'temporal', has_eeg=True))

    # RT groups - non-temporal, no EEG
    all_results.extend(process_grouped_analysis('Late Fusion by RT (No Temporal)', 'Reaction_Time', 'non_temporal', has_eeg=False))

    # RT groups - non-temporal, with EEG
    all_results.extend(process_grouped_analysis('Late Fusion by RT (No Temporal) with EEG', 'Reaction_Time', 'non_temporal', has_eeg=True))

    print("\n3. Processing ambiguity grouped analyses...")

    # Ambiguity groups - temporal, no EEG
    all_results.extend(process_grouped_analysis('Late Fusion by Ambiguity', 'Ambiguity', 'temporal', has_eeg=False))

    # Ambiguity groups - temporal, with EEG
    all_results.extend(process_grouped_analysis('Late Fusion by Ambiguity with EEG', 'Ambiguity', 'temporal', has_eeg=True))

    # Ambiguity groups - non-temporal, no EEG
    all_results.extend(process_grouped_analysis('Late Fusion by Ambiguity (No Temporal)', 'Ambiguity', 'non_temporal', has_eeg=False))

    # Ambiguity groups - non-temporal, with EEG
    all_results.extend(process_grouped_analysis('Late Fusion by Ambiguity (No Temporal) with EEG', 'Ambiguity', 'non_temporal', has_eeg=True))

    # Create DataFrame
    print(f"\n4. Creating consolidated DataFrame with {len(all_results)} rows...")
    results_df = pd.DataFrame(all_results)

    # Reorder columns
    column_order = [
        'Analysis_Type',
        'Feature_Type',
        'Has_EEG',
        'Grouping_Variable',
        'Group_Name',
        'Accuracy',
        'Accuracy_SEM',
        'Accuracy_SD',
        'F1_Score',
        'F1_SEM',
        'Modality_Weights',
        'N_Subjects',
        'N_Trials',
        'Top_10_Features',
        'Notebook',
        'Notes'
    ]

    results_df = results_df[column_order]

    # Sort by analysis type and group
    results_df = results_df.sort_values(['Analysis_Type', 'Grouping_Variable', 'Group_Name'])

    # Save to CSV
    output_file = RESULTS_DIR / 'consolidated_all_analyses.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n✓ Consolidated results saved to: {output_file}")
    print(f"  Total analyses: {len(results_df)}")
    print(f"\nSummary by Analysis Type:")
    print(results_df['Analysis_Type'].value_counts().to_string())

    return results_df


if __name__ == '__main__':
    df = main()
    print("\nFirst few rows:")
    print(df.head(10).to_string())
