"""
Combine all analysis results into a single comprehensive dataframe.

This script consolidates:
1. Late fusion model results (PRE/POST)
2. Ambiguity group analysis (balanced/unbalanced)
3. Reaction time group analysis (balanced/unbalanced)
4. Statistical testing results
5. Feature importance (SHAP) results
6. Rolling window analysis
"""

import pandas as pd
import pickle
from pathlib import Path
import numpy as np

# Base paths
BASE_PATH = Path('/Users/pranmodu/Projects/columbia/liinc/data/results')

def load_fusion_results(time_period):
    """Load late fusion model results for PRE or POST."""
    path = BASE_PATH / f'fusion_model_results_{time_period}'

    results = []

    # Method comparison
    comp_file = path / f'late_fusion_model_{time_period}_method_comparison.csv'
    if comp_file.exists():
        df = pd.read_csv(comp_file)
        df['analysis_type'] = 'late_fusion'
        df['time_period'] = time_period
        df['class_weighting'] = 'unbalanced'
        df['subgroup'] = 'none'
        df['file_source'] = comp_file.name
        results.append(df)

    # Subject accuracies
    subj_file = path / f'late_fusion_model_{time_period}_subject_accuracies.csv'
    if subj_file.exists():
        df = pd.read_csv(subj_file)
        df['analysis_type'] = 'late_fusion_subject'
        df['time_period'] = time_period
        df['class_weighting'] = 'unbalanced'
        df['subgroup'] = 'none'
        df['file_source'] = subj_file.name
        results.append(df)

    # Modality weights
    weight_file = path / f'late_fusion_model_{time_period}_modality_weights.csv'
    if weight_file.exists():
        df = pd.read_csv(weight_file)
        df['analysis_type'] = 'modality_weights'
        df['time_period'] = time_period
        df['class_weighting'] = 'unbalanced'
        df['subgroup'] = 'none'
        df['file_source'] = weight_file.name
        results.append(df)

    # Visit-based analysis
    visit_file = path / f'late_fusion_by_visit_{time_period}_comparison.csv'
    if visit_file.exists():
        df = pd.read_csv(visit_file)
        df['analysis_type'] = 'late_fusion_by_visit'
        df['time_period'] = time_period
        df['class_weighting'] = 'unbalanced'
        df['subgroup'] = 'none'
        df['file_source'] = visit_file.name
        results.append(df)

    return results

def load_ambiguity_results(time_period, balanced=False):
    """Load ambiguity group analysis results."""
    path = BASE_PATH / f'fusion_model_results_{time_period}'
    balance_str = '_balanced' if balanced else ''

    results = []

    for group in ['Low', 'Medium', 'High']:
        # Comparison file
        comp_file = path / f'ambiguity_group_late_fusion{balance_str}_{time_period}_{group}_comparison.csv'
        if comp_file.exists():
            df = pd.read_csv(comp_file)
            df['analysis_type'] = 'ambiguity_group'
            df['time_period'] = time_period
            df['class_weighting'] = 'balanced' if balanced else 'unbalanced'
            df['subgroup'] = group
            df['file_source'] = comp_file.name
            results.append(df)

        # Subject accuracies
        subj_file = path / f'ambiguity_group_late_fusion{balance_str}_{time_period}_{group}_subject_accuracies.csv'
        if subj_file.exists():
            df = pd.read_csv(subj_file)
            df['analysis_type'] = 'ambiguity_group_subject'
            df['time_period'] = time_period
            df['class_weighting'] = 'balanced' if balanced else 'unbalanced'
            df['subgroup'] = group
            df['file_source'] = subj_file.name
            results.append(df)

        # Weights
        weight_file = path / f'ambiguity_group_late_fusion{balance_str}_{time_period}_{group}_weights.csv'
        if weight_file.exists():
            df = pd.read_csv(weight_file)
            df['analysis_type'] = 'ambiguity_group_weights'
            df['time_period'] = time_period
            df['class_weighting'] = 'balanced' if balanced else 'unbalanced'
            df['subgroup'] = group
            df['file_source'] = weight_file.name
            results.append(df)

    return results

def load_rt_results(time_period, balanced=False):
    """Load reaction time group analysis results."""
    path = BASE_PATH / f'fusion_model_results_{time_period}'
    balance_str = '_balanced' if balanced else ''

    results = []

    for group in ['Fast', 'Slow']:
        # Comparison file
        comp_file = path / f'reaction_time_group_late_fusion{balance_str}_{time_period}_{group}_comparison.csv'
        if comp_file.exists():
            df = pd.read_csv(comp_file)
            df['analysis_type'] = 'reaction_time_group'
            df['time_period'] = time_period
            df['class_weighting'] = 'balanced' if balanced else 'unbalanced'
            df['subgroup'] = group
            df['file_source'] = comp_file.name
            results.append(df)

        # Subject accuracies
        subj_file = path / f'reaction_time_group_late_fusion{balance_str}_{time_period}_{group}_subject_accuracies.csv'
        if subj_file.exists():
            df = pd.read_csv(subj_file)
            df['analysis_type'] = 'reaction_time_group_subject'
            df['time_period'] = time_period
            df['class_weighting'] = 'balanced' if balanced else 'unbalanced'
            df['subgroup'] = group
            df['file_source'] = subj_file.name
            results.append(df)

        # Weights
        weight_file = path / f'reaction_time_group_late_fusion{balance_str}_{time_period}_{group}_weights.csv'
        if weight_file.exists():
            df = pd.read_csv(weight_file)
            df['analysis_type'] = 'reaction_time_group_weights'
            df['time_period'] = time_period
            df['class_weighting'] = 'balanced' if balanced else 'unbalanced'
            df['subgroup'] = group
            df['file_source'] = weight_file.name
            results.append(df)

    return results

def load_statistical_results(time_period):
    """Load statistical testing results."""
    path = BASE_PATH / f'analysis_outputs_{time_period}'

    results = []

    # McNemar's test summary
    mcnemar_file = path / f'mcnemar_test_summary_{time_period}.csv'
    if mcnemar_file.exists():
        df = pd.read_csv(mcnemar_file)
        df['analysis_type'] = 'mcnemar_test'
        df['time_period'] = time_period
        df['class_weighting'] = 'balanced'  # Statistical tests use balanced
        df['subgroup'] = 'none'
        df['file_source'] = mcnemar_file.name
        results.append(df)

    # Comprehensive statistical summary
    stat_file = path / f'statistical_testing_summary_{time_period}.csv'
    if stat_file.exists():
        df = pd.read_csv(stat_file)
        df['analysis_type'] = 'statistical_summary'
        df['time_period'] = time_period
        df['class_weighting'] = 'balanced'
        df['subgroup'] = 'none'
        df['file_source'] = stat_file.name
        results.append(df)

    return results

def load_shap_results(time_period):
    """Load SHAP feature importance results."""
    path = BASE_PATH / f'analysis_outputs_{time_period}'

    results = []

    modalities = ['all', 'behavior', 'gaze', 'physiology', 'combined', 'phys_gaze']

    for modality in modalities:
        shap_file = path / f'shap_importance_{modality}_{time_period}.csv'
        if shap_file.exists():
            df = pd.read_csv(shap_file)
            df['analysis_type'] = f'shap_{modality}'
            df['time_period'] = time_period
            df['class_weighting'] = 'na'
            df['subgroup'] = 'none'
            df['file_source'] = shap_file.name
            results.append(df)

    return results

def load_rolling_window_results():
    """Load rolling window post-decision analysis results."""
    path = BASE_PATH / 'fusion_model_results_POST'

    results = []

    # Summary
    summary_file = path / 'rolling_window_post_decision_summary.csv'
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['analysis_type'] = 'rolling_window_summary'
        df['time_period'] = 'POST_rolling'
        df['class_weighting'] = 'unbalanced'
        df['subgroup'] = 'none'
        df['file_source'] = summary_file.name
        results.append(df)

    # Detailed analysis
    detail_file = path / 'rolling_window_post_decision_analysis.csv'
    if detail_file.exists():
        df = pd.read_csv(detail_file)
        df['analysis_type'] = 'rolling_window_detailed'
        df['time_period'] = 'POST_rolling'
        df['class_weighting'] = 'unbalanced'
        df['subgroup'] = 'none'
        df['file_source'] = detail_file.name
        results.append(df)

    return results

def main():
    """Combine all results into comprehensive dataframe."""
    print("="*70)
    print("COMBINING ALL ANALYSIS RESULTS")
    print("="*70)

    all_results = []

    # 1. Late fusion results
    print("\n1. Loading late fusion results...")
    for time_period in ['PRE', 'POST']:
        results = load_fusion_results(time_period)
        all_results.extend(results)
        print(f"   ✓ Loaded {len(results)} files for {time_period}")

    # 2. Ambiguity group results
    print("\n2. Loading ambiguity group results...")
    for time_period in ['PRE', 'POST']:
        for balanced in [False, True]:
            results = load_ambiguity_results(time_period, balanced)
            all_results.extend(results)
            balance_str = 'balanced' if balanced else 'unbalanced'
            print(f"   ✓ Loaded {len(results)} files for {time_period} ({balance_str})")

    # 3. Reaction time group results
    print("\n3. Loading reaction time group results...")
    for time_period in ['PRE', 'POST']:
        for balanced in [False, True]:
            results = load_rt_results(time_period, balanced)
            all_results.extend(results)
            balance_str = 'balanced' if balanced else 'unbalanced'
            print(f"   ✓ Loaded {len(results)} files for {time_period} ({balance_str})")

    # 4. Statistical testing results
    print("\n4. Loading statistical testing results...")
    for time_period in ['PRE', 'POST']:
        results = load_statistical_results(time_period)
        all_results.extend(results)
        print(f"   ✓ Loaded {len(results)} files for {time_period}")

    # 5. SHAP feature importance
    print("\n5. Loading SHAP feature importance results...")
    for time_period in ['PRE', 'POST']:
        results = load_shap_results(time_period)
        all_results.extend(results)
        print(f"   ✓ Loaded {len(results)} files for {time_period}")

    # 6. Rolling window results
    print("\n6. Loading rolling window results...")
    results = load_rolling_window_results()
    all_results.extend(results)
    print(f"   ✓ Loaded {len(results)} files")

    # Combine all dataframes
    print(f"\n{'='*70}")
    print(f"Combining {len(all_results)} dataframes...")

    if not all_results:
        print("ERROR: No results found!")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(all_results, ignore_index=True, sort=False)

    print(f"✓ Combined dataframe shape: {combined_df.shape}")
    print(f"\nAnalysis types included:")
    for analysis_type in combined_df['analysis_type'].unique():
        count = (combined_df['analysis_type'] == analysis_type).sum()
        print(f"  - {analysis_type}: {count} rows")

    # Save combined results
    output_file = BASE_PATH / 'combined_all_results.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"✓ SAVED: {output_file}")
    print(f"{'='*70}")

    # Print summary statistics
    print("\nSUMMARY BY DIMENSIONS:")
    print(f"\n1. Analysis Types: {combined_df['analysis_type'].nunique()}")
    print(f"   {sorted(combined_df['analysis_type'].unique())}")

    print(f"\n2. Time Periods: {combined_df['time_period'].nunique()}")
    print(f"   {sorted(combined_df['time_period'].unique())}")

    print(f"\n3. Class Weighting: {combined_df['class_weighting'].nunique()}")
    print(f"   {sorted(combined_df['class_weighting'].unique())}")

    print(f"\n4. Subgroups: {combined_df['subgroup'].nunique()}")
    print(f"   {sorted(combined_df['subgroup'].unique())}")

    # Create pivot summary
    print(f"\n{'='*70}")
    print("CREATING PIVOT SUMMARY...")
    print(f"{'='*70}")

    # Create summary by analysis type
    summary_data = []

    for analysis_type in combined_df['analysis_type'].unique():
        subset = combined_df[combined_df['analysis_type'] == analysis_type]

        summary_data.append({
            'analysis_type': analysis_type,
            'n_rows': len(subset),
            'time_periods': ', '.join(sorted(subset['time_period'].unique())),
            'class_weighting': ', '.join(sorted(subset['class_weighting'].unique())),
            'subgroups': ', '.join(sorted(subset['subgroup'].unique())),
            'n_files': subset['file_source'].nunique()
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = BASE_PATH / 'combined_results_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print(f"\n✓ SAVED SUMMARY: {summary_file}")
    print("\n" + summary_df.to_string(index=False))

    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  1. {output_file.name} - Full combined dataset ({combined_df.shape[0]:,} rows)")
    print(f"  2. {summary_file.name} - Summary by analysis type")

    return combined_df, summary_df

if __name__ == "__main__":
    combined_df, summary_df = main()
