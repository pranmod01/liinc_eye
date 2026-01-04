"""
Shared visualization functions for multimodal fusion analysis.

This module provides reusable plotting functions to eliminate code duplication
across analysis notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


def plot_method_comparison(comparison_df: pd.DataFrame,
                           timeframe: str = 'PRE',
                           figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot accuracy and F1-score comparison across different methods.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: 'Method', 'Accuracy', 'F1-Score'
    timeframe : str
        'PRE' or 'POST' for title labeling
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Accuracy comparison
    comparison_df.plot(x='Method', y='Accuracy', kind='barh', ax=ax[0],
                      legend=False, color='steelblue')
    ax[0].set_xlabel('Accuracy')
    ax[0].set_xlim([0, 1])
    ax[0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax[0].set_title(f'{timeframe}-Decision: Accuracy')
    ax[0].grid(alpha=0.3, axis='x')

    # F1-Score comparison
    comparison_df.plot(x='Method', y='F1-Score', kind='barh', ax=ax[1],
                      legend=False, color='coral')
    ax[1].set_xlabel('F1-Score')
    ax[1].set_xlim([0, 1])
    ax[1].set_title(f'{timeframe}-Decision: F1-Score')
    ax[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_modality_weights(weights_df: pd.DataFrame,
                          timeframe: str = 'PRE',
                          figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot modality weights for different fusion methods.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame with columns: 'Modality', 'Average', 'Weighted', 'Stacking'
    timeframe : str
        'PRE' or 'POST' for title labeling
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = ['steelblue', 'coral', 'mediumseagreen']
    modality_names = weights_df['Modality'].values

    for idx, method in enumerate(['Average', 'Weighted', 'Stacking']):
        ax = axes[idx]
        weights = weights_df[method].values
        bars = ax.bar(modality_names, weights, color=colors)
        ax.set_ylabel('Weight')
        ax.set_title(f'{method} Fusion ({timeframe})')
        ax.set_ylim([0, max(weights) * 1.2])
        ax.grid(alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(weights):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_group_comparison(comparison_df: pd.DataFrame,
                         group_column: str = 'Group',
                         metric: str = 'Accuracy',
                         error_column: Optional[str] = 'Accuracy_SEM',
                         ylabel: str = 'Accuracy',
                         title: str = 'Performance by Group',
                         figsize: Tuple[int, int] = (10, 6),
                         color: str = 'steelblue') -> plt.Figure:
    """
    Plot performance metric across groups with error bars.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with group comparison results
    group_column : str
        Column name containing group labels
    metric : str
        Column name for the metric to plot
    error_column : str or None
        Column name for error bars (typically SEM)
    ylabel : str
        Y-axis label
    title : str
        Plot title
    figsize : tuple
        Figure size
    color : str
        Bar/point color

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(comparison_df))

    if error_column and error_column in comparison_df.columns:
        ax.errorbar(x_pos, comparison_df[metric],
                   yerr=comparison_df[error_column],
                   fmt='o-', capsize=5, color=color, markersize=8, linewidth=2)
    else:
        ax.plot(x_pos, comparison_df[metric], 'o-', color=color,
               markersize=8, linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_df[group_column])
    ax.set_xlabel(group_column)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Chance')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_group_comparison_dual(comparison_df: pd.DataFrame,
                               group_column: str = 'Group',
                               figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot both accuracy and F1-score across groups side-by-side.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: group_column, 'Accuracy', 'Accuracy_SEM',
        'F1-Score', 'F1_SEM'
    group_column : str
        Column name containing group labels
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Accuracy
    ax = axes[0]
    ax.errorbar(comparison_df[group_column], comparison_df['Accuracy'],
               yerr=comparison_df.get('Accuracy_SEM', 0),
               fmt='o-', capsize=5, color='steelblue', markersize=8)
    ax.set_xlabel(group_column)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy by {group_column} (error bars = SEM)')
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.grid(alpha=0.3)

    # F1-Score
    ax = axes[1]
    ax.errorbar(comparison_df[group_column], comparison_df['F1-Score'],
               yerr=comparison_df.get('F1_SEM', 0),
               fmt='o-', capsize=5, color='coral', markersize=8)
    ax.set_xlabel(group_column)
    ax.set_ylabel('F1-Score')
    ax.set_title(f'F1-Score by {group_column} (error bars = SEM)')
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_subject_distribution(group_results: Dict,
                              figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot subject-level accuracy distributions for each group.

    Parameters
    ----------
    group_results : dict
        Dictionary mapping group names to result dictionaries containing
        'accuracy_per_subject', 'accuracy_mean', 'accuracy_sem', 'n_subjects'
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    n_groups = len(group_results)
    fig, axes = plt.subplots(1, n_groups, figsize=figsize)

    if n_groups == 1:
        axes = [axes]

    for idx, (group, results) in enumerate(group_results.items()):
        ax = axes[idx]
        subject_accs = results['accuracy_per_subject']

        # Histogram
        ax.hist(subject_accs, bins=min(10, len(subject_accs)//5 + 1),
               color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(results['accuracy_mean'], color='red', linestyle='--',
                  linewidth=2, label=f"Mean: {results['accuracy_mean']:.3f}")
        ax.axvline(results['accuracy_mean'] - results['accuracy_sem'],
                  color='orange', linestyle=':', linewidth=1.5, label=f"±SEM")
        ax.axvline(results['accuracy_mean'] + results['accuracy_sem'],
                  color='orange', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count (subjects)')
        ax.set_title(f'{group} (n={results["n_subjects"]} subjects)')
        ax.set_xlim([0, 1])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Subject-Level Accuracy Distribution by Group',
                fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_modality_weights_by_group(comparison_df: pd.DataFrame,
                                   group_column: str = 'Group',
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot modality weights across different groups.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: group_column, 'Physiology_Weight',
        'Behavior_Weight', 'Gaze_Weight'
    group_column : str
        Column name containing group labels
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(comparison_df))
    width = 0.25

    ax.bar(x - width, comparison_df['Physiology_Weight'], width,
          label='Physiology', color='steelblue')
    ax.bar(x, comparison_df['Behavior_Weight'], width,
          label='Behavior', color='coral')
    ax.bar(x + width, comparison_df['Gaze_Weight'], width,
          label='Gaze', color='mediumseagreen')

    ax.set_xlabel(group_column)
    ax.set_ylabel('Weight')
    ax.set_title(f'Modality Weights by {group_column}')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[group_column])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_multi_seed_convergence(seed_results: List[Dict],
                                figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot accuracy convergence across multiple random seeds.

    Parameters
    ----------
    seed_results : list of dict
        List of result dictionaries, each containing 'seed', 'accuracy_mean',
        'accuracy_sem', 'f1_mean', 'f1_sem'
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    seeds = [r['seed'] for r in seed_results]
    accs = [r['accuracy_mean'] for r in seed_results]
    acc_sems = [r['accuracy_sem'] for r in seed_results]
    f1s = [r['f1_mean'] for r in seed_results]
    f1_sems = [r['f1_sem'] for r in seed_results]

    # Accuracy across seeds
    ax = axes[0]
    ax.errorbar(seeds, accs, yerr=acc_sems, fmt='o-', capsize=5,
               color='steelblue', markersize=8)
    ax.axhline(np.mean(accs), color='red', linestyle='--', alpha=0.5,
              label=f'Mean: {np.mean(accs):.3f}')
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Across Random Seeds')
    ax.legend()
    ax.grid(alpha=0.3)

    # F1-Score across seeds
    ax = axes[1]
    ax.errorbar(seeds, f1s, yerr=f1_sems, fmt='o-', capsize=5,
               color='coral', markersize=8)
    ax.axhline(np.mean(f1s), color='red', linestyle='--', alpha=0.5,
              label=f'Mean: {np.mean(f1s):.3f}')
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Across Random Seeds')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_visit_performance(comparison_df: pd.DataFrame,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot performance across experimental visits.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: 'Visit', 'Accuracy', 'Accuracy_SEM',
        'F1-Score', 'F1_SEM', 'N_Subjects', 'N_Trials'
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Accuracy by visit
    ax = axes[0]
    ax.errorbar(comparison_df['Visit'], comparison_df['Accuracy'],
               yerr=comparison_df.get('Accuracy_SEM', 0),
               fmt='o-', capsize=5, color='steelblue', markersize=8, linewidth=2)
    ax.set_xlabel('Visit Number')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Across Experimental Visits')
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.grid(alpha=0.3)

    # Sample size by visit
    ax = axes[1]
    ax.bar(comparison_df['Visit'], comparison_df['N_Subjects'],
          color='steelblue', alpha=0.7, label='Subjects')
    ax.set_xlabel('Visit Number')
    ax.set_ylabel('Count')
    ax.set_title('Sample Size by Visit')
    ax2 = ax.twinx()
    ax2.plot(comparison_df['Visit'], comparison_df['N_Trials'],
            'o-', color='coral', markersize=8, linewidth=2, label='Trials')
    ax2.set_ylabel('Trials', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def set_style(style: str = 'whitegrid'):
    """
    Set consistent plotting style for all visualizations.

    Parameters
    ----------
    style : str
        Seaborn style name ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
    """
    sns.set_style(style)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
