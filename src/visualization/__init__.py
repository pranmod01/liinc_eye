"""Visualization utilities for multimodal fusion analysis."""

from .plots import (
    plot_method_comparison,
    plot_modality_weights,
    plot_group_comparison,
    plot_group_comparison_dual,
    plot_subject_distribution,
    plot_modality_weights_by_group,
    plot_multi_seed_convergence,
    plot_visit_performance,
    set_style
)

__all__ = [
    'plot_method_comparison',
    'plot_modality_weights',
    'plot_group_comparison',
    'plot_group_comparison_dual',
    'plot_subject_distribution',
    'plot_modality_weights_by_group',
    'plot_multi_seed_convergence',
    'plot_visit_performance',
    'set_style'
]
