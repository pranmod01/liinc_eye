"""
Machine learning models for multimodal fusion.
"""

from .fusion import (
    weighted_late_fusion,
    weighted_late_fusion_train_optimal_bins,
)

__all__ = ['weighted_late_fusion', 'weighted_late_fusion_train_optimal_bins']
