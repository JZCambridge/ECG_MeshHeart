"""
ECG to Motion Latent Generation with Weighted Loss and Subgroup Metrics

This package provides weighted training loss and detailed subgroup metrics
for ECG-to-motion latent space generation.
"""

from .model_resnet1d_morphology_encoder_weighted import (
    ECGMotionEncoder,
    make_model,
    compute_sample_weights,
    compute_subgroup_metrics,
)
from .loader_ecg_preprocessed_weighted import (
    MotionDataModulePreprocessed,
    ECGMotionDatasetPreprocessed,
)

__all__ = [
    'ECGMotionEncoder',
    'make_model',
    'compute_sample_weights',
    'compute_subgroup_metrics',
    'MotionDataModulePreprocessed',
    'ECGMotionDatasetPreprocessed',
]
