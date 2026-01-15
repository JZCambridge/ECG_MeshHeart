"""
ECG-to-Motion Encoder Package

Trains a deterministic encoder to map 12-lead ECG + demographics + morphology
directly to 512-dimensional motion latent space.

Main components:
- model_resnet1d_morphology_encoder: Deterministic encoder model
- main_motion_latent_generation: Training script
- inference_motion_latent_encoder: Inference script
- loader_ecg_preprocessed: Data loader for preprocessed ECG data

Key features:
- Direct 512-dimensional output (no VAE splitting)
- 3-layer MLP output head: [1024] → [1024] → [1024] → [512]
- MSE loss only (no KL divergence)
- Improved logging with file + console + TensorBoard

Author: Converted from VAE version
"""

__version__ = "2.0.0"
__author__ = "Converted from VAE version"

from .model_resnet1d_morphology_encoder import (
    ECGMotionEncoder,
    ResNet1dWithTabular,
    BasicBlock1d,
    make_model,
)

from .loader_ecg_preprocessed import (
    ECGMotionDatasetPreprocessed,
    MotionDataModulePreprocessed,
)

__all__ = [
    "ECGMotionEncoder",
    "ResNet1dWithTabular",
    "BasicBlock1d",
    "make_model",
    "ECGMotionDatasetPreprocessed",
    "MotionDataModulePreprocessed",
]
