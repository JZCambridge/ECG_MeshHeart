"""
ECG-to-Motion VAE Package

Trains a Variational Autoencoder (VAE) to encode 12-lead ECG + demographics + morphology
into a latent distribution (mu, logvar) for 512-dimensional motion latent space.

Compatible with hybrid_example encoder architecture.

Main components:
- model_resnet1d_morphology_vae: VAE encoder model (outputs mu, logvar)
- main_preprocess_motion_vae: Training script
- inference_motion_vae: Inference with stochastic sampling
- loader_ecg_preprocessed: Fast dataloader for preprocessed ECG data

Key differences from supervised version:
- Outputs (mu, logvar) instead of direct predictions
- Uses VAE loss: MSE reconstruction + beta * KL divergence
- Supports stochastic sampling for inference: z = mu + sigma * epsilon
- Checkpoint compatible with hybrid_example encoder

Author: Adapted from echonext_preprocess_motion
"""

__version__ = "1.0.0"
__author__ = "Adapted from echonext_preprocess_motion"

from .model_resnet1d_morphology_vae import (
    ECGMotionEncoderVAE,
    ResNet1dWithTabular,
    BasicBlock1d,
    make_model,
)

from .loader_ecg_preprocessed import (
    ECGMotionDatasetPreprocessed,
    MotionDataModulePreprocessed,
)

__all__ = [
    "ECGMotionEncoderVAE",
    "ResNet1dWithTabular",
    "BasicBlock1d",
    "make_model",
    "ECGMotionDatasetPreprocessed",
    "MotionDataModulePreprocessed",
]
