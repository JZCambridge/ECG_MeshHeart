"""
ResNet1D-based ECG-to-Motion Encoder Model with Morphology Features
Uses ResNet1dWithTabular to encode ECG directly to 512-dim motion latents
Deterministic encoder with 3-layer MLP output head (no VAE components)
"""

import os
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sample_weights(clinical_features, weight_config):
    """
    Compute per-sample weights based on clinical thresholds using MAX rule.

    Args:
        clinical_features: [B, 3] tensor with [lvef, wt_max, lvedvi]
        weight_config: Dict with thresholds and weights

    Returns:
        sample_weights: [B] tensor with per-sample weights (uses MAX)
    """
    batch_size = clinical_features.shape[0]
    device = clinical_features.device
    weights = torch.ones(batch_size, device=device)

    # Extract clinical values
    lvef = clinical_features[:, 0]
    wt_max = clinical_features[:, 1]
    lvedvi = clinical_features[:, 2]

    # Apply conditions and take MAX
    if weight_config.get('use_lvef', False):
        lvef_mask = lvef < weight_config['lvef_threshold']
        lvef_weights = torch.where(
            lvef_mask,
            torch.tensor(weight_config['lvef_weight'], device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32)
        )
        weights = torch.maximum(weights, lvef_weights)

    if weight_config.get('use_wt', False):
        wt_mask = wt_max > weight_config['wt_threshold']
        wt_weights = torch.where(
            wt_mask,
            torch.tensor(weight_config['wt_weight'], device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32)
        )
        weights = torch.maximum(weights, wt_weights)

    if weight_config.get('use_lvedvi', False):
        lvedvi_mask = lvedvi > weight_config['lvedvi_threshold']
        lvedvi_weights = torch.where(
            lvedvi_mask,
            torch.tensor(weight_config['lvedvi_weight'], device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32)
        )
        weights = torch.maximum(weights, lvedvi_weights)

    return weights


def compute_subgroup_metrics(predictions, ground_truths, clinical_features, subgroup_config=None):
    """
    Compute MSE and MAE for overall and clinical subgroups.

    Args:
        predictions: [N, 512] numpy array
        ground_truths: [N, 512] numpy array
        clinical_features: [N, 3] numpy array with [lvef, wt_max, lvedvi]
        subgroup_config: Dict with subgroup definitions (unused for now)

    Returns:
        Dict with metrics for each subgroup
    """
    results = {}

    # Overall metrics
    mse_overall = np.mean((predictions - ground_truths) ** 2)
    mae_overall = np.mean(np.abs(predictions - ground_truths))
    results["overall"] = {"mse": mse_overall, "mae": mae_overall, "count": len(predictions)}

    # Extract clinical values
    lvef = clinical_features[:, 0]
    wt_max = clinical_features[:, 1]
    lvedvi = clinical_features[:, 2]

    # Define subgroups
    subgroups = [
        ("lvef_lt_50", lvef < 50),
        ("lvef_lt_45", lvef < 45),
        ("wt_gt_13", wt_max > 13),
        ("wt_gt_15", wt_max > 15),
        ("lvedvi_gt_75", lvedvi > 75),
        ("lvedvi_gt_62", lvedvi > 62),
    ]

    # Compute metrics for each subgroup
    for name, mask in subgroups:
        if mask.sum() > 0:
            pred_sub = predictions[mask]
            gt_sub = ground_truths[mask]
            mse = np.mean((pred_sub - gt_sub) ** 2)
            mae = np.mean(np.abs(pred_sub - gt_sub))
            results[name] = {"mse": mse, "mae": mae, "count": int(mask.sum())}
        else:
            results[name] = {"mse": np.nan, "mae": np.nan, "count": 0}

    return results


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.5,
        kernel_size: int = 7,
        padding: int = 3,
        bias: bool = False,
        inplace: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=inplace)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=bias
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1dWithTabular(nn.Module):
    """ResNet1D backbone with 3-layer MLP output head for direct 512-dim prediction"""

    def __init__(
        self,
        len_tabular_feature_vector: int,
        filter_size: int = 64,
        input_channels: int = 12,
        dropout_value: float = 0.5,
        num_classes: int = 512,  # Direct 512-dim output
        conv1_kernel_size: int = 15,
        conv1_stride: int = 2,
        padding: int = 7,
        bias: bool = False,
    ):
        super().__init__()
        self.inplanes = filter_size
        self.layers = [3, 4, 6, 3]
        self.conv1 = nn.Conv1d(
            input_channels,
            self.inplanes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=padding,
            bias=bias,
        )
        self.dropout_value = dropout_value
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, filter_size, self.layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 2 * filter_size, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 4 * filter_size, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 8 * filter_size, self.layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)

        # 3-layer MLP output head: [concat_dim] → [1024] → [1024] → [512]
        intermediate_dim = 8 * filter_size * BasicBlock1d.expansion * 2 + len_tabular_feature_vector
        self.output_mlp = nn.Sequential(
            nn.Linear(intermediate_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),
            nn.Linear(1024, 512)  # Final output: 512 motion latents
        )
        self.dropout = nn.Dropout(dropout_value)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout=self.dropout_value))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=self.dropout_value))
        return nn.Sequential(*layers)

    def forward(self, x_and_tabular: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, tabular = x_and_tabular
        # Change the shape of x to be adopted for the model input
        # Prior to transpose and squeeze, expect model input size to be [BATCH_SIZE, 1, FEATURE_SIZE, 12]
        x = torch.transpose(x, 2, 3)
        x = torch.squeeze(x, 1)
        # Input size after transpose and squeeze: [BATCH_SIZE, 12, FEATURE_SIZE]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        # Concat with tabular
        x = torch.cat((x, tabular), dim=1)
        # Forward through 3-layer MLP
        out = self.output_mlp(x)
        return out


class ECGMotionEncoder(nn.Module):
    """
    ECG-to-Motion Encoder using ResNet1D + Tabular data.
    Encodes 12-lead ECG + demographics + morphology → 512-dim motion latents (deterministic).

    Key features:
    - Deterministic encoder (no variational components)
    - 3-layer MLP output head
    - MSE loss only (no KL divergence)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        checkpoint_dir: Optional[str] = None,
        num_epochs: int = 100,
        weight_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        self.weight_config = weight_config or {}

        # Training metrics storage (simplified - no KL or correlation)
        self.train_losses = []
        self.val_losses = []
        self.train_mae_losses = []
        self.val_mae_losses = []

        self.best_val_loss = float("inf")
        self.best_val_mae = float("inf")
        self.epoch = 0

    def forward(self, batch: Dict):
        """
        Forward pass through encoder.

        Args:
            batch: Dictionary containing 'ecg_raw' [B, 12, 2500], 'demographics' [B, 8],
                   'ecg_morphology' [B, 16], and 'motion_latent' [B, 512]

        Returns:
            Dictionary with prediction and ground truth
        """
        ecg_raw = batch["ecg_raw"]  # [B, 12, 2500]
        demographics = batch["demographics"]  # [B, 8]
        ecg_morphology = batch["ecg_morphology"]  # [B, 16]

        # Concatenate demographics and morphology → [B, 24]
        tabular = torch.cat([demographics, ecg_morphology], dim=1)

        # Reshape ECG to match expected input: [B, 1, 2500, 12]
        ecg_reshaped = ecg_raw.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, 2500, 12]

        # Forward through ResNet + MLP → [B, 512]
        prediction = self.model((ecg_reshaped, tabular))

        return {
            "prediction": prediction,
            "motion_gt": batch["motion_latent"],
        }

    def loss_function(self, out: Dict, sample_weights=None):
        """
        Compute MSE loss between prediction and ground truth with optional weights.

        Args:
            out: Output dictionary from forward pass
            sample_weights: Optional [B] tensor with per-sample weights

        Returns:
            loss: Weighted MSE loss (scalar tensor)
        """
        prediction = out["prediction"]  # [B, 512]
        motion_gt = out["motion_gt"]    # [B, 512]

        if sample_weights is not None:
            # Compute per-sample MSE, then apply weights
            mse_per_sample = F.mse_loss(prediction, motion_gt, reduction="none").mean(dim=1)  # [B]
            weighted_loss = (mse_per_sample * sample_weights).mean()
            return weighted_loss
        else:
            # Standard MSE loss
            loss = F.mse_loss(prediction, motion_gt, reduction="mean")
            return loss

    def compute_mae(self, prediction: torch.Tensor, motion_gt: torch.Tensor):
        """
        Compute Mean Absolute Error between prediction and ground truth.

        Args:
            prediction: [B, 512] - model prediction
            motion_gt: [B, 512] - ground truth motion latents

        Returns:
            mae: scalar - mean absolute error
        """
        mae = F.l1_loss(prediction, motion_gt, reduction="mean")
        return mae

    def save_checkpoint(self, optimizer, epoch, val_loss, prefix="checkpoint"):
        """
        Save model checkpoint.

        Args:
            optimizer: Optimizer state
            epoch: Current epoch
            val_loss: Validation loss
            prefix: Checkpoint filename prefix
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }

        if self.checkpoint_dir is None:
            raise ValueError("Checkpoint directory not specified")

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{prefix}.ckpt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def results_table(self):
        """
        Create a pandas DataFrame with training results.

        Returns:
            DataFrame with loss metrics
        """
        df = pd.DataFrame.from_dict({
            "train_loss": [x.item() if torch.is_tensor(x) else x for x in self.train_losses],
            "val_loss": [x.item() if torch.is_tensor(x) else x for x in self.val_losses],
            "train_mae": [x.item() if torch.is_tensor(x) else x for x in self.train_mae_losses],
            "val_mae": [x.item() if torch.is_tensor(x) else x for x in self.val_mae_losses],
        })

        return df


def make_model(config: Dict, weight_config: Optional[Dict] = None):
    """
    Create ResNet1D-based ECG-to-Motion Encoder model.

    Args:
        config: Dictionary with model hyperparameters
        weight_config: Optional dictionary with weighted loss configuration

    Returns:
        ECGMotionEncoder model
    """
    # Create ResNet1D model with 24 tabular features and direct 512 output
    resnet_model = ResNet1dWithTabular(
        len_tabular_feature_vector=24,  # 8 demographics + 16 morphology
        filter_size=config.get("filter_size", 64),
        input_channels=12,  # 12 ECG leads
        dropout_value=config.get("dropout", 0.5),
        num_classes=512,  # Direct 512-dim output
        conv1_kernel_size=config.get("conv1_kernel_size", 15),
        conv1_stride=config.get("conv1_stride", 2),
        padding=config.get("padding", 7),
    )

    # Create full encoder model wrapper with weight config
    model = ECGMotionEncoder(
        model=resnet_model,
        lr=config["lr"],
        checkpoint_dir=config["checkpoint_dir"],
        num_epochs=config["epochs"],
        weight_config=weight_config,
    )

    return model


if __name__ == "__main__":
    # Test Encoder model creation
    print("=== Testing ResNet1D ECGMotionEncoder with Morphology ===")

    config = {
        "filter_size": 64,
        "dropout": 0.5,
        "lr": 1e-3,
        "checkpoint_dir": "./test_checkpoints",
        "epochs": 100,
    }

    model = make_model(config)

    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 8
    dummy_batch = {
        "ecg_raw": torch.randn(batch_size, 12, 2500),
        "demographics": torch.randn(batch_size, 8),
        "ecg_morphology": torch.randn(batch_size, 16),
        "motion_latent": torch.randn(batch_size, 512),
    }

    out = model(dummy_batch)
    print(f"\nForward pass test:")
    print(f"  Input ECG shape: {dummy_batch['ecg_raw'].shape}")
    print(f"  Input demographics shape: {dummy_batch['demographics'].shape}")
    print(f"  Input morphology shape: {dummy_batch['ecg_morphology'].shape}")
    print(f"  Output prediction shape: {out['prediction'].shape}")

    loss = model.loss_function(out)
    print(f"\nLoss computation test:")
    print(f"  MSE loss: {loss.item():.4f}")

    mae = model.compute_mae(out['prediction'], out['motion_gt'])
    print(f"  MAE: {mae.item():.4f}")

    print("\n✓ Encoder Model test completed!")
