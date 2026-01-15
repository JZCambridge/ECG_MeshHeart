"""
ResNet1D-based ECG-to-Motion Prediction Model with Morphology Features
Uses ResNet1dWithTabular for motion latent prediction from preprocessed ECG + demographics + morphology
"""

import os
from typing import Dict, Optional, Tuple, Type

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(
        self,
        len_tabular_feature_vector: int,
        filter_size: int = 64,
        input_channels: int = 12,
        dropout_value: float = 0.5,
        num_classes: int = 64,  # 64-dimensional motion latent
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
        self.output = nn.Linear(
            8 * filter_size * BasicBlock1d.expansion * 2 + len_tabular_feature_vector, num_classes
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
        out = self.output(x)
        return out


class ECGMotionEncoder(nn.Module):
    """
    ECG-to-Motion Encoder using ResNet1D + Tabular data (demographics + morphology).
    Predicts 64-dimensional motion latents from 12-lead ECG (2500 timepoints) + demographics + morphology.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        checkpoint_dir: Optional[str] = None,
        num_epochs: int = 100,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs

        # Training metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_correlations = []

        self.best_val_loss = float("inf")
        self.best_val_corr = 0.0
        self.epoch = 0

    def forward(self, batch: Dict):
        """
        Forward pass through ResNet1D model.

        Args:
            batch: Dictionary containing 'ecg_raw' [B, 12, 2500], 'demographics' [B, 9],
                   'ecg_morphology' [B, 16], and 'motion_latent' [B, 64]

        Returns:
            Dictionary with predictions and ground truth
        """
        ecg_raw = batch["ecg_raw"]  # [B, 12, 2500]
        demographics = batch["demographics"]  # [B, 9]
        ecg_morphology = batch["ecg_morphology"]  # [B, 16]

        # Concatenate demographics and morphology → [B, 25]
        tabular = torch.cat([demographics, ecg_morphology], dim=1)

        # Reshape ECG to match expected input: [B, 1, 2500, 12]
        ecg_reshaped = ecg_raw.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, 2500, 12]

        # Forward through model
        motion_pred = self.model((ecg_reshaped, tabular))  # [B, 64]

        return {
            "motion_pred": motion_pred,
            "motion_gt": batch["motion_latent"],
        }

    def loss_function(self, out: Dict):
        """
        Compute MSE loss between predicted and ground truth motion latents.

        Args:
            out: Output dictionary from forward pass

        Returns:
            MSE loss
        """
        mse_loss = F.mse_loss(out["motion_pred"], out["motion_gt"], reduction="mean")
        return mse_loss

    def compute_mean_correlation(self, motion_pred: torch.Tensor, motion_gt: torch.Tensor):
        """
        Compute mean Pearson correlation across all 64 dimensions.

        Args:
            motion_pred: [B, 64] - predicted motion latents
            motion_gt: [B, 64] - ground truth motion latents

        Returns:
            mean_corr: scalar - mean Pearson correlation
        """
        correlations = []

        for dim in range(motion_pred.shape[1]):  # Iterate over 64 dimensions
            pred_dim = motion_pred[:, dim]
            gt_dim = motion_gt[:, dim]

            # Pearson correlation
            pred_centered = pred_dim - pred_dim.mean()
            gt_centered = gt_dim - gt_dim.mean()

            numerator = (pred_centered * gt_centered).sum()
            denominator = torch.sqrt((pred_centered ** 2).sum() * (gt_centered ** 2).sum())

            if denominator > 0:
                corr = numerator / denominator
                correlations.append(corr)

        if len(correlations) > 0:
            mean_corr = torch.stack(correlations).mean()
        else:
            mean_corr = torch.tensor(0.0)

        return mean_corr

    def save_checkpoint(self, optimizer, epoch, val_loss, val_corr, prefix="checkpoint"):
        """
        Save model checkpoint.

        Args:
            optimizer: Optimizer state
            epoch: Current epoch
            val_loss: Validation loss
            val_corr: Validation correlation
            prefix: Checkpoint filename prefix
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_corr': val_corr,
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
            "val_correlation": [x.item() if torch.is_tensor(x) else x for x in self.val_correlations],
        })

        return df


def make_model(config: Dict):
    """
    Create ResNet1D-based ECG-to-Motion encoder model with morphology features.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        ECGMotionEncoder model
    """
    # Create ResNet1D model with 25 tabular features (9 demographics + 16 morphology)
    resnet_model = ResNet1dWithTabular(
        len_tabular_feature_vector=25,  # 9 demographics + 16 morphology
        filter_size=config.get("filter_size", 64),
        input_channels=12,  # 12 ECG leads
        dropout_value=config.get("dropout", 0.5),
        num_classes=64,  # 64-dimensional motion latent
        conv1_kernel_size=config.get("conv1_kernel_size", 15),
        conv1_stride=config.get("conv1_stride", 2),
        padding=config.get("padding", 7),
    )

    # Create full model wrapper
    model = ECGMotionEncoder(
        model=resnet_model,
        lr=config["lr"],
        checkpoint_dir=config["checkpoint_dir"],
        num_epochs=config["epochs"],
    )

    return model


if __name__ == "__main__":
    # Test model creation
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
        "ecg_raw": torch.randn(batch_size, 12, 2500),  # 2500 timepoints
        "demographics": torch.randn(batch_size, 9),
        "ecg_morphology": torch.randn(batch_size, 16),
        "motion_latent": torch.randn(batch_size, 64),
    }

    out = model(dummy_batch)
    print(f"\nForward pass test:")
    print(f"  Input ECG shape: {dummy_batch['ecg_raw'].shape}")
    print(f"  Input demographics shape: {dummy_batch['demographics'].shape}")
    print(f"  Input morphology shape: {dummy_batch['ecg_morphology'].shape}")
    print(f"  Predicted motion shape: {out['motion_pred'].shape}")

    loss = model.loss_function(out)
    print(f"  MSE loss: {loss.item():.4f}")

    corr = model.compute_mean_correlation(out['motion_pred'], out['motion_gt'])
    print(f"  Mean correlation: {corr.item():.4f}")

    print("\n✓ Model test completed!")
