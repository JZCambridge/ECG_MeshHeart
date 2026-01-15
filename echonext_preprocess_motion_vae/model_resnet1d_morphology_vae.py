"""
ResNet1D-based ECG-to-Motion VAE Model with Morphology Features
Uses ResNet1dWithTabular to encode ECG to (mu, logvar) for VAE latent distribution
Compatible with hybrid_example encoder architecture
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
    """ResNet1D backbone that outputs mu and logvar for VAE"""

    def __init__(
        self,
        len_tabular_feature_vector: int,
        filter_size: int = 64,
        input_channels: int = 12,
        dropout_value: float = 0.5,
        num_classes: int = 1024,  # 1024 = 512 (mu) + 512 (logvar)
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


class ECGMotionEncoderVAE(nn.Module):
    """
    ECG-to-Motion VAE Encoder using ResNet1D + Tabular data.
    Encodes 12-lead ECG + demographics + morphology → (mu, logvar) for 512-dim latent space.

    Compatible with hybrid_example encoder architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        beta: float = 0.01,  # KL divergence weight
        checkpoint_dir: Optional[str] = None,
        num_epochs: int = 100,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.beta = beta
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs

        # Training metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_correlations = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []
        self.train_mae_losses = []  # Track training MAE
        self.val_mae_losses = []    # Track validation MAE

        self.best_val_loss = float("inf")
        self.best_val_corr = 0.0
        self.best_val_mae = float("inf")
        self.epoch = 0

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: [B, 512] - mean of latent distribution
            logvar: [B, 512] - log variance of latent distribution

        Returns:
            z: [B, 512] - sampled latent code
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: Dict):
        """
        Forward pass through VAE encoder.

        Args:
            batch: Dictionary containing 'ecg_raw' [B, 12, 2500], 'demographics' [B, 9],
                   'ecg_morphology' [B, 16], and 'motion_latent' [B, 512]

        Returns:
            Dictionary with mu, logvar, z (sampled latent), and ground truth
        """
        ecg_raw = batch["ecg_raw"]  # [B, 12, 2500]
        demographics = batch["demographics"]  # [B, 9]
        ecg_morphology = batch["ecg_morphology"]  # [B, 16]

        # Concatenate demographics and morphology → [B, 25]
        tabular = torch.cat([demographics, ecg_morphology], dim=1)

        # Reshape ECG to match expected input: [B, 1, 2500, 12]
        ecg_reshaped = ecg_raw.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, 2500, 12]

        # Forward through model → [B, 1024]
        out = self.model((ecg_reshaped, tabular))

        # Split into mu and logvar
        mu, logvar = out.chunk(2, dim=1)  # Each: [B, 512]

        # Sample latent via reparameterization
        z = self.reparameterize(mu, logvar)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "motion_gt": batch["motion_latent"],
        }

    def loss_function(self, out: Dict):
        """
        Compute VAE loss: reconstruction (MSE) + beta * KL divergence.

        Args:
            out: Output dictionary from forward pass

        Returns:
            total_loss, recon_loss, kl_loss
        """
        z = out["z"]
        motion_gt = out["motion_gt"]
        mu = out["mu"]
        logvar = out["logvar"]

        # Reconstruction loss (MSE between sampled z and ground truth motion)
        recon_loss = F.mse_loss(z, motion_gt, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)  # Normalize by batch size

        # Total VAE loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def compute_mean_correlation(self, z: torch.Tensor, motion_gt: torch.Tensor):
        """
        Compute mean Pearson correlation across all 512 dimensions.

        Args:
            z: [B, 512] - sampled latent
            motion_gt: [B, 512] - ground truth motion latents

        Returns:
            mean_corr: scalar - mean Pearson correlation
        """
        correlations = []

        for dim in range(z.shape[1]):  # Iterate over 512 dimensions
            z_dim = z[:, dim]
            gt_dim = motion_gt[:, dim]

            # Pearson correlation
            z_centered = z_dim - z_dim.mean()
            gt_centered = gt_dim - gt_dim.mean()

            numerator = (z_centered * gt_centered).sum()
            denominator = torch.sqrt((z_centered ** 2).sum() * (gt_centered ** 2).sum())

            if denominator > 0:
                corr = numerator / denominator
                correlations.append(corr)

        if len(correlations) > 0:
            mean_corr = torch.stack(correlations).mean()
        else:
            mean_corr = torch.tensor(0.0)

        return mean_corr

    def compute_mae(self, z: torch.Tensor, motion_gt: torch.Tensor):
        """
        Compute Mean Absolute Error between sampled latent and ground truth (for observation only).

        Args:
            z: [B, 512] - sampled latent
            motion_gt: [B, 512] - ground truth motion latents

        Returns:
            mae: scalar - mean absolute error
        """
        mae = F.l1_loss(z, motion_gt, reduction="mean")
        return mae

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
            'beta': self.beta,
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
            "train_recon_loss": [x.item() if torch.is_tensor(x) else x for x in self.train_recon_losses],
            "train_kl_loss": [x.item() if torch.is_tensor(x) else x for x in self.train_kl_losses],
            "val_recon_loss": [x.item() if torch.is_tensor(x) else x for x in self.val_recon_losses],
            "val_kl_loss": [x.item() if torch.is_tensor(x) else x for x in self.val_kl_losses],
            "train_mae": [x.item() if torch.is_tensor(x) else x for x in self.train_mae_losses],
            "val_mae": [x.item() if torch.is_tensor(x) else x for x in self.val_mae_losses],
        })

        return df


def make_model(config: Dict):
    """
    Create ResNet1D-based ECG-to-Motion VAE encoder model.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        ECGMotionEncoderVAE model
    """
    # Create ResNet1D model with 25 tabular features and 128 output dims
    resnet_model = ResNet1dWithTabular(
        len_tabular_feature_vector=25,  # 9 demographics + 16 morphology
        filter_size=config.get("filter_size", 64),
        input_channels=12,  # 12 ECG leads
        dropout_value=config.get("dropout", 0.5),
        num_classes=1024,  # 1024 = 512 (mu) + 512 (logvar)
        conv1_kernel_size=config.get("conv1_kernel_size", 15),
        conv1_stride=config.get("conv1_stride", 2),
        padding=config.get("padding", 7),
    )

    # Create full VAE model wrapper
    model = ECGMotionEncoderVAE(
        model=resnet_model,
        lr=config["lr"],
        beta=config.get("beta", 0.01),
        checkpoint_dir=config["checkpoint_dir"],
        num_epochs=config["epochs"],
    )

    return model


if __name__ == "__main__":
    # Test VAE model creation
    print("=== Testing ResNet1D ECGMotionEncoderVAE with Morphology ===")

    config = {
        "filter_size": 64,
        "dropout": 0.5,
        "lr": 1e-3,
        "beta": 0.01,
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
        "demographics": torch.randn(batch_size, 9),
        "ecg_morphology": torch.randn(batch_size, 16),
        "motion_latent": torch.randn(batch_size, 512),
    }

    out = model(dummy_batch)
    print(f"\nForward pass test:")
    print(f"  Input ECG shape: {dummy_batch['ecg_raw'].shape}")
    print(f"  Input demographics shape: {dummy_batch['demographics'].shape}")
    print(f"  Input morphology shape: {dummy_batch['ecg_morphology'].shape}")
    print(f"  Output mu shape: {out['mu'].shape}")
    print(f"  Output logvar shape: {out['logvar'].shape}")
    print(f"  Output z (sampled) shape: {out['z'].shape}")

    total_loss, recon_loss, kl_loss = model.loss_function(out)
    print(f"\nLoss computation test:")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL divergence loss: {kl_loss.item():.4f}")
    print(f"  Total VAE loss: {total_loss.item():.4f}")

    corr = model.compute_mean_correlation(out['z'], out['motion_gt'])
    print(f"  Mean correlation (z vs gt): {corr.item():.4f}")

    print("\n✓ VAE Model test completed!")
