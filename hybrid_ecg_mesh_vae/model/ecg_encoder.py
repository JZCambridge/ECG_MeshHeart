"""
ECG Encoder for ECG-to-Mesh Generation VAE

Adapted from EchoNext ResNet1D encoder to output mu and logvar for VAE latent space.
Takes 12-lead ECG + demographics + morphology as input and outputs latent distribution parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BasicBlock1d(nn.Module):
    """1D Basic Block for ResNet architecture"""
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
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
    """
    ResNet1D backbone that processes ECG signal and fuses with tabular features.
    """

    def __init__(
        self,
        len_tabular_feature_vector: int,
        filter_size: int = 64,
        input_channels: int = 12,
        dropout_value: float = 0.5,
        num_classes: int = 64,
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
        self, block: type, planes: int, blocks: int, stride: int = 1
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


class ECGEncoder(nn.Module):
    """
    ECG Encoder that outputs mu and logvar for VAE latent distribution.

    Takes 12-lead ECG + demographics + morphology as input and produces
    parameters for a latent distribution (mu, logvar) that can be sampled
    via reparameterization trick.

    Input:
        - ecg_raw: [B, 12, 2500] - 12-lead ECG signals
        - demographics: [B, 9] - Patient demographics
        - morphology: [B, 16] - ECG morphology features

    Output:
        - mu: [B, latent_dim] - Mean of latent distribution
        - logvar: [B, latent_dim] - Log variance of latent distribution
    """

    def __init__(
        self,
        latent_dim: int = 64,
        filter_size: int = 64,
        input_channels: int = 12,
        dropout_value: float = 0.5,
        conv1_kernel_size: int = 15,
        conv1_stride: int = 2,
        padding: int = 7,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ResNet1D backbone that outputs 2*latent_dim (for mu and logvar)
        self.resnet = ResNet1dWithTabular(
            len_tabular_feature_vector=25,  # 9 demographics + 16 morphology
            filter_size=filter_size,
            input_channels=input_channels,
            dropout_value=dropout_value,
            num_classes=latent_dim * 2,  # Output both mu and logvar
            conv1_kernel_size=conv1_kernel_size,
            conv1_stride=conv1_stride,
            padding=padding,
        )

    def forward(
        self, ecg_raw: torch.Tensor, demographics: torch.Tensor, morphology: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ECG encoder.

        Args:
            ecg_raw: [B, 12, 2500] - ECG signals
            demographics: [B, 9] - Demographics
            morphology: [B, 16] - Morphology features

        Returns:
            mu: [B, latent_dim] - Mean of latent distribution
            logvar: [B, latent_dim] - Log variance of latent distribution
        """
        # Concatenate demographics and morphology → [B, 25]
        tabular = torch.cat([demographics, morphology], dim=1)

        # Reshape ECG to match expected input: [B, 1, 2500, 12]
        ecg_reshaped = ecg_raw.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, 2500, 12]

        # Forward through ResNet
        out = self.resnet((ecg_reshaped, tabular))  # [B, latent_dim * 2]

        # Split output into mu and logvar
        mu, logvar = out.chunk(2, dim=1)  # Each: [B, latent_dim]

        return mu, logvar


if __name__ == "__main__":
    # Test ECG Encoder
    print("=== Testing ECG Encoder ===")

    latent_dim = 64
    batch_size = 8

    encoder = ECGEncoder(latent_dim=latent_dim)
    print(f"Encoder created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test forward pass
    dummy_ecg = torch.randn(batch_size, 12, 2500)
    dummy_demographics = torch.randn(batch_size, 9)
    dummy_morphology = torch.randn(batch_size, 16)

    mu, logvar = encoder(dummy_ecg, dummy_demographics, dummy_morphology)

    print(f"\nForward pass test:")
    print(f"  Input ECG shape: {dummy_ecg.shape}")
    print(f"  Input demographics shape: {dummy_demographics.shape}")
    print(f"  Input morphology shape: {dummy_morphology.shape}")
    print(f"  Output mu shape: {mu.shape}")
    print(f"  Output logvar shape: {logvar.shape}")

    print("\n✓ ECG Encoder test completed!")
