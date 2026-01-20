"""
ResNet1D-based ECG Binary Disease Classifier with Morphology Features
Modified from echonext_motion_latent_generation for binary classification
Deterministic encoder with single output neuron for disease prediction
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
    """ResNet1D backbone with 3-layer MLP output head for binary classification"""

    def __init__(
        self,
        len_tabular_feature_vector: int,
        filter_size: int = 64,
        input_channels: int = 12,
        dropout_value: float = 0.5,
        num_classes: int = 1,  # Single output for binary classification
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

        # 3-layer MLP output head: [concat_dim] → [1024] → [1024] → [1] (binary logit)
        intermediate_dim = 8 * filter_size * BasicBlock1d.expansion * 2 + len_tabular_feature_vector
        self.output_mlp = nn.Sequential(
            nn.Linear(intermediate_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value),
            nn.Linear(1024, 1)  # Single logit output for binary classification
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


class ECGBinaryClassifier(nn.Module):
    """
    ECG Binary Disease Classifier using ResNet1D + Tabular data.
    Encodes 12-lead ECG + demographics + morphology → single disease probability (0=healthy, 1=diseased).

    Key features:
    - Deterministic encoder (no variational components)
    - 3-layer MLP output head with single output neuron
    - BCE loss with logits for numerical stability
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

        # Training metrics storage for classification
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []

        # AUROC tracking (used when best_metric='auc')
        self.train_aurocs = []
        self.val_aurocs = []

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.best_train_auroc = 0.0
        self.best_val_auroc = 0.0
        self.epoch = 0

    def forward(self, batch: Dict):
        """
        Forward pass through classifier.

        Args:
            batch: Dictionary containing 'ecg_raw' [B, 12, 2500], 'demographics' [B, 8],
                   'ecg_morphology' [B, 16], and 'disease_label' [B] (binary: 0 or 1)

        Returns:
            Dictionary with logits and ground truth labels
        """
        ecg_raw = batch["ecg_raw"]  # [B, 12, 2500]
        demographics = batch["demographics"]  # [B, 8]
        ecg_morphology = batch["ecg_morphology"]  # [B, 16]

        # Concatenate demographics and morphology → [B, 24]
        tabular = torch.cat([demographics, ecg_morphology], dim=1)

        # Reshape ECG to match expected input: [B, 1, 2500, 12]
        ecg_reshaped = ecg_raw.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, 2500, 12]

        # Forward through ResNet + MLP → [B, 1]
        prediction = self.model((ecg_reshaped, tabular))

        return {
            "logits": prediction,
            "labels": batch["disease_label"],
        }

    def loss_function(self, out: Dict):
        """
        Compute binary cross-entropy loss with logits.

        Args:
            out: Output dictionary from forward pass

        Returns:
            loss: BCE loss (scalar tensor)
        """
        logits = out["logits"]  # [B, 1]
        labels = out["labels"]  # [B]

        # BCE with logits (numerically stable - combines sigmoid + BCE)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            labels.float(),
            reduction="mean"
        )

        return loss

    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute classification metrics (accuracy, precision, recall, F1).

        Args:
            logits: [B, 1] - raw logits
            labels: [B] - true labels (0 or 1)

        Returns:
            dict with accuracy, precision, recall, f1
        """
        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs >= 0.5).long()
        labels = labels.long()

        # Basic metrics
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def save_checkpoint(self, optimizer, epoch, val_loss, prefix="checkpoint"):
        """
        Save model checkpoint.

        Args:
            optimizer: Optimizer state
            epoch: Current epoch
            val_loss: Validation loss
            prefix: Checkpoint filename prefix (e.g., "best_checkpoint_trainauc")
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'best_train_auroc': self.best_train_auroc,
            'best_val_auroc': self.best_val_auroc,
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
            DataFrame with classification metrics
        """
        results = {
            "train_loss": [x.item() if torch.is_tensor(x) else x for x in self.train_losses],
            "val_loss": [x.item() if torch.is_tensor(x) else x for x in self.val_losses],
            "train_accuracy": [x.item() if torch.is_tensor(x) else x for x in self.train_accuracies],
            "val_accuracy": [x.item() if torch.is_tensor(x) else x for x in self.val_accuracies],
            "train_f1": [x.item() if torch.is_tensor(x) else x for x in self.train_f1_scores],
            "val_f1": [x.item() if torch.is_tensor(x) else x for x in self.val_f1_scores],
        }

        # Include AUROC if tracked
        if len(self.train_aurocs) > 0:
            results["train_auroc"] = [x.item() if torch.is_tensor(x) else x for x in self.train_aurocs]
            results["val_auroc"] = [x.item() if torch.is_tensor(x) else x for x in self.val_aurocs]

        df = pd.DataFrame.from_dict(results)
        return df


def make_model(config: Dict):
    """
    Create ResNet1D-based ECG Binary Classifier model.

    Args:
        config: Dictionary with model hyperparameters

    Returns:
        ECGBinaryClassifier model
    """
    # Create ResNet1D model with 24 tabular features and single output
    resnet_model = ResNet1dWithTabular(
        len_tabular_feature_vector=24,  # 8 demographics + 16 morphology
        filter_size=config.get("filter_size", 64),
        input_channels=12,  # 12 ECG leads
        dropout_value=config.get("dropout", 0.5),
        num_classes=1,  # Single output for binary classification
        conv1_kernel_size=config.get("conv1_kernel_size", 15),
        conv1_stride=config.get("conv1_stride", 2),
        padding=config.get("padding", 7),
    )

    # Create full classifier model wrapper
    model = ECGBinaryClassifier(
        model=resnet_model,
        lr=config["lr"],
        checkpoint_dir=config["checkpoint_dir"],
        num_epochs=config["epochs"],
    )

    return model


if __name__ == "__main__":
    # Test Binary Classifier model creation
    print("=== Testing ResNet1D ECGBinaryClassifier with Morphology ===")

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
        "demographics": torch.randn(batch_size, 9),
        "ecg_morphology": torch.randn(batch_size, 16),
        "disease_label": torch.randint(0, 2, (batch_size,)).float(),
    }

    out = model(dummy_batch)
    print(f"\nForward pass test:")
    print(f"  Input ECG shape: {dummy_batch['ecg_raw'].shape}")
    print(f"  Input demographics shape: {dummy_batch['demographics'].shape}")
    print(f"  Input morphology shape: {dummy_batch['ecg_morphology'].shape}")
    print(f"  Output logits shape: {out['logits'].shape}")
    print(f"  Labels shape: {out['labels'].shape}")

    loss = model.loss_function(out)
    print(f"\nLoss computation test:")
    print(f"  BCE loss: {loss.item():.4f}")

    metrics = model.compute_metrics(out['logits'], out['labels'])
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    print("\n✓ Binary Classifier Model test completed!")
