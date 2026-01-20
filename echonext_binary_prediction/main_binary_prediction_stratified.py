#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
ECG Binary Disease Prediction Training Script - SEX-STRATIFIED VERSION
Trains ResNet1D classifier to predict disease status from ECG signals
with sex-based filtering for separate male/female models.

Key features:
- Binary classification (0=healthy, 1=diseased)
- Sex stratification (filters by Sex column before training)
- BCE loss with logits for numerical stability
- Saves prediction probabilities for bootstrap analysis
- Comprehensive logging (file + console + TensorBoard)

Author: Adapted from main_binary_prediction.py
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import model and dataloader
from model_resnet1d_binary_classifier import ECGBinaryClassifier, make_model
from loader_ecg_binary import BinaryDataModule


def filter_csv_by_column(
    csv_path: str,
    filter_column: str,
    filter_value: int,
    label_column: str,
    logger: logging.Logger
) -> Tuple[str, Dict[str, int]]:
    """
    Filter CSV by column value and return path to filtered CSV.

    Args:
        csv_path: Original CSV path
        filter_column: Column name to filter by (e.g., "Sex")
        filter_value: Value to filter for (e.g., 0 or 1)
        label_column: Name of label column for class distribution
        logger: Logger instance

    Returns:
        (filtered_csv_path, statistics_dict)
    """
    logger.info(f"\nFiltering {csv_path}...")

    # Load CSV
    df = pd.read_csv(csv_path)
    original_count = len(df)
    logger.info(f"  Original samples: {original_count}")

    # Check if filter column exists
    if filter_column not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(
            f"Filter column '{filter_column}' not found in CSV.\n"
            f"Available columns: {available_cols}"
        )

    # Handle NaN values in filter column
    if df[filter_column].isna().any():
        nan_count = df[filter_column].isna().sum()
        logger.warning(f"  Found {nan_count} NaN values in '{filter_column}'. Excluding from filtered dataset.")
        df = df.dropna(subset=[filter_column])
        logger.info(f"  After removing NaN: {len(df)} samples")

    # Filter by column value
    df_filtered = df[df[filter_column] == filter_value].copy()
    filtered_count = len(df_filtered)

    # Check if filtered dataset is empty
    if filtered_count == 0:
        unique_values = sorted(df[filter_column].unique())
        raise ValueError(
            f"No samples found with {filter_column}={filter_value}.\n"
            f"Unique values in '{filter_column}': {unique_values}\n"
            f"Hint: For Sex column, use 0 for female and 1 for male."
        )

    logger.info(f"  Filtered samples: {filtered_count} ({100*filtered_count/original_count:.1f}%)")

    # Get filtered sex distribution
    sex_dist_filtered = df_filtered[filter_column].value_counts().to_dict()
    logger.info(f"  Sex distribution in filtered data:")
    for sex_val, count in sex_dist_filtered.items():
        logger.info(f"    {filter_column}={sex_val}: {count} ({100*count/filtered_count:.1f}%)")

    # Note about class balance
    if label_column in df_filtered.columns:
        label_count = df_filtered[label_column].notna().sum()
        logger.info(f"  Samples with label data after filtering: {label_count}")
        logger.info(f"  (Class balance will be computed after threshold application during dataset creation)")

    # Create temp filtered CSV path
    base_dir = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path).replace('.csv', '')
    filtered_csv_path = os.path.join(base_dir, f"{base_name}_filtered_{filter_column}{filter_value}.csv")

    # Save filtered CSV
    df_filtered.to_csv(filtered_csv_path, index=False)
    logger.info(f"  Saved filtered CSV to: {filtered_csv_path}")

    # Return statistics
    stats = {
        'original_count': original_count,
        'filtered_count': filtered_count,
    }

    return filtered_csv_path, stats


def setup_logging(checkpoint_dir: str):
    """
    Setup logging with file and console handlers.

    Args:
        checkpoint_dir: Directory where log files will be saved

    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    log_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(log_dir, 'training.log')

    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler with explicit flushing
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Force immediate flush after each log
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

    logger.info("="*80)
    logger.info("ECG Binary Disease Prediction Training")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    # Flush immediately
    for handler in logger.handlers:
        handler.flush()

    return logger


def compute_confusion_matrix_metrics(labels, probs, threshold=0.5):
    """
    Compute TPR, FNR, FPR, TNR from predictions.

    Args:
        labels: True binary labels (0 or 1)
        probs: Predicted probabilities [0, 1]
        threshold: Classification threshold (default 0.5)

    Returns:
        dict with tpr, fnr, fpr, tnr
    """
    preds = (probs >= threshold).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    # Sensitivity/TPR = TP / (TP + FN)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity/TNR = TN / (TN + FP)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # False Negative Rate = FN / (TP + FN)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # False Positive Rate = FP / (TN + FP)
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tpr": tpr,  # Sensitivity
        "fnr": fnr,
        "fpr": fpr,
        "tnr": tnr,  # Specificity
    }


def train_epoch(model, train_loader, optimizer, device, epoch, writer, logger):
    """
    Train for one epoch with binary classification.

    Args:
        model: ECGBinaryClassifier model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        writer: TensorBoard writer
        logger: Logger instance

    Returns:
        avg_loss, avg_accuracy, avg_f1, auroc, auprc
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    num_batches = 0

    # For AUROC/AUPRC calculation
    all_probs = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        # Forward pass
        out = model(batch)

        # Compute loss
        loss = model.loss_function(out)

        # Compute metrics
        metrics = model.compute_metrics(out["logits"], out["labels"])

        # For AUROC/AUPRC
        probs = torch.sigmoid(out["logits"].squeeze(-1)).detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}")
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_accuracy += metrics["accuracy"]
        total_f1 += metrics["f1"]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{metrics["accuracy"]:.4f}',
            'F1': f'{metrics["f1"]:.4f}'
        })

        # Log to TensorBoard (batch level)
        if batch_idx % 10 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Training/Batch_Accuracy', metrics["accuracy"], global_step)
            writer.add_scalar('Training/Batch_F1', metrics["f1"], global_step)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0

    # Calculate AUROC and AUPRC
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    auprc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

    # Calculate confusion matrix metrics
    cm_metrics = compute_confusion_matrix_metrics(all_labels, all_probs, threshold=0.5)

    return avg_loss, avg_accuracy, avg_f1, auroc, auprc, cm_metrics


def validate_epoch(model, val_loader, device, epoch, logger):
    """
    Validate for one epoch.

    Args:
        model: ECGBinaryClassifier model
        val_loader: Validation data loader
        device: Device to use
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        avg_loss, avg_accuracy, avg_f1, auroc, auprc
    """
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    num_batches = 0

    # For AUROC/AUPRC calculation
    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            out = model(batch)

            # Compute loss
            loss = model.loss_function(out)

            # Compute metrics
            metrics = model.compute_metrics(out["logits"], out["labels"])

            # For AUROC/AUPRC
            probs = torch.sigmoid(out["logits"].squeeze(-1)).cpu().numpy()
            labels = out["labels"].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += metrics["accuracy"]
            total_f1 += metrics["f1"]
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{metrics["accuracy"]:.4f}',
                'F1': f'{metrics["f1"]:.4f}'
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0

    # Calculate AUROC and AUPRC
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    auprc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

    # Calculate confusion matrix metrics
    cm_metrics = compute_confusion_matrix_metrics(all_labels, all_probs, threshold=0.5)

    return avg_loss, avg_accuracy, avg_f1, auroc, auprc, cm_metrics


def save_predictions(model, data_loader, dataset_name, device, output_dir, logger, suffix=""):
    """
    Generate and save predictions to CSV for bootstrap analysis.

    Args:
        model: Trained model
        data_loader: DataLoader (train or val)
        dataset_name: "train" or "val"
        device: Device to use
        output_dir: Directory to save CSV
        logger: Logger instance
        suffix: Optional suffix for filename (e.g., "_trainauc", "_valauc")

    Returns:
        Path to saved CSV file
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Generating predictions for {dataset_name} set...")
    logger.info(f"{'='*80}")

    model.eval()
    all_eids = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Saving {dataset_name} predictions"):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            out = model(batch)

            # Get probabilities
            probs = torch.sigmoid(out["logits"].squeeze(-1))

            # Store results
            all_eids.extend(batch["eid"].cpu().numpy())
            all_labels.extend(out["labels"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Create DataFrame
    predictions_df = pd.DataFrame({
        'eid': all_eids,
        'true_label': all_labels,
        'predicted_probability': all_probs
    })

    # Save to CSV
    output_path = os.path.join(output_dir, f"{dataset_name}_predictions{suffix}.csv")
    predictions_df.to_csv(output_path, index=False)

    logger.info(f"✅ Saved {len(predictions_df)} predictions to: {output_path}")
    logger.info(f"  Class distribution: {np.sum(all_labels == 1):.0f} diseased, {np.sum(all_labels == 0):.0f} healthy")
    logger.info(f"  Probability range: [{np.min(all_probs):.4f}, {np.max(all_probs):.4f}]")

    return output_path


def train_model(model, train_loader, val_loader, config, device, logger):
    """
    Main training loop for Binary Classifier.

    Args:
        model: ECGBinaryClassifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to use
        logger: Logger instance

    Returns:
        Trained model
    """
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4)
    )

    # Setup scheduler based on best_metric choice
    best_metric = config.get("best_metric", "loss")
    if best_metric == "loss":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
        )
    elif best_metric == "auc":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True
        )

    # Setup TensorBoard logging
    tensorboard_dir = os.path.join(config["checkpoint_dir"], "tensorboard_logs")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    logger.info("="*80)
    logger.info(f"Starting Binary Classification training for {config['epochs']} epochs")
    logger.info("="*80)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr']}")
    logger.info(f"Weight decay: {config.get('weight_decay', 0.0)}")
    logger.info(f"Filter size: {config.get('filter_size', 64)}")
    logger.info(f"Dropout: {config.get('dropout', 0.1)}")
    logger.info(f"Best metric: {best_metric}")
    logger.info(f"Threshold direction: {config.get('threshold_direction', 'less_than')}")
    logger.info("="*80)

    # Force flush after training info
    for handler in logger.handlers:
        handler.flush()

    # Tracking variables - depends on best_metric choice
    if best_metric == "loss":
        # Backward compatible: track only validation loss
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        best_epoch = 0
        logger.info("Using LOSS-BASED best model selection (backward compatible)\n")

    elif best_metric == "auc":
        # Dual AUC tracking: independent best models for train and val AUC
        best_train_auroc = 0.0
        best_train_auroc_epoch = 0
        best_val_auroc = 0.0
        best_val_auroc_epoch = 0
        logger.info("Using AUC-BASED best model selection (dual tracking: train AUC + val AUC)\n")

    for epoch in range(config["epochs"]):
        logger.info(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Train
        train_loss, train_acc, train_f1, train_auroc, train_auprc, train_cm = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, logger
        )

        # Validate
        val_loss, val_acc, val_f1, val_auroc, val_auprc, val_cm = validate_epoch(
            model, val_loader, device, epoch, logger
        )

        # Update scheduler (aligned with best_metric)
        if best_metric == "loss":
            scheduler.step(val_loss)
        elif best_metric == "auc":
            scheduler.step(val_auroc)

        # Store metrics in model
        model.train_losses.append(train_loss)
        model.val_losses.append(val_loss)
        model.train_accuracies.append(train_acc)
        model.val_accuracies.append(val_acc)
        model.train_f1_scores.append(train_f1)
        model.val_f1_scores.append(val_f1)

        # Store AUROC separately (used when best_metric='auc')
        if best_metric == "auc":
            model.train_aurocs.append(train_auroc)
            model.val_aurocs.append(val_auroc)

        # Log to TensorBoard (epoch level)
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
        writer.add_scalar('Metrics/Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Metrics/Val_Accuracy', val_acc, epoch)
        writer.add_scalar('Metrics/Train_AUROC', train_auroc, epoch)
        writer.add_scalar('Metrics/Val_AUROC', val_auroc, epoch)

        # Log confusion matrix metrics to TensorBoard
        writer.add_scalar('Metrics/Train_TPR', train_cm['tpr'], epoch)
        writer.add_scalar('Metrics/Train_FNR', train_cm['fnr'], epoch)
        writer.add_scalar('Metrics/Train_FPR', train_cm['fpr'], epoch)
        writer.add_scalar('Metrics/Train_TNR', train_cm['tnr'], epoch)
        writer.add_scalar('Metrics/Val_TPR', val_cm['tpr'], epoch)
        writer.add_scalar('Metrics/Val_FNR', val_cm['fnr'], epoch)
        writer.add_scalar('Metrics/Val_FPR', val_cm['fpr'], epoch)
        writer.add_scalar('Metrics/Val_TNR', val_cm['tnr'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log metrics (using .3f format)
        logger.info("="*80)
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.3f}, Acc: {train_acc:.3f}, AUROC: {train_auroc:.3f}")
        logger.info(f"  Train TPR: {train_cm['tpr']:.3f}, FNR: {train_cm['fnr']:.3f}, "
                   f"FPR: {train_cm['fpr']:.3f}, TNR: {train_cm['tnr']:.3f}")
        logger.info(f"  Val Loss: {val_loss:.3f}, Acc: {val_acc:.3f}, AUROC: {val_auroc:.3f}")
        logger.info(f"  Val TPR: {val_cm['tpr']:.3f}, FNR: {val_cm['fnr']:.3f}, "
                   f"FPR: {val_cm['fpr']:.3f}, TNR: {val_cm['tnr']:.3f}")
        logger.info("="*80)

        # Force flush to ensure logs are written
        for handler in logger.handlers:
            handler.flush()

        # Best model saving logic - branches based on best_metric
        if best_metric == "loss":
            # BACKWARD COMPATIBLE: Save best model based on validation loss
            if val_loss < model.best_val_loss:
                model.best_val_loss = val_loss
                logger.info(f"\n✓ New best model! Val Loss: {val_loss:.3f}")
                logger.info("Saving checkpoint...")
                model.save_checkpoint(optimizer, epoch, val_loss, prefix="best_checkpoint")

                best_val_loss = val_loss
                best_epoch = epoch

                # Generate predictions for best model (overwrite previous)
                logger.info("\n--- Generating predictions for best model (loss-based) ---")
                train_pred_path = save_predictions(
                    model, train_loader, "train", device, config["checkpoint_dir"], logger, suffix=""
                )
                val_pred_path = save_predictions(
                    model, val_loader, "val", device, config["checkpoint_dir"], logger, suffix=""
                )
                logger.info(f"✓ Best model predictions saved")
                logger.info(f"  Train: {train_pred_path}")
                logger.info(f"  Val: {val_pred_path}")

                # Force flush after important checkpoint
                for handler in logger.handlers:
                    handler.flush()

            logger.info(f"\nCurrent best: epoch {best_epoch+1} | Val Loss: {best_val_loss:.3f}")

        elif best_metric == "auc":
            # DUAL AUC TRACKING: Independent tracking for train AUC and val AUC

            # Check if train AUC improved
            if train_auroc > best_train_auroc:
                best_train_auroc = train_auroc
                best_train_auroc_epoch = epoch
                model.best_train_auroc = train_auroc

                logger.info(f"\n✓ New BEST TRAIN AUC! Train AUROC: {train_auroc:.3f}")
                logger.info("Saving checkpoint for best train AUC...")
                model.save_checkpoint(optimizer, epoch, val_loss, prefix="best_checkpoint_trainauc")

                # Generate predictions for best train AUC model
                logger.info("\n--- Generating predictions for best TRAIN AUC model ---")
                train_pred_path = save_predictions(
                    model, train_loader, "train", device, config["checkpoint_dir"], logger, suffix="_trainauc"
                )
                val_pred_path = save_predictions(
                    model, val_loader, "val", device, config["checkpoint_dir"], logger, suffix="_trainauc"
                )
                logger.info(f"✓ Best train AUC model predictions saved")
                logger.info(f"  Train: {train_pred_path}")
                logger.info(f"  Val: {val_pred_path}")

                # Force flush
                for handler in logger.handlers:
                    handler.flush()

            # Check if val AUC improved (independent of train AUC)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_auroc_epoch = epoch
                model.best_val_auroc = val_auroc

                logger.info(f"\n✓ New BEST VAL AUC! Val AUROC: {val_auroc:.3f}")
                logger.info("Saving checkpoint for best val AUC...")
                model.save_checkpoint(optimizer, epoch, val_loss, prefix="best_checkpoint_valauc")

                # Generate predictions for best val AUC model
                logger.info("\n--- Generating predictions for best VAL AUC model ---")
                train_pred_path = save_predictions(
                    model, train_loader, "train", device, config["checkpoint_dir"], logger, suffix="_valauc"
                )
                val_pred_path = save_predictions(
                    model, val_loader, "val", device, config["checkpoint_dir"], logger, suffix="_valauc"
                )
                logger.info(f"✓ Best val AUC model predictions saved")
                logger.info(f"  Train: {train_pred_path}")
                logger.info(f"  Val: {val_pred_path}")

                # Force flush
                for handler in logger.handlers:
                    handler.flush()

            # Log current best values
            logger.info(f"\nCurrent bests: Train AUC={best_train_auroc:.3f} (epoch {best_train_auroc_epoch+1}), "
                       f"Val AUC={best_val_auroc:.3f} (epoch {best_val_auroc_epoch+1})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            model.save_checkpoint(optimizer, epoch, val_loss, prefix=f"checkpoint_epoch_{epoch+1}")

        model.epoch += 1

    writer.close()
    logger.info("\n" + "="*80)
    logger.info("Binary Classification Training completed!")

    if best_metric == "loss":
        logger.info(f"Best validation loss: {model.best_val_loss:.3f}")
    elif best_metric == "auc":
        logger.info(f"Best train AUROC: {model.best_train_auroc:.3f} (epoch {best_train_auroc_epoch+1})")
        logger.info(f"Best val AUROC: {model.best_val_auroc:.3f} (epoch {best_val_auroc_epoch+1})")

    logger.info("="*80 + "\n")

    # Final flush to ensure all logs are written
    for handler in logger.handlers:
        handler.flush()

    return model


def main(args):
    """
    Main function for training ECG Binary Classifier.

    Args:
        args: Command-line arguments

    Returns:
        None
    """
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(checkpoint_dir)
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Setup device with GPU ID
    if torch.cuda.is_available() and args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using device: {device} (GPU {args.gpu_id})")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device} (default GPU)")
    else:
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")

    # Filter data by sex
    logger.info("\n" + "="*80)
    logger.info("FILTERING DATA BY SEX")
    logger.info("="*80)
    logger.info(f"Filter column: {args.filter_column}")
    logger.info(f"Filter value: {args.filter_value}")

    # Filter training set
    logger.info("\n--- Training Set ---")
    train_csv_filtered, train_stats = filter_csv_by_column(
        args.train_csv, args.filter_column, args.filter_value,
        args.label_column, logger
    )

    # Filter validation set
    logger.info("\n--- Validation Set ---")
    val_csv_filtered, val_stats = filter_csv_by_column(
        args.val_csv, args.filter_column, args.filter_value,
        args.label_column, logger
    )

    logger.info("\n" + "="*80)
    logger.info("FILTERING COMPLETED")
    logger.info("="*80)
    logger.info(f"Training set: {train_stats['filtered_count']} samples ({args.filter_column}={args.filter_value})")
    logger.info(f"Validation set: {val_stats['filtered_count']} samples ({args.filter_column}={args.filter_value})")
    logger.info("="*80)

    # Force flush after filtering
    for handler in logger.handlers:
        handler.flush()

    # Configuration
    config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "filter_size": args.filter_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "checkpoint_dir": checkpoint_dir,
        "epochs": args.epochs,
        "best_metric": args.best_metric,
        "threshold_direction": args.threshold_direction,
        "filter_column": args.filter_column,
        "filter_value": args.filter_value,
    }

    logger.info("="*80)
    logger.info("BINARY DISEASE PREDICTION TRAINING")
    logger.info("="*80)
    logger.info(f"Label column: {args.label_column}")
    if args.threshold is not None:
        direction_str = "< threshold" if args.threshold_direction == "less_than" else "> threshold"
        logger.info(f"Threshold: {args.threshold} (values {direction_str} → diseased)")
    else:
        logger.info("Threshold: None (using binary labels directly)")
    logger.info(f"Filter: {args.filter_column}={args.filter_value}")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Force flush after configuration
    for handler in logger.handlers:
        handler.flush()

    # Create dataset
    dataset = BinaryDataModule(
        train_csv_path=train_csv_filtered,
        val_csv_path=val_csv_filtered,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=args.label_column,
        threshold=args.threshold,
        threshold_direction=args.threshold_direction,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
    )
    dataset.setup(stage="fit")

    # Create Binary Classifier model
    model = make_model(config)
    model.to(device)

    # Get data loaders
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    logger.info(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Force flush after model info
    for handler in logger.handlers:
        handler.flush()

    # Train the model
    model = train_model(model, train_loader, val_loader, config, device, logger)

    # Save results table
    logger.info("\nSaving results table...")
    results_df = model.results_table()
    results_path = os.path.join(checkpoint_dir, "training_history.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"✅ Results saved to {results_path}")

    # Predictions already saved during training (when best model was found)
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    if args.best_metric == "loss":
        logger.info(f"\nPredictions from best model (loss-based) saved to:")
        logger.info(f"  Train: {checkpoint_dir}/train_predictions.csv")
        logger.info(f"  Val: {checkpoint_dir}/val_predictions.csv")
        logger.info(f"\nTo analyze predictions, run:")
        logger.info(f"  python analyze_predictions.py --predictions {checkpoint_dir}/val_predictions.csv")
    elif args.best_metric == "auc":
        logger.info(f"\nPredictions from TWO best models saved:")
        logger.info(f"\n1. Best Train AUC model:")
        logger.info(f"  Train: {checkpoint_dir}/train_predictions_trainauc.csv")
        logger.info(f"  Val: {checkpoint_dir}/val_predictions_trainauc.csv")
        logger.info(f"\n2. Best Val AUC model:")
        logger.info(f"  Train: {checkpoint_dir}/train_predictions_valauc.csv")
        logger.info(f"  Val: {checkpoint_dir}/val_predictions_valauc.csv")
        logger.info(f"\nTo analyze predictions, run:")
        logger.info(f"  python analyze_predictions.py --predictions {checkpoint_dir}/val_predictions_valauc.csv")

    logger.info("="*80)

    # Final flush before exit
    for handler in logger.handlers:
        handler.flush()

    return None


if __name__ == "__main__":
    # Date and time to string YYYYMMDD_HHMMSS
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    print("Current date and time:", dt_string)

    parser = argparse.ArgumentParser(
        description="ECG Binary Disease Prediction Training"
    )

    # Data paths
    parser.add_argument(
        "--train_csv",
        type=str,
        # default="ukb/jz_ecg/DM_measurements/diseased_wt_volume/batches_matched/matched_first_20260106.csv",
        # default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/train_ecg_motion_templ.csv",
        default="ukb/jz_ecg/DM_measurements/processed_wt_volume/batches/wt_volume_first_20260106.csv",
        help="Path to training CSV file with binary disease labels.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        # default="ukb/jz_ecg/DM_measurements/diseased_wt_volume/batches_matched/matched_second_20260106.csv",
        # default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/test_ecg_motion_templ.csv",
        default="ukb/jz_ecg/DM_measurements/processed_wt_volume/batches/wt_volume_second_20260106.csv",
        help="Path to validation CSV file with binary disease labels.",
    )
    parser.add_argument(
        "--preprocessed_ecg",
        type=str,
        default="ukb/jz_ecg/ecg_echonext_15Oct25/preprocessed_ecg_12x2500_v1_15Oct25_parallel.pt",
        help="Path to preprocessed ECG .pt file (12×2500).",
    )
    parser.add_argument(
        "--ecg_phenotypes_path",
        type=str,
        default="cardiac/pi514/ukbb_ecg/Final/2_Factor_ECG/data/pt_data_ecg/ecg_phenotypes.csv",
        help="Path to ECG morphology phenotypes CSV file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="jzheng12/Codes/ECG_MeshHeart/output/echonext_binary_prediction/",
        help="Checkpoint directory.",
    )

    # Label configuration
    parser.add_argument(
        "--label_column",
        type=str,
        default="LVEDVi_ml/m2", #'wt_max', #'lv_ef_fr0_percent', #"diseased",
        help="Name of label column in CSV (can be binary or continuous).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=74, #61, #None,
        help="Threshold for converting continuous variable to binary (values < threshold → diseased=1). "
             "If None, expects binary labels (0 or 1) directly. Example: --threshold 50 for LVEF.",
    )
    parser.add_argument(
        "--threshold_direction",
        type=str,
        default="greater_than",
        choices=["less_than", "greater_than"],
        help="Direction for threshold comparison. 'less_than': values < threshold → diseased "
             "(default, for LVEF). 'greater_than': values > threshold → diseased (for WT_MAX).",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="auc",
        choices=["loss", "auc"],
        help="Metric to use for selecting best model. 'loss': min validation loss (default, backward compatible). "
             "'auc': separate tracking for best train AUC and best val AUC.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="allbatches_",
        help="Prefix for output files.",
    )

    # Training parameters
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate, EchoNext default 5e-5")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size, EchoNext default 16")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization), EchoNext default 0.01")

    # Model parameters
    parser.add_argument("--filter_size", type=int, default=16, help="ResNet filter size, EchoNext default 16")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate, EchoNext default 0.5")

    # Sex stratification arguments
    parser.add_argument(
        "--filter_column",
        type=str,
        default="Sex",
        help="Column name to filter by for stratified training (default: 'Sex')"
    )
    parser.add_argument(
        "--filter_value",
        type=int,
        default=1, #0,
        help="Value to filter for (default: 0 for female). Use 0=female, 1=male."
    )

    args = parser.parse_args()

    args.checkpoint_dir = (
        args.checkpoint_dir +
        args.output_prefix +
        args.label_column.replace('/', '_') + '_' +
        args.threshold_direction + str(args.threshold) + '_' +
        args.filter_column + str(args.filter_value) + '_' +
        dt_string
    )

    main(args)
