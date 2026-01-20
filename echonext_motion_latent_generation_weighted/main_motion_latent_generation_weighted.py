#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
ECG-to-Motion Encoder Training Script with Preprocessed ECG (12×2500) + Morphology
Trains ResNet1D deterministic encoder to map ECG directly to 512-dim motion latents

Key features:
- Deterministic encoder (no VAE components)
- 3-layer MLP output head
- MSE loss only (no KL divergence)
- Improved logging (file + console + TensorBoard)

Author: Adapted from echonext_preprocess_motion_vae for Encoder
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import weighted versions of model and dataloader
from model_resnet1d_morphology_encoder_weighted import (
    ECGMotionEncoder,
    make_model,
    compute_sample_weights,
    compute_subgroup_metrics
)
from loader_ecg_preprocessed_weighted import MotionDataModulePreprocessed


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
    logger.info("ECG-to-Motion Encoder Training")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    # Flush immediately
    for handler in logger.handlers:
        handler.flush()

    return logger


def train_epoch(model, train_loader, optimizer, device, epoch, writer, logger, use_weighted_loss=False, compute_subgroups=False):
    """
    Train for one epoch with optional weighted loss and subgroup metrics.

    Args:
        model: ECGMotionEncoder model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        writer: TensorBoard writer
        logger: Logger instance
        use_weighted_loss: Whether to apply weighted loss based on clinical conditions
        compute_subgroups: Whether to compute subgroup metrics

    Returns:
        Tuple of (avg_loss_weighted, avg_loss_unweighted, avg_mae, subgroup_results)
    """
    model.train()
    total_loss_weighted = 0.0    # For backprop when weighted loss enabled
    total_loss_unweighted = 0.0  # Always compute for monitoring
    total_mae = 0.0
    num_batches = 0

    # Accumulation for subgroup metrics
    if compute_subgroups:
        all_predictions = []
        all_ground_truths = []
        all_clinical = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        # Forward pass
        out = model(batch)

        # Always compute unweighted loss for monitoring
        loss_unweighted = model.loss_function(out, sample_weights=None)

        # Compute sample weights if weighted loss is enabled
        if use_weighted_loss and "clinical" in batch:
            sample_weights = compute_sample_weights(batch["clinical"], model.weight_config)
            loss_weighted = model.loss_function(out, sample_weights=sample_weights)
        else:
            loss_weighted = loss_unweighted

        # Compute MAE (unweighted for monitoring)
        mae = model.compute_mae(out["prediction"], out["motion_gt"])

        # Check for NaN
        if torch.isnan(loss_weighted) or torch.isinf(loss_weighted):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}")
            continue

        # Backward pass uses weighted loss
        loss_weighted.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss_weighted += loss_weighted.item()
        total_loss_unweighted += loss_unweighted.item()
        total_mae += mae.item()
        num_batches += 1

        # Store for subgroup analysis
        if compute_subgroups and "clinical" in batch:
            all_predictions.append(out["prediction"].detach().cpu().numpy())
            all_ground_truths.append(out["motion_gt"].detach().cpu().numpy())
            all_clinical.append(batch["clinical"].detach().cpu().numpy())

        # Compute running averages
        avg_weighted_so_far = total_loss_weighted / num_batches
        avg_unweighted_so_far = total_loss_unweighted / num_batches
        avg_mae_so_far = total_mae / num_batches

        # Update progress bar
        if use_weighted_loss:
            pbar.set_postfix({
                'W_MSE': f'{loss_weighted.item():.4f}',
                'U_MSE': f'{loss_unweighted.item():.4f}',
                'MAE': f'{mae.item():.4f}',
                'Avg_W': f'{avg_weighted_so_far:.4f}',
                'Avg_U': f'{avg_unweighted_so_far:.4f}'
            })
        else:
            pbar.set_postfix({
                'Batch_MSE': f'{loss_unweighted.item():.4f}',
                'Batch_MAE': f'{mae.item():.4f}',
                'Avg_MSE': f'{avg_unweighted_so_far:.4f}',
                'Avg_MAE': f'{avg_mae_so_far:.4f}'
            })

        # Log to TensorBoard (batch level)
        if batch_idx % 10 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/Batch_Loss_Unweighted', loss_unweighted.item(), global_step)
            if use_weighted_loss:
                writer.add_scalar('Training/Batch_Loss_Weighted', loss_weighted.item(), global_step)
            writer.add_scalar('Training/Batch_MAE', mae.item(), global_step)

    avg_loss_weighted = total_loss_weighted / num_batches if num_batches > 0 else 0
    avg_loss_unweighted = total_loss_unweighted / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    # Compute subgroup metrics if enabled
    subgroup_results = None
    if compute_subgroups and all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truths = np.concatenate(all_ground_truths, axis=0)
        clinical = np.concatenate(all_clinical, axis=0)
        subgroup_results = compute_subgroup_metrics(
            predictions, ground_truths, clinical, {}
        )

    return avg_loss_weighted, avg_loss_unweighted, avg_mae, subgroup_results


def validate_epoch(model, val_loader, device, epoch, logger, compute_subgroups=False):
    """
    Validate for one epoch with optional subgroup metrics.

    Args:
        model: ECGMotionEncoder model
        val_loader: Validation data loader
        device: Device to use
        epoch: Current epoch number
        logger: Logger instance
        compute_subgroups: Whether to compute subgroup metrics

    Returns:
        Tuple of (avg_loss, avg_mae, subgroup_results)
    """
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    # Accumulation for subgroup metrics
    if compute_subgroups:
        all_predictions = []
        all_ground_truths = []
        all_clinical = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            out = model(batch)

            # Compute loss (unweighted)
            loss = model.loss_function(out)

            # Compute MAE
            mae = model.compute_mae(out["prediction"], out["motion_gt"])

            # Accumulate metrics
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

            # Store for subgroup analysis
            if compute_subgroups and "clinical" in batch:
                all_predictions.append(out["prediction"].cpu().numpy())
                all_ground_truths.append(out["motion_gt"].cpu().numpy())
                all_clinical.append(batch["clinical"].cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'Loss/MSE': f'{loss.item():.4f}',
                'MAE': f'{mae.item():.4f}'
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    # Compute subgroup metrics if enabled
    subgroup_results = None
    if compute_subgroups and all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truths = np.concatenate(all_ground_truths, axis=0)
        clinical = np.concatenate(all_clinical, axis=0)
        subgroup_results = compute_subgroup_metrics(
            predictions, ground_truths, clinical, {}
        )

    return avg_loss, avg_mae, subgroup_results


def test_epoch(model, test_loader, device, epoch, logger, compute_subgroups=False):
    """
    Test for one epoch with optional subgroup metrics.

    Args:
        model: ECGMotionEncoder model
        test_loader: Test data loader
        device: Device to use
        epoch: Current epoch number
        logger: Logger instance
        compute_subgroups: Whether to compute subgroup metrics

    Returns:
        Tuple of (avg_loss, avg_mae, subgroup_results)
    """
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    # Accumulation for subgroup metrics
    if compute_subgroups:
        all_predictions = []
        all_ground_truths = []
        all_clinical = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Epoch {epoch+1} [Test]')

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            out = model(batch)

            # Compute loss (unweighted)
            loss = model.loss_function(out)

            # Compute MAE
            mae = model.compute_mae(out["prediction"], out["motion_gt"])

            # Accumulate metrics
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

            # Store for subgroup analysis
            if compute_subgroups and "clinical" in batch:
                all_predictions.append(out["prediction"].cpu().numpy())
                all_ground_truths.append(out["motion_gt"].cpu().numpy())
                all_clinical.append(batch["clinical"].cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'Loss/MSE': f'{loss.item():.4f}',
                'MAE': f'{mae.item():.4f}'
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    # Compute subgroup metrics if enabled
    subgroup_results = None
    if compute_subgroups and all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truths = np.concatenate(all_ground_truths, axis=0)
        clinical = np.concatenate(all_clinical, axis=0)
        subgroup_results = compute_subgroup_metrics(
            predictions, ground_truths, clinical, {}
        )

    return avg_loss, avg_mae, subgroup_results


def log_subgroup_metrics(subgroup_results, prefix, logger, writer=None, epoch=None):
    """
    Log subgroup metrics to console and TensorBoard.

    Args:
        subgroup_results: Dictionary of subgroup metrics
        prefix: Prefix for logging (e.g., "Validation", "Test")
        logger: Logger instance
        writer: TensorBoard writer (optional)
        epoch: Current epoch number (optional, for TensorBoard)
    """
    logger.info(f"\n{prefix} Subgroup Metrics:")
    logger.info("="*80)
    logger.info(f"{'Subgroup':<20} {'Count':<10} {'MSE':<12} {'MAE':<12}")
    logger.info("-"*80)

    # Overall first
    overall = subgroup_results["overall"]
    logger.info(f"{'Overall':<20} {overall['count']:<10} {overall['mse']:<12.3f} {overall['mae']:<12.3f}")

    # Then subgroups
    subgroup_names = {
        "lvef_lt_50": "LVEF < 50%",
        "lvef_lt_45": "LVEF < 45%",
        "wt_gt_13": "WT > 13",
        "wt_gt_15": "WT > 15",
        "lvedvi_gt_75": "LVEDVi > 75",
        "lvedvi_gt_62": "LVEDVi > 62",
    }

    for key, display_name in subgroup_names.items():
        if key in subgroup_results:
            result = subgroup_results[key]
            count = result['count']

            # Format values with 3 decimal places or show N/A if NaN
            if np.isnan(result['mse']) or np.isnan(result['mae']):
                mse_str = "N/A"
                mae_str = "N/A"
                logger.info(f"{display_name:<20} {count:<10} {mse_str:<12} {mae_str:<12}")
            else:
                logger.info(f"{display_name:<20} {count:<10} {result['mse']:<12.3f} {result['mae']:<12.3f}")

    logger.info("="*80)

    # TensorBoard logging
    if writer is not None and epoch is not None:
        for key, result in subgroup_results.items():
            if not np.isnan(result['mse']):
                writer.add_scalar(f'{prefix}_Subgroups/{key}_MSE', result['mse'], epoch)
                writer.add_scalar(f'{prefix}_Subgroups/{key}_MAE', result['mae'], epoch)
                writer.add_scalar(f'{prefix}_Subgroups/{key}_Count', result['count'], epoch)


def train_model(model, train_loader, val_loader, test_loader, config, device, logger, checkpoint_period=1):
    """
    Main training loop for Encoder with optional test dataset.

    Args:
        model: ECGMotionEncoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional, can be None)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )

    # Setup TensorBoard logging
    tensorboard_dir = os.path.join(config["checkpoint_dir"], "tensorboard_logs")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Extract weighted loss and subgroup flags
    use_weighted_loss = config.get("use_weighted_loss", False)
    compute_subgroups = config.get("compute_subgroups", False)

    logger.info("="*80)
    logger.info(f"Starting Encoder training for {config['epochs']} epochs")
    logger.info("="*80)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader is not None:
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr']}")
    logger.info(f"Weight decay: {config.get('weight_decay', 0.0)}")
    logger.info(f"Filter size: {config.get('filter_size', 64)}")
    logger.info(f"Dropout: {config.get('dropout', 0.5)}")
    logger.info(f"Weighted loss: {use_weighted_loss}")
    logger.info(f"Compute subgroups: {compute_subgroups}")
    logger.info("="*80)

    # Force flush after training info
    for handler in logger.handlers:
        handler.flush()

    best_val_loss = float('inf')
    best_mae_loss = float('inf')
    best_epoch = 0

    for epoch in range(config["epochs"]):
        logger.info(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Train
        train_loss_weighted, train_loss_unweighted, train_mae, train_subgroups = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, logger, use_weighted_loss, compute_subgroups
        )

        # Validate
        val_loss, val_mae, val_subgroups = validate_epoch(
            model, val_loader, device, epoch, logger, compute_subgroups
        )

        # Test (if test_loader provided)
        test_loss, test_mae, test_subgroups = None, None, None
        if test_loader is not None:
            test_loss, test_mae, test_subgroups = test_epoch(
                model, test_loader, device, epoch, logger, compute_subgroups
            )

        # Update scheduler
        scheduler.step(val_loss)

        # Store metrics (always use unweighted for history - backward compatibility)
        model.train_losses.append(train_loss_unweighted)
        model.val_losses.append(val_loss)
        model.train_mae_losses.append(train_mae)
        model.val_mae_losses.append(val_mae)

        # Log to TensorBoard (epoch level)
        if use_weighted_loss:
            writer.add_scalar('Loss/Train_Weighted_Epoch', train_loss_weighted, epoch)
            writer.add_scalar('Loss/Train_Unweighted_Epoch', train_loss_unweighted, epoch)
        else:
            writer.add_scalar('Loss/Train_Epoch', train_loss_unweighted, epoch)

        writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
        writer.add_scalar('Metrics/Train_MAE', train_mae, epoch)
        writer.add_scalar('Metrics/Val_MAE', val_mae, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        if test_loss is not None:
            writer.add_scalar('Loss/Test_Epoch', test_loss, epoch)
            writer.add_scalar('Metrics/Test_MAE', test_mae, epoch)

        # Log metrics
        logger.info("="*80)
        logger.info(f"Epoch {epoch+1} Summary:")
        if use_weighted_loss:
            logger.info(f"  Train - Weighted MSE: {train_loss_weighted:.4f}, Unweighted MSE: {train_loss_unweighted:.4f}, MAE: {train_mae:.4f}")
        else:
            logger.info(f"  Train Loss (MSE): {train_loss_unweighted:.4f}, MAE: {train_mae:.4f}")
        logger.info(f"  Val Loss (MSE): {val_loss:.4f}, MAE: {val_mae:.4f}")
        if test_loss is not None:
            logger.info(f"  Test Loss (MSE): {test_loss:.4f}, MAE: {test_mae:.4f}")
        logger.info("="*80)

        # Log subgroup metrics for training
        if train_subgroups is not None:
            log_subgroup_metrics(train_subgroups, "Training", logger, writer, epoch)

        # Log subgroup metrics for validation
        if val_subgroups is not None:
            log_subgroup_metrics(val_subgroups, "Validation", logger, writer, epoch)

        # Log subgroup metrics for test
        if test_subgroups is not None:
            log_subgroup_metrics(test_subgroups, "Test", logger, writer, epoch)

        # Force flush to ensure logs are written
        for handler in logger.handlers:
            handler.flush()

        # Save best model
        if val_loss < model.best_val_loss:
            model.best_val_loss = val_loss
            model.best_val_mae = val_mae
            logger.info(f"\n✓ New best model! Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            logger.info("Saving checkpoint...")
            model.save_checkpoint(optimizer, epoch, val_loss, prefix="best_checkpoint")

            best_mae_loss = val_mae
            best_epoch = epoch
            best_val_loss = val_loss

            # Force flush after important checkpoint
            for handler in logger.handlers:
                handler.flush()

        logger.info(f"\nBest val: epoch {best_epoch} | Val Loss: {best_val_loss:.4f} | MAE: {best_mae_loss:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_period == 0:
            model.save_checkpoint(optimizer, epoch, val_loss, prefix=f"checkpoint_epoch_{epoch+1}")

        model.epoch += 1

    writer.close()
    logger.info("\n" + "="*80)
    logger.info("Encoder Training completed!")
    logger.info(f"Best validation loss: {model.best_val_loss:.4f}")
    logger.info(f"Best validation MAE: {model.best_val_mae:.4f}")
    logger.info("="*80 + "\n")

    # Final flush to ensure all logs are written
    for handler in logger.handlers:
        handler.flush()

    return model


def main(args):
    """
    Main function for training ECG-to-Motion Encoder.

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

    # Configuration (removed beta)
    config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "filter_size": args.filter_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "checkpoint_dir": checkpoint_dir,
        "epochs": args.epochs,
    }

    logger.info("="*80)
    logger.info("ENCODER TRAINING MODE")
    logger.info("="*80)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Force flush after configuration
    for handler in logger.handlers:
        handler.flush()

    # Create dataset with clinical columns and test dataset
    dataset = MotionDataModulePreprocessed(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        test_csv_path=args.test_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        load_clinical_cols=True,
    )
    dataset.setup(stage="fit")

    # Configure weight config for model
    weight_config = {
        'use_lvef': args.use_weighted_loss,
        'lvef_threshold': args.weight_lvef_threshold,
        'lvef_weight': args.weight_lvef_value,
        'use_wt': args.use_weighted_loss,
        'wt_threshold': args.weight_wt_threshold,
        'wt_weight': args.weight_wt_value,
        'use_lvedvi': args.use_weighted_loss,
        'lvedvi_threshold': args.weight_lvedvi_threshold,
        'lvedvi_weight': args.weight_lvedvi_value,
    }

    # Create Encoder model with weight config
    model = make_model(config, weight_config=weight_config)
    model.to(device)

    # Get data loaders
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader() if args.test_csv else None

    # Update config with new flags
    config["use_weighted_loss"] = args.use_weighted_loss
    config["compute_subgroups"] = args.compute_subgroups

    logger.info(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Force flush after model info
    for handler in logger.handlers:
        handler.flush()

    # Train the model
    model = train_model(model, train_loader, val_loader, test_loader, config, device, logger)

    # Save results table
    logger.info("\nSaving results table...")
    results_df = model.results_table()
    results_path = os.path.join(checkpoint_dir, "training_history.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    # Final flush before exit
    for handler in logger.handlers:
        handler.flush()

    return None


if __name__ == "__main__":
    # data and time to string YYYYMMDD_HHMMSS
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    print("Current date and time:", dt_string)

    parser = argparse.ArgumentParser(description="ECG-to-Motion Encoder Training with Preprocessed ECG (12×2500) + Morphology")

    # Data paths
    parser.add_argument(
        "--train_csv",
        type=str,
        # default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/all_lvef_filtered.csv", # use everything for training
        default="ukb/jz_meshheart/echonext_motion_latents_16Jan26/echonext_splits/combined_wt_volume_ecg_motion.csv",
        help="Path to training motion latent CSV file.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="ukb/jz_meshheart/echonext_motion_latents_16Jan26/echonext_splits/val_wt_volume_ecg_motion.csv",
        help="Path to validation motion latent CSV file.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="ukb/jz_meshheart/echonext_motion_latents_16Jan26/echonext_splits/test_wt_volume_ecg_motion.csv",
        help="Path to test CSV file (optional).",
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
        default="jzheng12/Codes/ECG_MeshHeart/output/echonext_motion_latent_generation_weighted/checkpoints_" + dt_string,
        help="Checkpoint directory.",
    )

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=5e-5")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default=16")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs, default=500")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers, default=0")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2), default=0")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay (L2 regularization), default=0.01")

    # Model parameters
    parser.add_argument("--filter_size", type=int, default=64, help="ResNet filter size, default=16")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate, default=0.5")

    # Weighted loss arguments
    parser.add_argument("--use_weighted_loss", action="store_true", default=True,
                       help="Enable weighted loss during training")
    
    parser.add_argument("--weight_lvef_threshold", type=float, default=50.0,
                       help="LVEF threshold for weighting (default: 50)")
    parser.add_argument("--weight_lvef_value", type=float, default=10.0,
                       help="Weight value for LVEF condition (default: 10)")
    
    parser.add_argument("--weight_wt_threshold", type=float, default=13.0,
                       help="Wall thickness threshold for weighting (default: 15)")
    parser.add_argument("--weight_wt_value", type=float, default=5.0,
                       help="Weight value for WT condition (default: 4)")
    
    parser.add_argument("--weight_lvedvi_threshold", type=float, default=62.0,
                       help="LVEDVi threshold for weighting (default: 62)")
    parser.add_argument("--weight_lvedvi_value", type=float, default=10.0,
                       help="Weight value for LVEDVi condition (default: 10)")

    # Subgroup metrics arguments
    parser.add_argument("--compute_subgroups", action="store_true", default=True,
                       help="Enable subgroup metrics computation")

    # Note: --beta argument removed (no longer needed for deterministic encoder)

    args = parser.parse_args()
    main(args)
