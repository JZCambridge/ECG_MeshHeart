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

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and dataloader
from model_resnet1d_morphology_encoder import ECGMotionEncoder, make_model
from loader_ecg_preprocessed import MotionDataModulePreprocessed


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


def train_epoch(model, train_loader, optimizer, device, epoch, writer, logger):
    """
    Train for one epoch.

    Args:
        model: ECGMotionEncoder model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        writer: TensorBoard writer
        logger: Logger instance

    Returns:
        Average training loss (MSE), MAE
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        # Forward pass
        out = model(batch)

        # Compute loss (MSE only)
        loss = model.loss_function(out)

        # Compute MAE (for observation)
        mae = model.compute_mae(out["prediction"], out["motion_gt"])

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
        total_mae += mae.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'Loss/MSE': f'{loss.item():.4f}',
            'MAE': f'{mae.item():.4f}'
        })

        # Log to TensorBoard (batch level)
        if batch_idx % 10 == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Training/Batch_MAE', mae.item(), global_step)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    return avg_loss, avg_mae


def validate_epoch(model, val_loader, device, epoch, logger):
    """
    Validate for one epoch.

    Args:
        model: ECGMotionEncoder model
        val_loader: Validation data loader
        device: Device to use
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        Tuple of (avg_loss, avg_mae)
    """
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

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

            # Compute MAE
            mae = model.compute_mae(out["prediction"], out["motion_gt"])

            # Accumulate metrics
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss/MSE': f'{loss.item():.4f}',
                'MAE': f'{mae.item():.4f}'
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    return avg_loss, avg_mae


def train_model(model, train_loader, val_loader, config, device, logger):
    """
    Main training loop for Encoder.

    Args:
        model: ECGMotionEncoder model
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )

    # Setup TensorBoard logging
    tensorboard_dir = os.path.join(config["checkpoint_dir"], "tensorboard_logs")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    logger.info("="*80)
    logger.info(f"Starting Encoder training for {config['epochs']} epochs")
    logger.info("="*80)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr']}")
    logger.info(f"Weight decay: {config.get('weight_decay', 0.0)}")
    logger.info(f"Filter size: {config.get('filter_size', 64)}")
    logger.info(f"Dropout: {config.get('dropout', 0.5)}")
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
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, logger
        )

        # Validate
        val_loss, val_mae = validate_epoch(
            model, val_loader, device, epoch, logger
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Store metrics
        model.train_losses.append(train_loss)
        model.val_losses.append(val_loss)
        model.train_mae_losses.append(train_mae)
        model.val_mae_losses.append(val_mae)

        # Log to TensorBoard (epoch level)
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
        writer.add_scalar('Metrics/Train_MAE', train_mae, epoch)
        writer.add_scalar('Metrics/Val_MAE', val_mae, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log metrics
        logger.info("="*80)
        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Train Loss (MSE): {train_loss:.4f}, MAE: {train_mae:.4f}")
        logger.info(f"  Val Loss (MSE): {val_loss:.4f}, MAE: {val_mae:.4f}")
        logger.info("="*80)

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
        if (epoch + 1) % 10 == 0:
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

    # Create dataset
    dataset = MotionDataModulePreprocessed(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
    )
    dataset.setup(stage="fit")

    # Create Encoder model
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
        default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/train_ecg_motion_templ.csv",
        help="Path to training motion latent CSV file.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/test_ecg_motion_templ.csv", #900
        # default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/val_ecg_motion_templ.csv", #90 cases
        help="Path to validation motion latent CSV file.",
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
        default="jzheng12/Codes/ECG_MeshHeart/output/echonext_motion_latent_generation/checkpoints_" + dt_string,
        help="Checkpoint directory.",
    )

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2)")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay (L2 regularization)")

    # Model parameters
    parser.add_argument("--filter_size", type=int, default=64, help="ResNet filter size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Note: --beta argument removed (no longer needed for deterministic encoder)

    args = parser.parse_args()
    main(args)
