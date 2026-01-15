#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
ECG-to-Motion VAE Training Script with Preprocessed ECG (12×2500) + Morphology
Trains ResNet1D VAE encoder to learn latent distribution (mu, logvar) from ECG

Compatible with hybrid_example encoder architecture.

Author: Adapted from echonext_preprocess_motion for VAE
"""

import argparse
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
from model_resnet1d_morphology_vae import ECGMotionEncoderVAE, make_model
from loader_ecg_preprocessed import MotionDataModulePreprocessed


def train_epoch(model, train_loader, optimizer, device, epoch, writer):
    """
    Train for one epoch.

    Args:
        model: ECGMotionEncoderVAE model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        writer: TensorBoard writer

    Returns:
        Average training loss, reconstruction loss, KL loss, MAE
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
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

        # Compute VAE loss
        loss, recon_loss, kl_loss = model.loss_function(out)

        # Compute MAE (for observation only, not part of loss)
        mae = model.compute_mae(out["z"], out["motion_gt"])

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at batch {batch_idx}")
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_mae += mae.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon/MSE': f'{recon_loss.item():.4f}',
            'KL': f'{kl_loss.item():.4f}',
            'MAE': f'{mae.item():.4f}'
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    return avg_loss, avg_recon_loss, avg_kl_loss, avg_mae


def validate_epoch(model, val_loader, device, epoch):
    """
    Validate for one epoch.

    Args:
        model: ECGMotionEncoderVAE model
        val_loader: Validation data loader
        device: Device to use
        epoch: Current epoch number

    Returns:
        Tuple of (avg_loss, avg_recon_loss, avg_kl_loss, avg_correlation, avg_mae)
    """
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_corr = 0.0
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

            # Compute VAE loss
            loss, recon_loss, kl_loss = model.loss_function(out)

            # Compute correlation
            corr = model.compute_mean_correlation(out["z"], out["motion_gt"])

            # Compute MAE
            mae = model.compute_mae(out["z"], out["motion_gt"])

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_corr += corr.item()
            total_mae += mae.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon/MSE': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Corr': f'{corr.item():.4f}',
                'MAE': f'{mae.item():.4f}'
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0
    avg_corr = total_corr / num_batches if num_batches > 0 else 0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0

    return avg_loss, avg_recon_loss, avg_kl_loss, avg_corr, avg_mae


def train_model(model, train_loader, val_loader, config, device):
    """
    Main training loop for VAE.

    Args:
        model: ECGMotionEncoderVAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to use

    Returns:
        Trained model
    """
    # Setup optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )

    # Setup TensorBoard logging
    writer = SummaryWriter(log_dir=os.path.join(config["checkpoint_dir"], "tensorboard_logs"))

    print(f"\n{'='*60}")
    print(f"Starting VAE training for {config['epochs']} epochs")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Beta (KL weight): {config.get('beta', 0.01)}")
    print(f"Weight decay: {config.get('weight_decay', 0.0)}")
    print(f"Filter size: {config.get('filter_size', 64)}")
    print(f"Dropout: {config.get('dropout', 0.5)}")
    print(f"{'='*60}\n")

    # store best validation loss, epoch, correlation, reconstruction loss, kl loss
    best_val_loss = float('inf')
    best_rec_loss = float('inf')
    best_kl_loss = float('inf')
    best_mae_loss = float('inf')
    best_epoch = 0
    
    
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Train
        train_loss, train_recon, train_kl, train_mae = train_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_recon, val_kl, val_corr, val_mae = validate_epoch(
            model, val_loader, device, epoch
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Store epoch metrics
        model.train_losses.append(train_loss)
        model.val_losses.append(val_loss)
        model.val_correlations.append(val_corr)
        model.train_recon_losses.append(train_recon)
        model.train_kl_losses.append(train_kl)
        model.val_recon_losses.append(val_recon)
        model.val_kl_losses.append(val_kl)
        model.train_mae_losses.append(train_mae)
        model.val_mae_losses.append(val_mae)

        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Loss/Train_Recon', train_recon, epoch)
        writer.add_scalar('Loss/Train_KL', train_kl, epoch)
        writer.add_scalar('Loss/Val_Recon', val_recon, epoch)
        writer.add_scalar('Loss/Val_KL', val_kl, epoch)
        writer.add_scalar('Metrics/Val_Correlation', val_corr, epoch)
        writer.add_scalar('Metrics/Train_MAE', train_mae, epoch)
        writer.add_scalar('Metrics/Val_MAE', val_mae, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.3f} (Recon/MSE: {train_recon:.3f}, KL: {train_kl:.3f}, MAE: {train_mae:.3f})")
        print(f"  Val Loss: {val_loss:.3f} (Recon/MSE: {val_recon:.3f}, KL: {val_kl:.3f}, MAE: {val_mae:.3f})")
        print(f"  Val Correlation: {val_corr:.3f}")
        print(f"{'='*60}")

        # Save best model
        if val_loss < model.best_val_loss:
            model.best_val_loss = val_loss
            model.best_val_corr = val_corr
            model.best_val_mae = val_mae
            print(f"\n✓ New best model! Val Loss: {val_loss:.3f}, Val Corr: {val_corr:.3f}, Val MAE: {val_mae:.3f}")
            print("Saving checkpoint...")
            model.save_checkpoint(optimizer, epoch, val_loss, val_corr, prefix="best_checkpoint")

            # Update best epoch
            best_rec_loss = val_recon
            best_kl_loss = val_kl
            best_mae_loss = val_mae
            best_epoch = epoch
            best_val_loss = val_loss

        # print best epoch info
        print(f"\nBest val: epoch {best_epoch} | Val Loss: {best_val_loss:.3f} | Recon/MSE: {best_rec_loss:.3f} | KL: {best_kl_loss:.3f} | MAE: {best_mae_loss:.3f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            model.save_checkpoint(optimizer, epoch, val_loss, val_corr, prefix=f"checkpoint_intermediate")

        model.epoch += 1

        print(f'Best Val Loss so far: {model.best_val_loss:.3f} at epoch {model.epoch}')

    writer.close()
    print("\n" + "="*60)
    print("VAE Training completed!")
    print(f"Best validation loss: {model.best_val_loss:.4f}")
    print(f"Best validation correlation: {model.best_val_corr:.4f}")
    print(f"Best validation MAE: {model.best_val_mae:.4f}")
    print("="*60 + "\n")

    return model


def main(args):
    """
    Main function for training ECG-to-Motion VAE encoder.

    Args:
        args: Command-line arguments

    Returns:
        None
    """
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Setup device with GPU ID
    if torch.cuda.is_available() and args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
        print(f"Using device: {device} (GPU {args.gpu_id})")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (default GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # Configuration
    config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "filter_size": args.filter_size,
        "dropout": args.dropout,
        "beta": args.beta,
        "weight_decay": args.weight_decay,
        "checkpoint_dir": checkpoint_dir,
        "epochs": args.epochs,
    }

    print("\n" + "="*60)
    print("VAE TRAINING MODE")
    print("="*60 + "\n")

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

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

    # Create VAE model
    model = make_model(config)
    model.to(device)

    # Get data loaders
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train the model
    model = train_model(model, train_loader, val_loader, config, device)

    # Save results table
    print("\nSaving results table...")
    results_df = model.results_table()
    results_df.to_csv(os.path.join(checkpoint_dir, "training_history.csv"), index=False)
    print(f"Results saved to {os.path.join(checkpoint_dir, 'training_history.csv')}")

    return None


if __name__ == "__main__":
    # data and time to string YYYYMMDD_HHMMSS
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    print("Current date and time:", dt_string)

    parser = argparse.ArgumentParser(description="ECG-to-Motion VAE Training with Preprocessed ECG (12×2500) + Morphology")

    # Data paths
    parser.add_argument(
        "--train_csv",
        type=str,
        default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/train_ecg_motion_templ.csv",
        help="Path to training motion latent CSV file.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt512_ecg1024_71k/diseased_matched/lvef/all_lvef/val_ecg_motion_templ.csv",
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
        default="jzheng12/Codes/ECG_MeshHeart/output/echonext_preprocess_motion_vae_512/checkpoints_" + dt_string,
        help="Checkpoint directory.",
    )

    # Training parameters
    parser.add_argument("--lr", type=float, 
                        # default=1e-3, 
                        default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2)")
    parser.add_argument("--weight_decay", type=float, 
                        default=1e-3, 
                        help="Weight decay (L2 regularization)")

    # Model parameters
    parser.add_argument("--filter_size", type=int, default=64, help="ResNet filter size")
    
    parser.add_argument("--dropout", type=float, 
                        default=0.1, 
                        # default=0.5, 
                        help="Dropout rate")

    # VAE parameters
    parser.add_argument("--beta", type=float, default=1e-4, help="KL divergence weight (beta-VAE parameter)")

    args = parser.parse_args()
    main(args)
