#!/usr/bin/env python
"""
Main Training Script for Hybrid ECG-to-Mesh VAE

This script trains the hybrid model that:
1. Encodes ECG signals + patient info to latent distribution (mu, logvar)
2. Samples from latent via reparameterization trick
3. Decodes latent to generate 50-frame cardiac mesh sequences

Adapted from MeshHeart's main_pure_vaev0.py with:
- Hybrid dataloader for ECG + mesh
- Combined ECG encoder + mesh decoder
- Pretrained checkpoint initialization
- Reused loss functions (VAECELoss with Chamfer + KL + smoothness)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import datetime
import time
import logging

# Import hybrid components
# from data.hybrid_dataloader import HybridECGMeshDataset, HybridDataModule
# multi worker
from data.hybrid_dataloader import HybridDataModule
# from data.hybrid_dataloader_optimized import OptimizedHybridECGMeshDataset as HybridECGMeshDataset, OptimizedHybridDataModule as HybridDataModule
from model.hybrid_vae import HybridECGMeshVAE, initialize_from_pretrained
from model.config_hybrid import load_config, validate_config
import loss.loss as Loss
import util.utils as util
from torch.utils.tensorboard import SummaryWriter

# Import metrics from MeshHeart
from loss.metrics import (
    compute_mesh_metrics_for_sequence,
    compute_batch_metrics,
    EpochMetricsAccumulator,
    print_metrics_summary
)

def setup_logging(log_dir:str):
    """Setup logging configuration"""
    log_file = os.path.join(log_dir, 'training.log')

    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Get root logger and configure it to capture all module loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

def save_model(model, optimizer, epoch, train_loss, val_loss, checkpoint_name, model_type="best", logger=None):
    """
    Save model checkpoint with metadata.

    Args:
        model: HybridECGMeshVAE model
        optimizer: Optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        checkpoint_name: Path to save checkpoint
        model_type: Type of checkpoint ('best' or 'intermediate')
        logger: Logger instance (optional)
    """
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch_num': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_type': model_type,
        'save_time': datetime.datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_name)

    msg = f"{model_type.capitalize()} model saved: {checkpoint_name} | Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
    if logger:
        logger.info(msg)
    else:
        print(f"✅ {msg}")


def warmup_lr_chunk_based(optimizer, current_chunk, warmup_chunks, base_lr):
    """
    Linear warmup for learning rate based on chunk count.

    Args:
        optimizer: Optimizer instance
        current_chunk: Current global chunk step (0-indexed)
        warmup_chunks: Total number of chunks for warmup
        base_lr: Target learning rate after warmup

    Returns:
        Current learning rate or None if not in warmup phase
    """
    if current_chunk < warmup_chunks:
        lr = base_lr * (current_chunk + 1) / warmup_chunks
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    return None


def compute_batch_mesh_metrics(v_out, v_gt, patient_ids):
    """
    Compute mesh metrics for a batch.

    Args:
        v_out: Predicted mesh vertices [B, seq_len, points, 3]
        v_gt: Ground truth mesh vertices [B, seq_len, points, 3]
        patient_ids: Patient IDs for tracking

    Returns:
        hd_values: List of Hausdorff distances
        assd_values: List of ASSD values
    """
    batch_size = v_out.shape[0]
    hd_values = []
    assd_values = []

    for b in range(batch_size):
        try:
            pred_seq = v_out[b]  # [seq_len, num_vertices, 3]
            gt_seq = v_gt[b]     # [seq_len, num_vertices, 3]

            avg_hd, avg_assd = compute_mesh_metrics_for_sequence(pred_seq, gt_seq)

            hd_values.append(avg_hd.item())
            assd_values.append(avg_assd.item())

        except Exception as e:
            print(f"Warning: Error computing metrics for patient {patient_ids[b] if patient_ids else b}: {e}")
            continue

    return hd_values, assd_values


def train(model, trainloader, optimizer, device, config, writer, epoch, global_chunk_step, logger=None):
    """
    Training function for hybrid ECG-to-Mesh VAE with gradient accumulation.

    Args:
        model: HybridECGMeshVAE
        trainloader: Training data loader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        config: Configuration object
        writer: TensorBoard writer
        epoch: Current epoch number
        global_chunk_step: Global chunk counter for chunk-based logging
        logger: Logger instance (optional)

    Returns:
        avg_loss: Average training loss
        epoch_metrics: Dictionary with mesh metrics
    """
    model.train()
    avg_loss = []
    batch_losses = []

    # Initialize metrics accumulator
    metrics_accumulator = EpochMetricsAccumulator()

    # Gradient accumulation setup
    accumulation_steps = config.accumulation_steps
    optimizer.zero_grad()  # Zero gradients at the start

    # for idx, data in enumerate(trainloader):
    # use tqdm for progress bar with fixed bar size
    pbar = tqdm(trainloader, desc=f"Epoch {epoch} Training")#, ncols=60)
    for idx, data in enumerate(pbar):
        try:
            # Unpack hybrid data
            demographics = data['demographics'].to(device)
            ecg_raw = data['ecg_raw'].to(device)
            ecg_morphology = data['ecg_morphology'].to(device)
            heart_v = data['heart_v'].to(device)
            heart_f = data['heart_f'].to(device)
            heart_e = data['heart_e'].to(device)
            subid = data['eid']

            # Forward pass: ECG + patient info → latent → mesh
            v_out, mu, logvar = model(ecg_raw, demographics, ecg_morphology)

            # Loss computation (reuse VAECELoss from MeshHeart)
            loss, loss_recon = Loss.VAECELoss(
                v_out, heart_v, heart_f, logvar, mu,
                beta=config.beta,
                lambd=config.lambd,
                lambd_s=config.lambd_s,
                loss=config.loss
            )

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            avg_loss.append(loss.item() * accumulation_steps)  # Store unscaled loss for logging
            batch_losses.append(loss.item() * accumulation_steps)

            # Compute mesh metrics for this batch (conditionally based on config)
            # Speed optimization: These metrics are expensive (HD/ASSD with KNN)
            # Set compute_train_metrics_freq=0 to disable and speed up training by 3-5x
            hd_values, assd_values = [], []
            compute_metrics = False
            if hasattr(config, 'compute_train_metrics_freq') and config.compute_train_metrics_freq > 0:
                if config.compute_train_metrics_freq == -1:  # Every batch
                    compute_metrics = True
                elif idx % config.compute_train_metrics_freq == 0:  # Every N batches
                    compute_metrics = True

            if compute_metrics:
                try:
                    with torch.no_grad():
                        hd_values, assd_values = compute_batch_mesh_metrics(v_out, heart_v, subid)
                        metrics_accumulator.update_raw_values(hd_values, assd_values)
                except Exception as e:
                    msg = f"Warning: Could not compute mesh metrics for batch {idx}: {e}"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)

            # Backward pass (accumulate gradients)
            loss.backward()

            # Update weights every accumulation_steps
            if (idx + 1) % accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                if hasattr(config, 'grad_clip_value') and config.grad_clip_value:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_value)

                optimizer.step()
                optimizer.zero_grad()

            # Log batch-level metrics occasionally (chunk-based step counter)
            if idx % 10 == 0 and writer is not None:
                batch_step = global_chunk_step * len(trainloader) + idx
                writer.add_scalar('Batch/Train_Loss', loss.item(), batch_step)
                writer.add_scalar('Batch/Train_Loss_Recon', loss_recon.item(), batch_step)

                if hd_values and assd_values:
                    batch_hd = np.mean(hd_values)
                    batch_assd = np.mean(assd_values)
                    writer.add_scalar('Batch/Train_HD', batch_hd, batch_step)
                    writer.add_scalar('Batch/Train_ASSD', batch_assd, batch_step)
            
            # tqdm add infor for current loss and batch average loss with f 3 decimal
            pbar.set_postfix({'Loss': f'{loss.item() * accumulation_steps:.3f}', 'Loss_recon': f'{loss_recon.item() * accumulation_steps:.3f}', 'Batch Avg Loss': f'{np.mean(batch_losses):.3f}'})

        except Exception as e:
            msg = f"Error in training batch {idx}: {e}"
            if logger:
                logger.error(msg)
                import traceback
                logger.error(traceback.format_exc())
            else:
                print(msg)
                import traceback
                traceback.print_exc()
            continue

    # Final optimizer step if there are remaining accumulated gradients
    if len(trainloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = np.mean(avg_loss) if avg_loss else float('inf')
    epoch_metrics = metrics_accumulator.compute_epoch_stats()

    msg = f'Epoch {epoch}, Loss: {epoch_loss:.6f}'
    if logger:
        logger.info(msg)
    else:
        print(msg)

    # Only print metrics if they were computed
    if epoch_metrics['num_samples'] > 0:
        print_metrics_summary(epoch_metrics, "  Train ", logger=logger)
    else:
        msg = "  Train metrics disabled (set compute_train_metrics_freq > 0 to enable)"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Enhanced tensorboard logging
    if writer is not None:
        writer.add_scalar('Train_Loss_Std', np.std(batch_losses), epoch)
        writer.add_scalar('Train_Batches_Processed', len(batch_losses), epoch)
        # Only log metrics if they were computed
        if epoch_metrics['num_samples'] > 0:
            writer.add_scalar('Train_HD_Mean', epoch_metrics['hd_mean'], epoch)
            writer.add_scalar('Train_HD_Median', epoch_metrics['hd_median'], epoch)
            writer.add_scalar('Train_ASSD_Mean', epoch_metrics['assd_mean'], epoch)
            writer.add_scalar('Train_ASSD_Median', epoch_metrics['assd_median'], epoch)

    return epoch_loss, epoch_metrics


def val(model, validloader, device, config, writer, global_chunk_step, logger=None):
    """
    Validation function for hybrid ECG-to-Mesh VAE.

    Args:
        model: HybridECGMeshVAE
        validloader: Validation data loader
        device: Device (cuda/cpu)
        config: Configuration object
        writer: TensorBoard writer
        global_chunk_step: Global chunk counter for chunk-based logging
        logger: Logger instance (optional)

    Returns:
        val_loss: Average validation loss
        epoch_metrics: Dictionary with mesh metrics
    """
    if logger:
        logger.info('-------------validation--------------')
    else:
        print('-------------validation--------------')
    model.eval()

    metrics_accumulator = EpochMetricsAccumulator()

    with torch.no_grad():
        valid_error = []
        for idx, data in enumerate(validloader):
            try:
                # Unpack hybrid data
                demographics = data['demographics'].to(device)
                ecg_raw = data['ecg_raw'].to(device)
                ecg_morphology = data['ecg_morphology'].to(device)
                heart_v = data['heart_v'].to(device)
                heart_f = data['heart_f'].to(device)
                subid = data['eid']

                # Forward pass
                v_out, mu, logvar = model(ecg_raw, demographics, ecg_morphology)

                # Check for inf/nan in model outputs
                if torch.isnan(v_out).any():
                    msg = f"Detected nan in v_out (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isinf(v_out).any():
                    msg = f"Detected inf in v_out (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isnan(mu).any():
                    msg = f"Detected nan in mu (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isinf(mu).any():
                    msg = f"Detected inf in mu (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isnan(logvar).any():
                    msg = f"Detected nan in logvar (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isinf(logvar).any():
                    msg = f"Detected inf in logvar (batch {idx})"
                    logger.warning(msg) if logger else print(msg)

                # Check ground truth data
                if torch.isnan(heart_v).any():
                    msg = f"Detected nan in ground truth heart_v (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isinf(heart_v).any():
                    msg = f"Detected inf in ground truth heart_v (batch {idx})"
                    logger.warning(msg) if logger else print(msg)

                # Loss computation
                loss, loss_recon = Loss.VAECELoss(
                    v_out, heart_v, heart_f, logvar, mu,
                    beta=config.beta,
                    lambd=config.lambd,
                    lambd_s=config.lambd_s,
                    loss=config.loss
                )

                # Check for inf/nan in loss
                if torch.isnan(loss).any():
                    msg = f"Detected nan in loss (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isinf(loss).any():
                    msg = f"Detected inf in loss (batch {idx})"
                    logger.warning(msg) if logger else print(msg)
                if torch.isnan(loss_recon).any():
                    msg = f"Detected nan in loss_recon (batch {idx}) - skipping batch"
                    logger.warning(msg) if logger else print(msg)
                    continue
                if torch.isinf(loss_recon).any():
                    msg = f"Detected inf in loss_recon (batch {idx}) - skipping batch"
                    logger.warning(msg) if logger else print(msg)
                    continue

                valid_error.append(loss_recon)

                # Compute mesh metrics
                try:
                    hd_values, assd_values = compute_batch_mesh_metrics(v_out, heart_v, subid)
                    metrics_accumulator.update_raw_values(hd_values, assd_values)
                except Exception as e:
                    msg = f"Warning: Could not compute mesh metrics for validation batch {idx}: {e}"
                    logger.warning(msg) if logger else print(msg)

            except Exception as e:
                msg = f"Error in validation batch {idx}: {e}"
                logger.error(msg) if logger else print(msg)
                continue

        if valid_error:
            this_val_error = torch.mean(torch.stack(valid_error))
        else:
            msg = "Warning: No valid validation batches processed"
            logger.warning(msg) if logger else print(msg)
            this_val_error = float('inf')

        epoch_metrics = metrics_accumulator.compute_epoch_stats()

        msg = f'Chunk {global_chunk_step}, Validation Error: {this_val_error:.6f}'
        logger.info(msg) if logger else print(msg)
        print_metrics_summary(epoch_metrics, "  Val ", logger=logger)
        logger.info('-------------------------------------') if logger else print('-------------------------------------')

        # Log validation metrics (chunk-based)
        if writer is not None:
            writer.add_scalar('Chunk/Val_HD_Mean', epoch_metrics['hd_mean'], global_chunk_step)
            writer.add_scalar('Chunk/Val_HD_Std', epoch_metrics['hd_std'], global_chunk_step)
            writer.add_scalar('Chunk/Val_HD_Median', epoch_metrics['hd_median'], global_chunk_step)
            writer.add_scalar('Chunk/Val_ASSD_Mean', epoch_metrics['assd_mean'], global_chunk_step)
            writer.add_scalar('Chunk/Val_ASSD_Std', epoch_metrics['assd_std'], global_chunk_step)
            writer.add_scalar('Chunk/Val_ASSD_Median', epoch_metrics['assd_median'], global_chunk_step)

        return this_val_error, epoch_metrics


def validate_encoder_weights(model, validloader, gt_mu_csv_path, device, logger=None):
    """
    Simplified encoder validation using compare_encoders.py approach.

    Compares encoder mu output (NORMALIZED) against ground truth mu (NORMALIZED).
    No denormalization needed - both are in the same space.

    Args:
        model: HybridECGMeshVAE model with pretrained encoder
        validloader: Validation data loader
        gt_mu_csv_path: Path to validation_debug_table.csv with NORMALIZED mu values
        device: Device (cuda/cpu)
        logger: Logger instance (optional)

    Returns:
        validation_passed: Boolean (always True for mu comparison)
        metrics: Dictionary with validation metrics
    """
    import pandas as pd
    import numpy as np

    logger.info("\n" + "="*80) if logger else print("\n" + "="*80)
    logger.info("🔍 ENCODER WEIGHT VALIDATION") if logger else print("🔍 ENCODER WEIGHT VALIDATION")
    logger.info("="*80) if logger else print("="*80)

    # Load ground truth mu (normalized)
    gt_df = pd.read_csv(gt_mu_csv_path)
    latent_cols = [f'z_{i}' for i in range(1, 65)]
    gt_mu_dict = {}
    for _, row in gt_df.iterrows():
        eid = int(row['eid_18545'])
        gt_mu_dict[eid] = row[latent_cols].values.astype(np.float32)

    logger.info(f"📂 Loaded {len(gt_mu_dict)} ground truth mu samples") if logger else print(f"📂 Loaded {len(gt_mu_dict)} ground truth mu samples")

    # Run inference
    model.eval()
    mu_list, eids_list = [], []

    with torch.no_grad():
        for data in validloader:
            ecg_raw = data['ecg_raw'].to(device)
            demographics = data['demographics'].to(device)
            ecg_morphology = data['ecg_morphology'].to(device)
            eids = data['eid']

            mu, _ = model.encoder(ecg_raw, demographics, ecg_morphology)
            mu_list.append(mu.cpu().numpy())
            eids_list.append(eids.cpu().numpy())

    # Concatenate
    all_mu = np.concatenate(mu_list, axis=0)
    all_eids = np.concatenate(eids_list, axis=0)

    # Match with ground truth
    matched_mu, matched_gt_mu = [], []
    for i, eid in enumerate(all_eids):
        if eid in gt_mu_dict:
            matched_mu.append(all_mu[i])
            matched_gt_mu.append(gt_mu_dict[eid])

    matched_mu = np.array(matched_mu)
    matched_gt_mu = np.array(matched_gt_mu)

    # Compute metrics (both are NORMALIZED - no transform needed!)
    mse_per_sample = np.mean((matched_mu - matched_gt_mu) ** 2, axis=1)
    overall_mse = np.mean(mse_per_sample)
    mae = np.mean(np.abs(matched_mu - matched_gt_mu))
    corr = np.mean([np.corrcoef(matched_mu[:, d], matched_gt_mu[:, d])[0, 1] for d in range(64)])

    # Report results
    logger.info(f"\n✅ Matched samples: {len(matched_mu)}") if logger else print(f"\n✅ Matched samples: {len(matched_mu)}")
    logger.info(f"   mu range: [{matched_mu.min():.3f}, {matched_mu.max():.3f}]") if logger else print(f"   mu range: [{matched_mu.min():.3f}, {matched_mu.max():.3f}]")
    logger.info(f"   GT mu range: [{matched_gt_mu.min():.3f}, {matched_gt_mu.max():.3f}]") if logger else print(f"   GT mu range: [{matched_gt_mu.min():.3f}, {matched_gt_mu.max():.3f}]")
    logger.info(f"\n📊 Metrics:") if logger else print(f"\n📊 Metrics:")
    logger.info(f"   MSE: {overall_mse:.6f}") if logger else print(f"   MSE: {overall_mse:.6f}")
    logger.info(f"   MAE: {mae:.6f}") if logger else print(f"   MAE: {mae:.6f}")
    logger.info(f"   Correlation: {corr:.4f}") if logger else print(f"   Correlation: {corr:.4f}")
    logger.info("="*80 + "\n") if logger else print("="*80 + "\n")

    metrics = {
        'matched_samples': len(matched_mu),
        'mse': overall_mse,
        'mae': mae,
        'correlation': corr,
    }

    return True, metrics


def validate_decoder_weights(model, validloader, gt_csv_path, device, logger=None):
    """
    Validate that decoder weights are correctly loaded by comparing decoder outputs
    against ground truth meshes when feeding ground truth latent codes.

    Args:
        model: HybridECGMeshVAE model with pretrained decoder
        validloader: Validation data loader
        gt_csv_path: Path to ground truth CSV with columns: eid_18545, z_1, ..., z_64
        device: Device (cuda/cpu)
        logger: Logger instance (optional)

    Returns:
        validation_passed: Boolean indicating if validation passed
        metrics: Dictionary with validation metrics
    """
    import pandas as pd
    import numpy as np

    msg = "\n" + "="*80
    logger.info(msg) if logger else print(msg)
    msg = "🔍 DECODER WEIGHT VALIDATION"
    logger.info(msg) if logger else print(msg)
    msg = "="*80
    logger.info(msg) if logger else print(msg)
    msg = "Testing decoder reconstruction: gt_z → model.decoder → predicted_mesh"
    logger.info(msg) if logger else print(msg)
    msg = "Comparing predicted_mesh against ground truth mesh using HD/ASSD metrics"
    logger.info(msg) if logger else print(msg)
    msg = "="*80 + "\n"
    logger.info(msg) if logger else print(msg)

    # Load ground truth CSV
    msg = f"📂 Loading ground truth latents from: {gt_csv_path}"
    logger.info(msg) if logger else print(msg)

    try:
        gt_df = pd.read_csv(gt_csv_path)
        msg = f"✅ Ground truth loaded: {len(gt_df)} samples, {len(gt_df.columns)} columns"
        logger.info(msg) if logger else print(msg)
    except Exception as e:
        msg = f"❌ Error loading ground truth CSV: {e}"
        logger.error(msg) if logger else print(msg)
        return False, {}

    # Extract EID and latent columns
    eid_col = 'eid_18545'
    latent_cols = [f'z_{i}' for i in range(1, 65)]  # z_1 to z_64

    # Create mapping: eid -> ground truth latent vector
    gt_latent_mapping = {}
    for idx, row in gt_df.iterrows():
        eid = int(row[eid_col])
        latent_vector = row[latent_cols].values.astype(np.float32)
        gt_latent_mapping[eid] = latent_vector

    msg = f"📊 Ground truth latent mapping created: {len(gt_latent_mapping)} EIDs"
    logger.info(msg) if logger else print(msg)

    # Run decoder on ground truth latents
    model.eval()

    matched_count = 0
    unmatched_count = 0
    metrics_accumulator = EpochMetricsAccumulator()

    msg = "\n🔄 Running decoder on ground truth latents and computing metrics..."
    logger.info(msg) if logger else print(msg)

    with torch.no_grad():
        for idx, data in enumerate(validloader):
            try:
                # Unpack data
                eids = data['eid']
                heart_v = data['heart_v'].to(device)  # Ground truth meshes

                # Collect matched samples
                batch_gt_latents = []
                batch_gt_meshes = []
                batch_matched_eids = []

                for b in range(len(eids)):
                    eid = int(eids[b])

                    if eid in gt_latent_mapping:
                        batch_gt_latents.append(gt_latent_mapping[eid])
                        batch_gt_meshes.append(heart_v[b])
                        batch_matched_eids.append(eid)
                    else:
                        unmatched_count += 1

                # Skip batch if no matches
                if len(batch_gt_latents) == 0:
                    continue

                matched_count += len(batch_gt_latents)

                # Convert to tensors
                gt_z = torch.from_numpy(np.array(batch_gt_latents)).to(device).float()
                gt_meshes = torch.stack(batch_gt_meshes)  # [B, seq_len, points, 3]

                # Decode: gt_z -> predicted_mesh
                # Note: gt_z from CSV is already denormalized, pass directly to decoder
                pred_meshes = model.decoder(gt_z)

                # Compute metrics immediately for this batch
                try:
                    hd_values, assd_values = compute_batch_mesh_metrics(pred_meshes, gt_meshes, batch_matched_eids)
                    metrics_accumulator.update_raw_values(hd_values, assd_values)
                except Exception as e:
                    msg = f"⚠️  Error computing metrics for batch {idx}: {e}"
                    logger.warning(msg) if logger else print(msg)
                    continue

            except Exception as e:
                msg = f"⚠️  Error processing batch {idx}: {e}"
                logger.warning(msg) if logger else print(msg)
                continue

    if matched_count == 0:
        msg = "❌ No matching samples found between validation set and ground truth!"
        logger.error(msg) if logger else print(msg)
        return False, {}

    msg = f"\n✅ Matched {matched_count} samples"
    logger.info(msg) if logger else print(msg)
    if unmatched_count > 0:
        msg = f"⚠️  Unmatched samples: {unmatched_count}"
        logger.warning(msg) if logger else print(msg)

    # Compute epoch statistics
    epoch_metrics = metrics_accumulator.compute_epoch_stats()

    # Report results
    msg = "\n" + "="*80
    logger.info(msg) if logger else print(msg)
    msg = "📈 DECODER VALIDATION RESULTS"
    logger.info(msg) if logger else print(msg)
    msg = "="*80
    logger.info(msg) if logger else print(msg)

    msg = f"✅ Matched samples: {matched_count}"
    logger.info(msg) if logger else print(msg)

    msg = f"\n📊 Mesh Reconstruction Metrics:"
    logger.info(msg) if logger else print(msg)
    print_metrics_summary(epoch_metrics, "  Decoder ", logger=logger)

    # Validation criteria: HD and ASSD should be reasonably low
    # These thresholds may need adjustment based on your data
    # Typical good reconstruction: HD < 5mm, ASSD < 1mm
    hd_threshold = 10.0  # mm
    assd_threshold = 2.0  # mm

    validation_passed = (
        epoch_metrics['hd_mean'] < hd_threshold and
        epoch_metrics['assd_mean'] < assd_threshold
    )

    msg = "\n" + "="*80
    logger.info(msg) if logger else print(msg)
    if validation_passed:
        msg = "✅ DECODER VALIDATION PASSED"
        logger.info(msg) if logger else print(msg)
        msg = f"   HD mean {epoch_metrics['hd_mean']:.4f} < {hd_threshold} mm"
        logger.info(msg) if logger else print(msg)
        msg = f"   ASSD mean {epoch_metrics['assd_mean']:.4f} < {assd_threshold} mm"
        logger.info(msg) if logger else print(msg)
        msg = "   Decoder weights appear to be correctly loaded!"
        logger.info(msg) if logger else print(msg)
    else:
        msg = "❌ DECODER VALIDATION FAILED"
        logger.error(msg) if logger else print(msg)
        msg = f"   HD mean {epoch_metrics['hd_mean']:.4f} >= {hd_threshold} mm or"
        logger.error(msg) if logger else print(msg)
        msg = f"   ASSD mean {epoch_metrics['assd_mean']:.4f} >= {assd_threshold} mm"
        logger.error(msg) if logger else print(msg)
        msg = "   Decoder weights may not be correctly loaded!"
        logger.error(msg) if logger else print(msg)
    msg = "="*80 + "\n"
    logger.info(msg) if logger else print(msg)

    # Prepare metrics dictionary
    metrics = {
        'matched_samples': matched_count,
        'hd_mean': epoch_metrics['hd_mean'],
        'hd_median': epoch_metrics['hd_median'],
        'hd_std': epoch_metrics['hd_std'],
        'hd_min': epoch_metrics['hd_min'],
        'hd_max': epoch_metrics['hd_max'],
        'assd_mean': epoch_metrics['assd_mean'],
        'assd_median': epoch_metrics['assd_median'],
        'assd_std': epoch_metrics['assd_std'],
        'assd_min': epoch_metrics['assd_min'],
        'assd_max': epoch_metrics['assd_max'],
        'validation_passed': validation_passed,
    }

    return validation_passed, metrics


def log_hparams(writer, config, final_train_loss=None, final_val_loss=None, best_val_loss=None, logger=None):
    """
    Log hyperparameters to TensorBoard for hybrid VAE.

    Args:
        writer: TensorBoard writer
        config: Configuration object
        final_train_loss: Final training loss
        final_val_loss: Final validation loss
        best_val_loss: Best validation loss achieved
        logger: Logger instance (optional)
    """
    hparams = {
        'train_type': config.train_type,
        'tag': config.tag,
        'learning_rate': config.lr,
        'batch_size': config.batch,
        'z_dim': config.z_dim,
        'n_epochs': config.n_epochs,
        'beta': config.beta,
        'lambd': config.lambd,
        'lambd_s': config.lambd_s,
        'loss_type': config.loss,
        'n_samples': config.n_samples,
        'seq_len': config.seq_len,
        'ff_size': config.ff_size,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'activation': config.activation,
        'weight_decay': config.wd if config.wd else 0.0,
        'surf_type': config.surf_type,
        'val_freq': config.val_freq,
        'model_type': 'hybrid_ecg_mesh_vae',
        'ecg_filter_size': config.ecg_filter_size,
        'ecg_dropout': config.ecg_dropout,
        'load_pretrained': config.load_pretrained,
    }

    metrics = {
        'hparam/final_train_loss': final_train_loss if final_train_loss is not None else 0.0,
        'hparam/final_val_loss': final_val_loss if final_val_loss is not None else 0.0,
        'hparam/best_val_loss': best_val_loss if best_val_loss is not None else 0.0,
    }

    writer.add_hparams(hparams, metrics)

    if logger:
        logger.info("Hybrid VAE HParams logged to TensorBoard:")
        logger.info(f"   Model type: Hybrid ECG-to-Mesh VAE")
        logger.info(f"   Hyperparameters: {len(hparams)} items")
        logger.info(f"   Final train loss: {final_train_loss}")
        logger.info(f"   Final val loss: {final_val_loss}")
        logger.info(f"   Best val loss: {best_val_loss}")
    else:
        print("✅ Hybrid VAE HParams logged to TensorBoard:")
        print(f"   Model type: Hybrid ECG-to-Mesh VAE")
        print(f"   Hyperparameters: {len(hparams)} items")
        print(f"   Final train loss: {final_train_loss}")
        print(f"   Final val loss: {final_val_loss}")
        print(f"   Best val loss: {best_val_loss}")


def train_epoch_with_chunks(model, train_dataset, validloader, optimizer, device,
                           config, writer, epoch, best_val_loss, best_model_path, intermediate_model_path,
                           global_chunk_step, scheduler=None, logger=None):
    """
    Train for one epoch, optionally using chunks for large datasets.

    Args:
        model: HybridECGMeshVAE model
        train_dataset: Training dataset (not dataloader)
        validloader: Validation dataloader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        config: Configuration object
        writer: TensorBoard writer
        epoch: Current epoch number
        best_val_loss: Current best validation loss
        best_model_path: Path to save best model
        intermediate_model_path: Path to save intermediate checkpoints
        global_chunk_step: Global chunk counter for chunk-based logging
        scheduler: Learning rate scheduler (optional)
        logger: Logger instance (optional)

    Returns:
        Tuple of (avg_train_loss, final_val_loss, updated_best_val_loss, updated_global_chunk_step)
    """
    from torch.utils.data import DataLoader, Subset

    # Validate encoder weights before training (only on first epoch)
    if epoch == 1 and config.load_pretrained:
        gt_csv_path = 'jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/validation_best_table.csv'
        gt_mu_csv_path = 'jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/validation_debug_table.csv'

        msg = "\n🔍 Validating pretrained encoder weights before training..."
        logger.info(msg) if logger else print(msg)

        validation_passed, validation_metrics = validate_encoder_weights(
            model=model,
            validloader=validloader,
            gt_mu_csv_path=gt_mu_csv_path,
            device=device,
            logger=logger
        )

        if not validation_passed:
            msg = "⚠️  WARNING: Encoder validation failed, but continuing with training..."
            logger.warning(msg) if logger else print(msg)
            msg = "⚠️  Please check the encoder checkpoint path and weight loading!"
            logger.warning(msg) if logger else print(msg)

        # # Validate decoder weights
        # msg = "\n🔍 Validating pretrained decoder weights before training..."
        # logger.info(msg) if logger else print(msg)

        # decoder_validation_passed, decoder_validation_metrics = validate_decoder_weights(
        #     model, validloader, gt_csv_path, device, logger
        # )

        # if not decoder_validation_passed:
        #     msg = "⚠️  WARNING: Decoder validation failed, but continuing with training..."
        #     logger.warning(msg) if logger else print(msg)
        #     msg = "⚠️  Please check the decoder checkpoint path and weight loading!"
        #     logger.warning(msg) if logger else print(msg)

        # initial validation
        _, _ = val(model, validloader, device, config, writer, global_chunk_step, logger)

    # If no chunking configured, use standard full-epoch training
    if not hasattr(config, 'chunk_size') or config.chunk_size is None:
        trainloader = DataLoader(
            train_dataset,
            batch_size=config.batch,
            shuffle=True,
            num_workers=config.num_workers
        )
        train_loss, _ = train(model, trainloader, optimizer, device, config, writer, epoch, global_chunk_step, logger)
        val_loss, _ = val(model, validloader, device, config, writer, global_chunk_step, logger)
        global_chunk_step += 1
        return train_loss, val_loss, best_val_loss, global_chunk_step

    # Chunked training: divide dataset into chunks
    total_samples = len(train_dataset)
    chunk_size = config.chunk_size
    num_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ceiling division

    msg = f"\n📦 Chunked Training - Epoch {epoch}"
    logger.info(msg) if logger else print(msg)
    msg = f"   Total samples: {total_samples}"
    logger.info(msg) if logger else print(msg)
    msg = f"   Chunk size: {chunk_size}"
    logger.info(msg) if logger else print(msg)
    msg = f"   Number of chunks: {num_chunks}"
    logger.info(msg) if logger else print(msg)

    all_train_losses = []
    final_val_loss = None

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)

        # Apply learning rate scheduler (chunk-based)
        if config.use_scheduler and scheduler is not None:
            if global_chunk_step < config.warmup_chunks:
                # Warmup phase
                current_lr = warmup_lr_chunk_based(optimizer, global_chunk_step, config.warmup_chunks, config.lr)
                if chunk_idx % 1 == 0:  # Log every 1 chunk
                    msg = f"🔥 Warmup chunk {global_chunk_step+1}/{config.warmup_chunks}, LR: {current_lr:.2e}"
                    logger.info(msg) if logger else print(msg)
            elif config.scheduler_type == 'cosine':
                # Cosine annealing phase
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                if chunk_idx % 1 == 0:  # Log every 10 chunks
                    msg = f"📉 Cosine LR: {current_lr:.2e}"
                    logger.info(msg) if logger else print(msg)

        # Create subset for this chunk
        chunk_indices = list(range(start_idx, end_idx))
        chunk_dataset = Subset(train_dataset, chunk_indices)
        chunk_loader = DataLoader(
            chunk_dataset,
            batch_size=config.batch,
            shuffle=True,
            num_workers=config.num_workers
        )

        msg = f"\n  📦 Chunk {chunk_idx+1}/{num_chunks} (Global: {global_chunk_step}): samples [{start_idx}:{end_idx}] ({len(chunk_indices)} samples)"
        logger.info(msg) if logger else print(msg)

        # Train on this chunk
        chunk_train_loss, _ = train(model, chunk_loader, optimizer, device, config, writer, epoch, global_chunk_step, logger)
        all_train_losses.append(chunk_train_loss)

        # Validate after chunk
        chunk_val_loss, chunk_metrics = val(model, validloader, device, config, writer, global_chunk_step, logger)
        final_val_loss = chunk_val_loss

        # Chunk-level TensorBoard logging
        if writer is not None:
            writer.add_scalar('Chunk/Train_Loss', chunk_train_loss, global_chunk_step)
            writer.add_scalar('Chunk/Val_Loss', chunk_val_loss, global_chunk_step)
            writer.add_scalar('Chunk/Best_Val_Loss', best_val_loss, global_chunk_step)
            writer.add_scalar('Chunk/Learning_Rate', optimizer.param_groups[0]['lr'], global_chunk_step)
            if chunk_metrics['num_samples'] > 0:
                writer.add_scalar('Chunk/Val_HD_Mean', chunk_metrics['hd_mean'], global_chunk_step)
                writer.add_scalar('Chunk/Val_ASSD_Mean', chunk_metrics['assd_mean'], global_chunk_step)

        # Save intermediate checkpoint after each chunk
        save_model(model, optimizer, epoch, chunk_train_loss, chunk_val_loss,
                   intermediate_model_path, model_type="intermediate", logger=logger)

        # Update best model if this chunk achieved better validation
        if chunk_val_loss < best_val_loss:
            best_val_loss = chunk_val_loss

            # Save best model checkpoint
            save_model(model, optimizer, epoch, chunk_train_loss, chunk_val_loss,
                        best_model_path, model_type="best", logger=logger)

            msg = f"     🎉 New best validation loss: {best_val_loss:.3f}"
            logger.info(msg) if logger else print(msg)

        # Increment global chunk counter
        global_chunk_step += 1

    # Return averaged training loss across all chunks
    avg_train_loss = np.mean(all_train_losses)
    msg = f"\n  📊 Epoch {epoch} Summary:"
    logger.info(msg) if logger else print(msg)
    msg = f"     Average train loss: {avg_train_loss:.6f}"
    logger.info(msg) if logger else print(msg)
    msg = f"     Final val loss: {final_val_loss:.6f}"
    logger.info(msg) if logger else print(msg)
    msg = f"     Best val loss: {best_val_loss:.6f}"
    logger.info(msg) if logger else print(msg)

    # Epoch-level summary logging to TensorBoard
    if writer is not None:
        writer.add_scalar('Epoch/Train_Loss_Avg', avg_train_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss_Final', final_val_loss, epoch)
        writer.add_scalar('Epoch/Best_Val_Loss', best_val_loss, epoch)

    return avg_train_loss, final_val_loss, best_val_loss, global_chunk_step


def main(config):
    """
    Main training function for Hybrid ECG-to-Mesh VAE.

    Args:
        config: Configuration object from load_config()
    """
    
    # Validate configuration
    validate_config(config)

    # Basic configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    
    # set up model directory
    model_dir = config.model_dir
    device = config.device
    train_type = config.train_type

    n_epochs = config.n_epochs
    lr = config.lr
    
    # Generate model name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{train_type}_z{config.z_dim}_beta{config.beta}_lr{config.lr}_batch{config.batch}_{timestamp}"

    # Setup paths
    logdir = f"{model_dir}/tb/{model_name}"
    cp_path = f"{model_dir}/model/{model_name}"
    loggerdir = f"{model_dir}/logs/{model_name}"
    best_model_path = f"{cp_path}/best_model.pt"
    intermediate_model_path = f"{cp_path}/intermediate_checkpoint.pt"

    # Setup logging
    logger = setup_logging(loggerdir)

    logger.info('=' * 80)
    logger.info('🚀 Hybrid ECG-to-Mesh VAE Training')
    logger.info('=' * 80)
    logger.info('   Model: ECG Encoder + Mesh Decoder')
    logger.info('   Input: 12-lead ECG + Demographics + Morphology')
    logger.info('   Output: 50-frame cardiac mesh sequences')
    logger.info('   Architecture: ResNet1D → Latent Space → Transformer Decoder')
    logger.info('=' * 80)
    
    # print all config parameters
    logger.info('Configuration Parameters:')
    for attr, value in config.__dict__.items():
        logger.info(f"   {attr}: {value}")

    # Create directories
    util.setup_dir(logdir)
    util.setup_dir(cp_path)
    logger.info(f"📁 Directories created:")
    logger.info(f"   TensorBoard logs: {logdir}")
    logger.info(f"   Model checkpoints: {cp_path}")

    # Create hybrid datasets
    try:
        logger.info("📊 Creating hybrid ECG-Mesh datasets...")
        data_module = HybridDataModule(
            train_csv_path=config.train_csv,
            val_csv_path=config.val_csv,
            preprocessed_ecg_path=config.preprocessed_ecg_path,
            ecg_phenotypes_path=config.ecg_phenotypes_path,
            target_seg_dir=config.target_seg_dir,
            seq_len=config.seq_len,
            n_samples=config.n_samples,
            surf_type=config.surf_type,
            batch_size=config.batch,
            num_workers=config.num_workers,
        )
        data_module.setup("fit")

        trainloader = data_module.train_dataloader()
        validloader = data_module.val_dataloader()

        logger.info(f"✅ Datasets created successfully")
        logger.info(f"   Training samples: {len(data_module.train_dataset)}")
        logger.info(f"   Validation samples: {len(data_module.val_dataset)}")

        # Get motion scaler for latent denormalization
        motion_scaler = data_module.get_motion_scaler()
        if motion_scaler is not None:
            logger.info(f"✅ Motion scaler extracted from training data")
        else:
            logger.warning(f"⚠️  No motion scaler available - latents will remain normalized")

    except Exception as e:
        logger.error(f"❌ Error creating datasets: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Create hybrid model
    try:
        logger.info("🏗️  Creating Hybrid ECG-to-Mesh VAE...")
        model = HybridECGMeshVAE(
            latent_dim=config.z_dim,
            seq_len=config.seq_len,
            points=config.n_samples,
            ecg_filter_size=config.ecg_filter_size,
            ecg_dropout=config.ecg_dropout,
            ecg_conv1_kernel_size=config.ecg_conv1_kernel_size,
            ecg_conv1_stride=config.ecg_conv1_stride,
            ecg_padding=config.ecg_padding,
            ff_size=config.ff_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            activation=config.activation,
            decoder_dropout=config.decoder_dropout,
            motion_scaler=motion_scaler,
            use_motion_denorm=getattr(config, 'use_motion_denorm', True),
        )

        # Initialize from pretrained checkpoints if requested
        if config.load_pretrained:
            logger.info("🔄 Loading pretrained weights...")
            model = initialize_from_pretrained(
                model,
                config.ecg_encoder_checkpoint,
                config.mesh_decoder_checkpoint,
                device
            )

        model = model.to(device)
        logger.info("✅ Model created and moved to device!")

    except Exception as e:
        logger.error(f"❌ Error creating model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Setup optimizer
    if config.wd:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Setup learning rate scheduler (chunk-based)
    scheduler = None
    total_chunks_per_epoch = 1  # Default for non-chunked training
    if hasattr(config, 'chunk_size') and config.chunk_size:
        total_chunks_per_epoch = (len(data_module.train_dataset) + config.chunk_size - 1) // config.chunk_size

    logger.info(f"📊 Training configuration:")
    logger.info(f"   Total chunks per epoch: {total_chunks_per_epoch}")

    if config.use_scheduler:
        # Calculate total training chunks
        total_training_chunks = total_chunks_per_epoch * config.n_epochs
        cosine_chunks = config.total_chunks_for_cosine or (total_training_chunks - config.warmup_chunks)

        if config.scheduler_type == 'cosine':
            # CosineAnnealingLR - step every chunk after warmup
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_chunks,
                eta_min=config.min_lr
            )
            logger.info(f"📉 CosineAnnealingLR scheduler (CHUNK-BASED):")
            logger.info(f"   Warmup chunks: {config.warmup_chunks}")
            logger.info(f"   Cosine T_max: {cosine_chunks} chunks")
            logger.info(f"   Total training chunks: {total_training_chunks}")
            logger.info(f"   Min LR: {config.min_lr}")
        elif config.scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=config.min_lr
            )
            logger.info(f"📉 ReduceLROnPlateau scheduler:")
            logger.info(f"   Factor: 0.5, Patience: 10 chunks")

    # Setup tensorboard
    writer = SummaryWriter(logdir)

    # root cause of the replication error
    # # Test model with sample data
    # logger.info("🧪 Testing model with sample data...")
    # try:
    #     sample_batch = next(iter(trainloader))
    #     ecg_raw = sample_batch['ecg_raw'].to(device)
    #     demographics = sample_batch['demographics'].to(device)
    #     ecg_morphology = sample_batch['ecg_morphology'].to(device)
    #     heart_v = sample_batch['heart_v'].to(device)

    #     logger.info(f"✅ Sample batch loaded:")
    #     logger.info(f"   ECG: {ecg_raw.shape}")
    #     logger.info(f"   Demographics: {demographics.shape}")
    #     logger.info(f"   Morphology: {ecg_morphology.shape}")
    #     logger.info(f"   Mesh: {heart_v.shape}")

    #     with torch.no_grad():
    #         v_out, mu, logvar = model(ecg_raw, demographics, ecg_morphology)
    #         logger.info(f"✅ Model test successful:")
    #         logger.info(f"   Output mesh: {v_out.shape}")
    #         logger.info(f"   Latent mu: {mu.shape}")
    #         logger.info(f"   Latent logvar: {logvar.shape}")

    # except Exception as e:
    #     logger.error(f"❌ Model test failed: {e}")
    #     import traceback
    #     logger.error(traceback.format_exc())
    #     return

    # Display chunking info if enabled
    if hasattr(config, 'chunk_size') and config.chunk_size:
        logger.info(f"📦 Chunked Training:")
        logger.info(f"   Chunk size: {config.chunk_size} samples")
        logger.info(f"   Estimated chunks per epoch: {(len(data_module.train_dataset) + config.chunk_size - 1) // config.chunk_size}")

    logger.info("-" * 80)

    # Initialize global chunk counter for chunk-based logging and scheduler
    global_chunk_step = 0

    best_val_loss = float('inf')
    final_train_loss = None
    final_val_loss = None
    val_freq = config.val_freq

    # for epoch in tqdm(range(1, n_epochs + 1), desc="Training Hybrid VAE"):
    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()

        # Training with chunk-based tracking (wrapper handles both modes)
        train_loss, val_loss, best_val_loss, global_chunk_step = train_epoch_with_chunks(
            model, data_module.train_dataset, validloader, optimizer,
            device, config, writer, epoch, best_val_loss, best_model_path, intermediate_model_path,
            global_chunk_step, scheduler, logger
        )

        final_train_loss = train_loss
        final_val_loss = val_loss

        # Save best model (best_val_loss already updated by wrapper)
        if epoch % val_freq == 0:
            # Save best model checkpoint
            save_model(model, optimizer, epoch, train_loss, val_loss,
                      best_model_path, model_type="best", logger=logger)

            # Save intermediate checkpoint
            if epoch % 10 == 0 and epoch > 0:
                save_model(model, optimizer, epoch, train_loss, val_loss,
                          intermediate_model_path, model_type="intermediate", logger=logger)

            # Tensorboard logging
            try:
                writer.add_scalar('Train_Loss', train_loss, epoch)
                writer.add_scalar('Val_Loss', val_loss, epoch)
                writer.add_scalar('Best_Val_Loss', best_val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            except Exception as e:
                logger.error(f"Error writing to tensorboard: {e}")

        # Timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_duration // 60:.0f}:{epoch_duration % 60:.2f}")

    # Training completion
    logger.info("=" * 80)
    logger.info("🎊 Hybrid ECG-to-Mesh VAE Training Completed!")
    logger.info(f"📈 Final train loss: {final_train_loss:.6f}")
    logger.info(f"📉 Final validation loss: {final_val_loss:.6f}")
    logger.info(f"🏆 Best validation loss: {best_val_loss:.6f}")
    logger.info(f"💾 Best model saved at: {best_model_path}")
    logger.info("=" * 80)

    # Log final results
    log_hparams(writer, config, final_train_loss, final_val_loss, best_val_loss, logger)

    writer.add_scalar('Summary/Best_Val_Loss', best_val_loss, n_epochs)
    writer.add_scalar('Summary/Final_Train_Loss', final_train_loss, n_epochs)
    writer.add_scalar('Summary/Final_Val_Loss', final_val_loss, n_epochs)

    writer.close()
    logger.info("✅ Hybrid VAE training and logging complete!")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    # Load configuration
    config = load_config()

    # Run training
    main(config)
