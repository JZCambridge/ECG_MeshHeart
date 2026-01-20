#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Inference Script for Motion Latent Generation with Deterministic Encoder
Generates motion latent predictions using direct encoder prediction (no sampling)

Author: Adapted from echonext_preprocess_motion_vae for Encoder
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and dataloader
from model_resnet1d_morphology_encoder_weighted import ECGMotionEncoder, make_model
from loader_ecg_preprocessed_weighted import MotionDataModulePreprocessed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device, config):
    """
    Load trained Encoder model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.ckpt)
        device: Device to load model on
        config: Model configuration dictionary

    Returns:
        Loaded ECGMotionEncoder model
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model
    model = make_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Log checkpoint info
    epoch = checkpoint.get("epoch", "unknown")
    val_loss = checkpoint.get("val_loss", "unknown")

    logger.info(f"✅ Checkpoint loaded successfully!")
    logger.info(f"   Epoch: {epoch}")
    logger.info(f"   Validation Loss: {val_loss}")

    return model


def generate_motion_predictions(model, val_dataset, val_loader, device, output_path):
    """
    Generate motion latent predictions using deterministic encoder.

    Args:
        model: Trained ECGMotionEncoder model
        val_dataset: Validation dataset (for inverse transform)
        val_loader: Validation data loader
        device: Device to use
        output_path: Path to save predictions CSV

    Returns:
        Path to saved CSV and predictions DataFrame
    """
    logger.info(f"🔮 Generating motion latent predictions...")
    logger.info(f"   Mode: Deterministic encoder (direct prediction)")

    model.eval()
    all_predictions = []
    all_eids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing")):
            try:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward pass
                out = model(batch)
                prediction = out["prediction"]  # [B, 512]

                # Convert to numpy
                prediction_np = prediction.cpu().numpy()
                eids = batch["eid"].cpu().numpy()

                all_predictions.append(prediction_np)
                all_eids.append(eids)

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue

    # Concatenate
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_eids = np.concatenate(all_eids)

    logger.info(f"✅ Generated {len(all_eids)} predictions")

    # Apply inverse transformation
    logger.info("Applying inverse transformation...")
    if val_dataset.motion_scaler is not None:
        all_predictions = val_dataset.inverse_transform_motion(all_predictions)
        logger.info("✅ Inverse transformation applied")
    else:
        logger.warning("⚠️  No motion scaler found - predictions are in normalized scale")

    # Create DataFrame
    column_names = ['eid_18545'] + [f'z_{i}' for i in range(1, 513)]
    data = np.column_stack([all_eids, all_predictions])
    predictions_df = pd.DataFrame(data, columns=column_names)

    # Convert EID to int
    try:
        predictions_df['eid_18545'] = predictions_df['eid_18545'].astype(int)
    except Exception as e:
        logger.warning(f"Could not convert EIDs to integers: {e}")

    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"📊 Predictions saved to: {output_path}")
    logger.info(f"📈 Predictions shape: {predictions_df.shape}")

    return output_path, predictions_df


def main(args):
    """
    Main inference function.

    Args:
        args: Command-line arguments

    Returns:
        None
    """
    # Setup device
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

    # Model configuration (must match training config)
    config = {
        "lr": 1e-3,  # Not used for inference
        "batch_size": args.batch_size,
        "filter_size": args.filter_size,
        "dropout": args.dropout,
        "checkpoint_dir": os.path.dirname(args.checkpoint_path),
        "epochs": 100,  # Not used for inference
    }

    logger.info("\n" + "="*60)
    logger.info("MOTION LATENT INFERENCE (ENCODER)")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Validation CSV: {args.val_csv}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60 + "\n")

    # Load checkpoint
    model = load_checkpoint(args.checkpoint_path, device, config)

    # Create validation dataset
    logger.info("\n=== Setting up validation dataset ===")
    data_module = MotionDataModulePreprocessed(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup(stage="fit")

    val_dataset = data_module.val_dataset
    val_loader = data_module.val_dataloader()

    logger.info(f"✅ Validation dataset ready: {len(val_dataset)} samples")

    # Generate predictions
    output_path, predictions_df = generate_motion_predictions(
        model=model,
        val_dataset=val_dataset,
        val_loader=val_loader,
        device=device,
        output_path=args.output_path
    )

    # Show sample predictions
    logger.info("\n" + "="*60)
    logger.info("SAMPLE PREDICTIONS (first 3 rows)")
    logger.info("="*60)
    print(predictions_df.head(3).to_string(index=False))
    logger.info("="*60)

    logger.info("\n✅ Inference completed successfully!")
    logger.info(f"📁 Output saved to: {output_path}")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference: Generate motion latent predictions from trained ECG-to-Motion Encoder model"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default='jzheng12/Codes/ECG_MeshHeart/output/echonext_motion_latent_generation_weighted/checkpoints_20260118_152930/checkpoint_epoch_45.ckpt',
        help="Path to trained Encoder model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ukb/jz_meshheart/echonext_motion_latents_16Jan26/echonext_splits/inference/all_echonext_latent_ep45.csv",
        help="Path to save predictions CSV file",
    )

    # Data paths
    parser.add_argument(
        "--train_csv",
        type=str,
        default="ukb/jz_meshheart/echonext_motion_latents_16Jan26/echonext_splits/combined_wt_volume_ecg_motion.csv",
        help="Path to training CSV (needed for fitting scalers)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="ukb/jz_meshheart/echonext_motion_latents_16Jan26/echonext_splits/combined_wt_volume_ecg_motion.csv",
        help="Path to validation CSV file for inference",
    )
    parser.add_argument(
        "--preprocessed_ecg",
        type=str,
        default="ukb/jz_ecg/ecg_echonext_15Oct25/preprocessed_ecg_12x2500_v1_15Oct25_parallel.pt",
        help="Path to preprocessed ECG .pt file (12×2500)",
    )
    parser.add_argument(
        "--ecg_phenotypes_path",
        type=str,
        default="cardiac/pi514/ukbb_ecg/Final/2_Factor_ECG/data/pt_data_ecg/ecg_phenotypes.csv",
        help="Path to ECG morphology phenotypes CSV file",
    )

    # Inference parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use (e.g., 0, 1, 2)",
    )

    # Model parameters (must match training config)
    parser.add_argument(
        "--filter_size",
        type=int,
        default=64,
        help="ResNet filter size (must match training)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (must match training)",
    )

    # Note: --beta, --stochastic, --deterministic, --num_samples arguments removed (not needed for encoder)
    
    args = parser.parse_args()
    
    # check output file folders exist else create
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    main(args)
