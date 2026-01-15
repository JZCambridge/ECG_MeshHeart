#!/usr/bin/env python
"""
Diagnostic Script: Comprehensive Dataloader and Scaler Validation

This script validates the hybrid dataloader against the original one by comparing:
1. Motion scaler parameters between dataloaders
2. Sample-by-sample data loading (ECG, demographics, morphology)
3. Sample-by-sample encoder outputs
4. EID alignment and ordering

Logs are saved to: hybrid_encoder_debug/results/diagnose_scaler_issue_YYYYMMDD_HHMMSS.log
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
from typing import Dict
import logging

# Import both dataloaders
from echonext_preprocess_motion_vae.loader_ecg_preprocessed import MotionDataModulePreprocessed
from hybrid_ecg_mesh_vae.data.hybrid_dataloader_optimized import OptimizedHybridDataModule

# Import encoders
from echonext_preprocess_motion_vae.model_resnet1d_morphology_vae import ECGMotionEncoderVAE, ResNet1dWithTabular
from hybrid_ecg_mesh_vae.model.ecg_encoder import ECGEncoder

# Create results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(RESULTS_DIR, f'diagnose_scaler_issue_{timestamp}.log')

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"📄 Log file: {log_file}")


def compare_scalers(scaler1, scaler2, name1="Scaler1", name2="Scaler2"):
    """Compare two StandardScaler objects"""
    logger.info(f"\n📊 Comparing {name1} vs {name2}")
    logger.info("=" * 80)

    if scaler1 is None or scaler2 is None:
        logger.error(f"One or both scalers are None: {name1}={scaler1}, {name2}={scaler2}")
        return False

    # Compare means
    mean_diff = np.abs(scaler1.mean_ - scaler2.mean_)
    scale_diff = np.abs(scaler1.scale_ - scaler2.scale_)

    logger.info(f"\n{name1} statistics:")
    logger.info(f"  Mean range: [{scaler1.mean_.min():.4f}, {scaler1.mean_.max():.4f}]")
    logger.info(f"  Scale range: [{scaler1.scale_.min():.4f}, {scaler1.scale_.max():.4f}]")
    logger.info(f"  Mean (first 5): {scaler1.mean_[:5]}")
    logger.info(f"  Scale (first 5): {scaler1.scale_[:5]}")

    logger.info(f"\n{name2} statistics:")
    logger.info(f"  Mean range: [{scaler2.mean_.min():.4f}, {scaler2.mean_.max():.4f}]")
    logger.info(f"  Scale range: [{scaler2.scale_.min():.4f}, {scaler2.scale_.max():.4f}]")
    logger.info(f"  Mean (first 5): {scaler2.mean_[:5]}")
    logger.info(f"  Scale (first 5): {scaler2.scale_[:5]}")

    logger.info(f"\nDifferences:")
    logger.info(f"  Max mean difference: {mean_diff.max():.6f}")
    logger.info(f"  Max scale difference: {scale_diff.max():.6f}")
    logger.info(f"  Mean mean difference: {mean_diff.mean():.6f}")
    logger.info(f"  Mean scale difference: {scale_diff.mean():.6f}")

    # Check if scalers are identical
    identical = np.allclose(scaler1.mean_, scaler2.mean_, rtol=1e-5) and \
                np.allclose(scaler1.scale_, scaler2.scale_, rtol=1e-5)

    if identical:
        logger.info("\n✅ Scalers are IDENTICAL")
    else:
        logger.error("\n❌ Scalers are DIFFERENT!")
        logger.error("   This will cause incorrect denormalization during validation!")

    return identical


def compare_dataloader_samples(original_loader, hybrid_loader, n_samples=5):
    """Compare raw data samples from both dataloaders"""
    logger.info("\n🔍 Comparing Dataloader Sample Outputs")
    logger.info("=" * 80)

    # Get first batch from each loader
    original_batch = next(iter(original_loader))
    hybrid_batch = next(iter(hybrid_loader))

    # Check EIDs
    original_eids = original_batch['eid'].numpy()
    hybrid_eids = hybrid_batch['eid'].numpy()

    logger.info(f"\n📋 EID Comparison:")
    logger.info(f"  Original batch size: {len(original_eids)}")
    logger.info(f"  Hybrid batch size: {len(hybrid_eids)}")
    logger.info(f"  Original EIDs (first 10): {original_eids[:10]}")
    logger.info(f"  Hybrid EIDs (first 10): {hybrid_eids[:10]}")

    # Find common EIDs
    common_eids = []
    for i, eid in enumerate(original_eids[:n_samples]):
        if eid in hybrid_eids:
            hybrid_idx = np.where(hybrid_eids == eid)[0][0]
            common_eids.append((i, hybrid_idx, eid))

    logger.info(f"\n  Common EIDs in first {n_samples} samples: {len(common_eids)}")

    if len(common_eids) == 0:
        logger.warning("❌ No common EIDs found - dataloaders have different sample ordering!")
        return False

    # Compare sample-by-sample
    all_match = True
    for orig_idx, hybrid_idx, eid in common_eids:
        logger.info(f"\n{'='*80}")
        logger.info(f"EID {eid} (Original idx={orig_idx}, Hybrid idx={hybrid_idx})")
        logger.info(f"{'='*80}")

        # Compare ECG
        orig_ecg = original_batch['ecg_raw'][orig_idx].numpy()
        hybrid_ecg = hybrid_batch['ecg_raw'][hybrid_idx].numpy()

        ecg_diff = np.abs(orig_ecg - hybrid_ecg)
        ecg_max_diff = ecg_diff.max()
        ecg_mean_diff = ecg_diff.mean()

        logger.info(f"\n📊 ECG Comparison:")
        logger.info(f"  Shape: Original={orig_ecg.shape}, Hybrid={hybrid_ecg.shape}")
        logger.info(f"  Original range: [{orig_ecg.min():.4f}, {orig_ecg.max():.4f}]")
        logger.info(f"  Hybrid range: [{hybrid_ecg.min():.4f}, {hybrid_ecg.max():.4f}]")
        logger.info(f"  Max difference: {ecg_max_diff:.6f}")
        logger.info(f"  Mean difference: {ecg_mean_diff:.6f}")

        if ecg_max_diff > 1e-5:
            logger.error(f"  ❌ ECG data DIFFERS significantly!")
            all_match = False
        else:
            logger.info(f"  ✅ ECG data matches")

        # Compare demographics
        orig_demo = original_batch['demographics'][orig_idx].numpy()
        hybrid_demo = hybrid_batch['demographics'][hybrid_idx].numpy()

        demo_diff = np.abs(orig_demo - hybrid_demo)
        demo_max_diff = demo_diff.max()

        logger.info(f"\n📊 Demographics Comparison:")
        logger.info(f"  Original: {orig_demo}")
        logger.info(f"  Hybrid: {hybrid_demo}")
        logger.info(f"  Max difference: {demo_max_diff:.6f}")

        if demo_max_diff > 1e-5:
            logger.error(f"  ❌ Demographics DIFFER!")
            all_match = False
        else:
            logger.info(f"  ✅ Demographics match")

        # Compare morphology
        orig_morph = original_batch['ecg_morphology'][orig_idx].numpy()
        hybrid_morph = hybrid_batch['ecg_morphology'][hybrid_idx].numpy()

        morph_diff = np.abs(orig_morph - hybrid_morph)
        morph_max_diff = morph_diff.max()

        logger.info(f"\n📊 Morphology Comparison:")
        logger.info(f"  Original (first 5): {orig_morph[:5]}")
        logger.info(f"  Hybrid (first 5): {hybrid_morph[:5]}")
        logger.info(f"  Max difference: {morph_max_diff:.6f}")

        if morph_max_diff > 1e-5:
            logger.error(f"  ❌ Morphology DIFFERS!")
            all_match = False
        else:
            logger.info(f"  ✅ Morphology matches")

    logger.info(f"\n{'='*80}")
    if all_match:
        logger.info("✅ ALL SAMPLES MATCH between dataloaders")
    else:
        logger.error("❌ SOME SAMPLES DIFFER between dataloaders")

    return all_match


def compare_encoder_outputs(original_loader, hybrid_loader, original_model, hybrid_model, device, gt_csv_path, n_samples=10):
    """Compare encoder outputs for same samples between original and hybrid loaders"""
    logger.info("\n🔍 Comparing Encoder Outputs")
    logger.info("=" * 80)

    # Load ground truth
    gt_df = pd.read_csv(gt_csv_path)
    latent_cols = [f'z_{i}' for i in range(1, 65)]
    gt_mapping = {}
    for _, row in gt_df.iterrows():
        eid = int(row['eid_18545'])
        mu = row[latent_cols].values.astype(np.float32)
        gt_mapping[eid] = mu

    logger.info(f"Loaded {len(gt_mapping)} ground truth samples")

    # Get first batch from each loader
    original_batch = next(iter(original_loader))
    hybrid_batch = next(iter(hybrid_loader))

    # Check EIDs
    original_eids = original_batch['eid'].numpy()
    hybrid_eids = hybrid_batch['eid'].numpy()

    # Run inference
    original_model.eval()
    hybrid_model.eval()

    with torch.no_grad():
        # Original encoder
        for key in ['ecg_raw', 'demographics', 'ecg_morphology']:
            original_batch[key] = original_batch[key].to(device)
        original_out = original_model(original_batch)
        original_mu = original_out['mu'].cpu().numpy()

        # Hybrid encoder
        hybrid_ecg = hybrid_batch['ecg_raw'].to(device)
        hybrid_demo = hybrid_batch['demographics'].to(device)
        hybrid_morph = hybrid_batch['ecg_morphology'].to(device)
        hybrid_mu, _ = hybrid_model(hybrid_ecg, hybrid_demo, hybrid_morph)
        hybrid_mu = hybrid_mu.cpu().numpy()

    # Compare for common EIDs
    logger.info(f"\n📊 Per-Sample Encoder Comparison:")
    matched_count = 0
    for i, eid in enumerate(original_eids[:n_samples]):
        if eid in gt_mapping and eid in hybrid_eids:
            gt_mu = gt_mapping[eid]
            hybrid_idx = np.where(hybrid_eids == eid)[0][0]

            # Compute differences (all in NORMALIZED space)
            orig_vs_gt_mse = np.mean((original_mu[i] - gt_mu) ** 2)
            hybrid_vs_gt_mse = np.mean((hybrid_mu[hybrid_idx] - gt_mu) ** 2)
            orig_vs_hybrid_mse = np.mean((original_mu[i] - hybrid_mu[hybrid_idx]) ** 2)

            logger.info(f"\nEID {eid}:")
            logger.info(f"  Original mu vs GT:  MSE = {orig_vs_gt_mse:.6f}")
            logger.info(f"  Hybrid mu vs GT:    MSE = {hybrid_vs_gt_mse:.6f}")
            logger.info(f"  Original vs Hybrid: MSE = {orig_vs_hybrid_mse:.6f}")

            if orig_vs_hybrid_mse < 1e-5:
                logger.info(f"  ✅ Encoder outputs MATCH")
            else:
                logger.error(f"  ❌ Encoder outputs DIFFER")

            matched_count += 1

    logger.info(f"\n✅ Compared {matched_count} samples")


def diagnose_hybrid_dataloader(train_csv, val_csv, preprocessed_ecg_path, ecg_phenotypes_path, target_seg_dir):
    """Diagnose motion_scaler in hybrid dataloader"""
    logger.info("\n🔬 DIAGNOSING HYBRID DATALOADER")
    logger.info("=" * 80)

    # Create hybrid data module
    logger.info("\nCreating hybrid data module...")
    hybrid_dm = OptimizedHybridDataModule(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        target_seg_dir=target_seg_dir,
        seq_len=50,
        n_samples=1412,
        surf_type='all',
        batch_size=4,
        num_workers=0,
        use_shared_memory=False,
    )
    hybrid_dm.setup("fit")

    # Get scalers
    train_motion_scaler = hybrid_dm.train_dataset.motion_scaler
    val_motion_scaler = hybrid_dm.val_dataset.motion_scaler

    logger.info("\n📊 Hybrid Dataloader Scaler Analysis:")
    logger.info(f"  Train dataset has motion_scaler: {train_motion_scaler is not None}")
    logger.info(f"  Val dataset has motion_scaler: {val_motion_scaler is not None}")

    if train_motion_scaler is not None:
        logger.info(f"\nTrain motion_scaler:")
        logger.info(f"  Mean (first 5): {train_motion_scaler.mean_[:5]}")
        logger.info(f"  Scale (first 5): {train_motion_scaler.scale_[:5]}")

    if val_motion_scaler is not None:
        logger.info(f"\nVal motion_scaler:")
        logger.info(f"  Mean (first 5): {val_motion_scaler.mean_[:5]}")
        logger.info(f"  Scale (first 5): {val_motion_scaler.scale_[:5]}")

    # Check if they're the same object
    if train_motion_scaler is val_motion_scaler:
        logger.info("\n✅ Train and val datasets share the SAME motion_scaler object")
    else:
        logger.error("\n❌ Train and val datasets have DIFFERENT motion_scaler objects!")
        logger.error("   This is a BUG - validation should reuse training scaler!")

    return hybrid_dm, train_motion_scaler, val_motion_scaler


def main():
    """Main diagnostic routine"""

    # Configuration
    checkpoint_path = 'jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
    train_csv_path = 'ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt64_12ecg64_71k/train_ecg_motion.csv'
    val_csv_path = 'ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt64_12ecg64_71k/val_ecg_motion.csv'
    preprocessed_ecg_path = 'ukb/jz_ecg/ecg_echonext_15Oct25/preprocessed_ecg_12x2500_v1_15Oct25_parallel.pt'
    ecg_phenotypes_path = 'cardiac/pi514/ukbb_ecg/Final/2_Factor_ECG/data/pt_data_ecg/ecg_phenotypes.csv'
    target_seg_dir = 'ukb/jz_meshheart/raw_cine_cardiac_image_cmr_3x_sax/meshheart_cine_cardiac_image_cmr_sax_2/cardiac_MRI_cine_SAX_data'
    gt_mu_csv_path = 'jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/validation_debug_table.csv'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 80)
    logger.info("🔍 COMPREHENSIVE DATALOADER DIAGNOSTIC TOOL")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("=" * 80)

    # ========================================
    # TEST 1: Diagnose hybrid dataloader
    # ========================================
    hybrid_dm, train_scaler_hybrid, val_scaler_hybrid = diagnose_hybrid_dataloader(
        train_csv_path, val_csv_path, preprocessed_ecg_path, ecg_phenotypes_path, target_seg_dir
    )

    # ========================================
    # TEST 2: Setup original dataloader
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 SETTING UP ORIGINAL DATALOADER")
    logger.info("=" * 80)

    original_dm = MotionDataModulePreprocessed(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        batch_size=4,
        num_workers=0,
    )
    original_dm.setup("fit")

    train_scaler_original = original_dm.train_dataset.motion_scaler
    val_scaler_original = original_dm.val_dataset.motion_scaler

    logger.info(f"  Train dataset has motion_scaler: {train_scaler_original is not None}")
    logger.info(f"  Val dataset has motion_scaler: {val_scaler_original is not None}")

    if train_scaler_original is val_scaler_original:
        logger.info("\n✅ Train and val datasets share the SAME motion_scaler object")

    # ========================================
    # TEST 3: Compare scalers
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🔬 CROSS-DATALOADER SCALER COMPARISON")
    logger.info("=" * 80)

    scalers_match = compare_scalers(
        train_scaler_original,
        train_scaler_hybrid,
        "Original Train Scaler",
        "Hybrid Train Scaler"
    )

    # ========================================
    # TEST 4: Compare dataloader samples
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🔍 DATALOADER SAMPLE COMPARISON")
    logger.info("=" * 80)

    original_val_loader = original_dm.val_dataloader()
    hybrid_val_loader = hybrid_dm.val_dataloader()

    samples_match = compare_dataloader_samples(original_val_loader, hybrid_val_loader, n_samples=5)

    # ========================================
    # TEST 5: Compare encoder outputs
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("🔍 ENCODER OUTPUT COMPARISON")
    logger.info("=" * 80)

    # Load models
    logger.info("\nLoading original encoder...")
    resnet_model = ResNet1dWithTabular(
        len_tabular_feature_vector=25,
        filter_size=64,
        input_channels=12,
        dropout_value=0.5,
        num_classes=128,
        conv1_kernel_size=15,
        conv1_stride=2,
        padding=7,
    )
    original_model = ECGMotionEncoderVAE(
        model=resnet_model,
        lr=1e-3,
        beta=0.01,
        checkpoint_dir=None,
        num_epochs=100,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    original_model.load_state_dict(checkpoint['state_dict'])
    original_model = original_model.to(device)
    original_model.eval()
    logger.info("✅ Original encoder loaded")

    logger.info("\nLoading hybrid encoder...")
    hybrid_model = ECGEncoder(
        latent_dim=64,
        filter_size=64,
        input_channels=12,
        dropout_value=0.5,
        conv1_kernel_size=15,
        conv1_stride=2,
        padding=7,
    )
    encoder_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'model.' in k:
            new_key = k.replace('model.', 'resnet.')
            encoder_state_dict[new_key] = v
    hybrid_model.load_state_dict(encoder_state_dict, strict=False)
    hybrid_model = hybrid_model.to(device)
    hybrid_model.eval()
    logger.info("✅ Hybrid encoder loaded")

    compare_encoder_outputs(
        original_val_loader,
        hybrid_val_loader,
        original_model,
        hybrid_model,
        device,
        gt_mu_csv_path,
        n_samples=10
    )

    # ========================================
    # FINAL SUMMARY
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("📋 DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)

    logger.info("\n1️⃣  Hybrid Dataloader Scaler:")
    if train_scaler_hybrid is val_scaler_hybrid:
        logger.info("   ✅ Train and val share the same scaler (CORRECT)")
    else:
        logger.error("   ❌ Train and val have different scalers (BUG!)")

    logger.info("\n2️⃣  Cross-Dataloader Scaler Comparison:")
    if scalers_match:
        logger.info("   ✅ Original and Hybrid scalers match (CORRECT)")
    else:
        logger.error("   ❌ Original and Hybrid scalers differ (POTENTIAL ISSUE!)")

    logger.info("\n3️⃣  Dataloader Sample Comparison:")
    if samples_match:
        logger.info("   ✅ Original and Hybrid dataloaders produce identical samples (CORRECT)")
    else:
        logger.error("   ❌ Original and Hybrid dataloaders produce different samples (BUG!)")

    logger.info("\n4️⃣  Recommendation:")
    if not scalers_match or not samples_match:
        logger.error("   The hybrid dataloader has issues!")
        if not scalers_match:
            logger.error("   - Motion scaler is fitted on different data")
        if not samples_match:
            logger.error("   - Dataloader produces different samples (ECG/demographics/morphology)")
        logger.error("   FIX: Review hybrid_dataloader_optimized.py implementation")
    else:
        logger.info("   ✅ Dataloaders appear correct and consistent!")

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Diagnostic complete! Log saved to: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
