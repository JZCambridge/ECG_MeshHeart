#!/usr/bin/env python
"""
Encoder Comparison Script
Compares Original ECGMotionEncoderVAE vs Hybrid ECGEncoder

This script:
1. Loads both encoder models with pretrained weights
2. Runs inference on validation set
3. Compares mu and z outputs against ground truth
4. Generates detailed performance reports
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple
import logging

# Import original encoder
from echonext_preprocess_motion_vae.model_resnet1d_morphology_vae import (
    ECGMotionEncoderVAE,
    ResNet1dWithTabular,
)

# Import hybrid encoder
from hybrid_ecg_mesh_vae.model.ecg_encoder import ECGEncoder

# Import dataloader
from echonext_preprocess_motion_vae.loader_ecg_preprocessed import MotionDataModulePreprocessed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EncoderComparison:
    """Compare Original vs Hybrid Encoder models"""

    def __init__(
        self,
        checkpoint_path: str,
        train_csv_path: str,
        val_csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        gt_mu_csv_path: str,
        gt_z_csv_path: str,
        device: str = 'cuda:0',
        batch_size: int = 32,
    ):
        """
        Initialize comparison environment

        Args:
            checkpoint_path: Path to pretrained checkpoint
            train_csv_path: Path to training CSV (needed for fitting scalers)
            val_csv_path: Path to validation CSV
            preprocessed_ecg_path: Path to preprocessed ECG .pt file
            ecg_phenotypes_path: Path to ECG morphology phenotypes CSV
            gt_mu_csv_path: Path to ground truth mu CSV
            gt_z_csv_path: Path to ground truth z CSV
            device: Device for computation
            batch_size: Batch size for inference
        """
        self.checkpoint_path = checkpoint_path
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.gt_mu_csv_path = gt_mu_csv_path
        self.gt_z_csv_path = gt_z_csv_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        logger.info("=" * 80)
        logger.info("🔍 ENCODER COMPARISON TOOL")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Training CSV: {train_csv_path}")
        logger.info(f"Validation CSV: {val_csv_path}")
        logger.info(f"GT mu CSV: {gt_mu_csv_path}")
        logger.info(f"GT z CSV: {gt_z_csv_path}")
        logger.info("=" * 80)

    def load_dataloader(self):
        """Load validation dataloader with scalers fitted on training data"""
        logger.info("\n📊 Loading validation dataloader...")
        logger.info("   NOTE: Using training CSV to fit motion and morphology scalers")

        data_module = MotionDataModulePreprocessed(
            train_csv_path=self.train_csv_path,  # Training data for fitting scalers
            val_csv_path=self.val_csv_path,      # Validation data for inference
            preprocessed_ecg_path=self.preprocessed_ecg_path,
            ecg_phenotypes_path=self.ecg_phenotypes_path,
            batch_size=self.batch_size,
            num_workers=4,
        )

        data_module.setup("fit")
        val_loader = data_module.val_dataloader()

        # Store motion scaler for denormalization
        self.motion_scaler = data_module.motion_scaler

        logger.info(f"✅ Validation dataloader created")
        logger.info(f"   Train samples (for scaler): {len(data_module.train_dataset)}")
        logger.info(f"   Val samples (for inference): {len(val_loader.dataset)}")
        logger.info(f"   Batches: {len(val_loader)}")
        logger.info(f"   Motion scaler fitted: {self.motion_scaler is not None}")

        return val_loader

    def load_original_encoder(self):
        """Load original ECGMotionEncoderVAE model"""
        logger.info("\n🔧 Loading Original Encoder (ECGMotionEncoderVAE)...")

        # Create model
        resnet_model = ResNet1dWithTabular(
            len_tabular_feature_vector=25,  # 9 demographics + 16 morphology
            filter_size=64,
            input_channels=12,
            dropout_value=0.5,
            num_classes=128,  # 128 = 64 (mu) + 64 (logvar)
            conv1_kernel_size=15,
            conv1_stride=2,
            padding=7,
        )

        model = ECGMotionEncoderVAE(
            model=resnet_model,
            lr=1e-3,
            beta=0.01,
            checkpoint_dir=None,
            num_epochs=100,
        )

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info(f"✅ Original encoder loaded successfully")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def load_hybrid_encoder(self):
        """Load hybrid ECGEncoder model"""
        logger.info("\n🔧 Loading Hybrid Encoder (ECGEncoder)...")

        # Create model
        model = ECGEncoder(
            latent_dim=64,
            filter_size=64,
            input_channels=12,
            dropout_value=0.5,
            conv1_kernel_size=15,
            conv1_stride=2,
            padding=7,
        )

        # Load checkpoint - extract encoder weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint['state_dict']

        # Map keys: model.* → encoder.resnet.*
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if 'model.' in k:
                new_key = k.replace('model.', 'resnet.')
                encoder_state_dict[new_key] = v

        # Load weights
        missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        logger.info(f"✅ Hybrid encoder loaded successfully")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        if missing:
            logger.warning(f"   Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"   Unexpected keys: {len(unexpected)}")

        return model

    def run_inference_original(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference with original encoder

        Returns:
            mu_all: [N, 64] - mean of latent distribution
            z_all: [N, 64] - sampled latent codes
            eids_all: [N] - patient EIDs
        """
        logger.info("\n🔄 Running inference with Original Encoder...")

        mu_list = []
        z_list = []
        eids_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Original Encoder"):
                # Move to device
                for key in ['ecg_raw', 'demographics', 'ecg_morphology', 'motion_latent']:
                    batch[key] = batch[key].to(self.device)

                # Forward pass
                out = model(batch)

                # Store results
                mu_list.append(out['mu'].cpu().numpy())
                z_list.append(out['z'].cpu().numpy())
                eids_list.append(batch['eid'].cpu().numpy())

        # Concatenate
        mu_all = np.concatenate(mu_list, axis=0)
        z_all = np.concatenate(z_list, axis=0)
        eids_all = np.concatenate(eids_list, axis=0)

        logger.info(f"✅ Original encoder inference complete")
        logger.info(f"   mu shape: {mu_all.shape}")
        logger.info(f"   z shape: {z_all.shape}")
        logger.info(f"   Samples: {len(eids_all)}")

        return mu_all, z_all, eids_all

    def run_inference_hybrid(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference with hybrid encoder

        Returns:
            mu_all: [N, 64] - mean of latent distribution
            z_all: [N, 64] - sampled latent codes
            eids_all: [N] - patient EIDs
        """
        logger.info("\n🔄 Running inference with Hybrid Encoder...")

        mu_list = []
        z_list = []
        eids_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Hybrid Encoder"):
                # Extract inputs
                ecg_raw = batch['ecg_raw'].to(self.device)
                demographics = batch['demographics'].to(self.device)
                ecg_morphology = batch['ecg_morphology'].to(self.device)
                eids = batch['eid']

                # Forward pass
                mu, logvar = model(ecg_raw, demographics, ecg_morphology)

                # Reparameterize to get z
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                # Store results
                mu_list.append(mu.cpu().numpy())
                z_list.append(z.cpu().numpy())
                eids_list.append(eids.cpu().numpy())

        # Concatenate
        mu_all = np.concatenate(mu_list, axis=0)
        z_all = np.concatenate(z_list, axis=0)
        eids_all = np.concatenate(eids_list, axis=0)

        logger.info(f"✅ Hybrid encoder inference complete")
        logger.info(f"   mu shape: {mu_all.shape}")
        logger.info(f"   z shape: {z_all.shape}")
        logger.info(f"   Samples: {len(eids_all)}")

        return mu_all, z_all, eids_all

    def load_ground_truth(self) -> Tuple[Dict, Dict]:
        """
        Load ground truth mu and z from CSVs

        Returns:
            gt_mu_dict: {eid: mu_array}
            gt_z_dict: {eid: z_array}
        """
        logger.info("\n📂 Loading ground truth from CSVs...")

        # Load mu
        df_mu = pd.read_csv(self.gt_mu_csv_path)
        logger.info(f"   GT mu CSV: {len(df_mu)} samples")

        # Load z
        df_z = pd.read_csv(self.gt_z_csv_path)
        logger.info(f"   GT z CSV: {len(df_z)} samples")

        # Create dictionaries
        latent_cols = [f'z_{i}' for i in range(1, 65)]

        gt_mu_dict = {}
        for _, row in df_mu.iterrows():
            eid = int(row['eid_18545'])
            mu = row[latent_cols].values.astype(np.float32)
            gt_mu_dict[eid] = mu

        gt_z_dict = {}
        for _, row in df_z.iterrows():
            eid = int(row['eid_18545'])
            z = row[latent_cols].values.astype(np.float32)
            gt_z_dict[eid] = z

        logger.info(f"✅ Ground truth loaded")
        logger.info(f"   GT mu samples: {len(gt_mu_dict)}")
        logger.info(f"   GT z samples: {len(gt_z_dict)}")

        return gt_mu_dict, gt_z_dict

    def match_and_compute_metrics(
        self,
        mu: np.ndarray,
        z: np.ndarray,
        eids: np.ndarray,
        gt_mu_dict: Dict,
        gt_z_dict: Dict,
        model_name: str
    ) -> Dict:
        """
        Match predictions with ground truth and compute metrics

        Args:
            mu: Predicted mu [N, 64]
            z: Predicted z [N, 64]
            eids: Patient EIDs [N]
            gt_mu_dict: Ground truth mu dictionary
            gt_z_dict: Ground truth z dictionary
            model_name: Name of model for logging

        Returns:
            Dictionary with metrics
        """
        logger.info(f"\n📊 Computing metrics for {model_name}...")

        # Match samples
        matched_mu = []
        matched_z = []
        matched_gt_mu = []
        matched_gt_z = []
        matched_eids = []

        for i, eid in enumerate(eids):
            if eid in gt_mu_dict and eid in gt_z_dict:
                matched_mu.append(mu[i])
                matched_z.append(z[i])
                matched_gt_mu.append(gt_mu_dict[eid])
                matched_gt_z.append(gt_z_dict[eid])
                matched_eids.append(eid)

        # Convert to arrays
        matched_mu = np.array(matched_mu)
        matched_z = np.array(matched_z)
        matched_gt_mu = np.array(matched_gt_mu)
        matched_gt_z = np.array(matched_gt_z)

        logger.info(f"   Matched samples: {len(matched_eids)}/{len(eids)}")

        # Denormalize z for comparison (mu stays normalized)
        if self.motion_scaler is not None:
            matched_z_denorm = self.motion_scaler.inverse_transform(matched_z)
            logger.info(f"   Applied inverse transform to z")
            logger.info(f"   z range (normalized): [{matched_z.min():.3f}, {matched_z.max():.3f}]")
            logger.info(f"   z range (denormalized): [{matched_z_denorm.min():.3f}, {matched_z_denorm.max():.3f}]")
            logger.info(f"   GT z range: [{matched_gt_z.min():.3f}, {matched_gt_z.max():.3f}]")
        else:
            matched_z_denorm = matched_z
            logger.warning(f"   No motion scaler available - using normalized z")

        logger.info(f"   mu range: [{matched_mu.min():.3f}, {matched_mu.max():.3f}]")
        logger.info(f"   GT mu range: [{matched_gt_mu.min():.3f}, {matched_gt_mu.max():.3f}]")

        # Compute metrics for mu (no transform)
        mse_mu_per_sample = np.mean((matched_mu - matched_gt_mu) ** 2, axis=1)
        mse_mu_overall = np.mean(mse_mu_per_sample)
        mae_mu_overall = np.mean(np.abs(matched_mu - matched_gt_mu))

        # Compute metrics for z (denormalized)
        mse_z_per_sample = np.mean((matched_z_denorm - matched_gt_z) ** 2, axis=1)
        mse_z_overall = np.mean(mse_z_per_sample)
        mae_z_overall = np.mean(np.abs(matched_z_denorm - matched_gt_z))

        # Per-dimension metrics
        mse_mu_per_dim = np.mean((matched_mu - matched_gt_mu) ** 2, axis=0)
        mse_z_per_dim = np.mean((matched_z_denorm - matched_gt_z) ** 2, axis=0)

        # Correlation per dimension
        corr_mu_per_dim = []
        corr_z_per_dim = []

        for dim in range(64):
            # Mu correlation
            corr_mu = np.corrcoef(matched_mu[:, dim], matched_gt_mu[:, dim])[0, 1]
            corr_mu_per_dim.append(corr_mu)

            # Z correlation
            corr_z = np.corrcoef(matched_z_denorm[:, dim], matched_gt_z[:, dim])[0, 1]
            corr_z_per_dim.append(corr_z)

        corr_mu_per_dim = np.array(corr_mu_per_dim)
        corr_z_per_dim = np.array(corr_z_per_dim)

        # Mean correlation
        mean_corr_mu = np.mean(corr_mu_per_dim)
        mean_corr_z = np.mean(corr_z_per_dim)

        metrics = {
            'model_name': model_name,
            'matched_samples': len(matched_eids),
            # Mu metrics
            'mse_mu_overall': mse_mu_overall,
            'mae_mu_overall': mae_mu_overall,
            'mean_corr_mu': mean_corr_mu,
            'mse_mu_per_sample': mse_mu_per_sample,
            'mse_mu_per_dim': mse_mu_per_dim,
            'corr_mu_per_dim': corr_mu_per_dim,
            # Z metrics
            'mse_z_overall': mse_z_overall,
            'mae_z_overall': mae_z_overall,
            'mean_corr_z': mean_corr_z,
            'mse_z_per_sample': mse_z_per_sample,
            'mse_z_per_dim': mse_z_per_dim,
            'corr_z_per_dim': corr_z_per_dim,
            # Raw data for export
            'eids': matched_eids,
        }

        logger.info(f"✅ Metrics computed for {model_name}")
        logger.info(f"   mu MSE: {mse_mu_overall:.6f}")
        logger.info(f"   mu MAE: {mae_mu_overall:.6f}")
        logger.info(f"   mu correlation: {mean_corr_mu:.4f}")
        logger.info(f"   z MSE: {mse_z_overall:.6f}")
        logger.info(f"   z MAE: {mae_z_overall:.6f}")
        logger.info(f"   z correlation: {mean_corr_z:.4f}")

        return metrics

    def generate_report(self, metrics_original: Dict, metrics_hybrid: Dict):
        """Generate comparison report and save to files"""
        logger.info("\n📝 Generating comparison report...")

        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(output_dir, exist_ok=True)

        # Summary report
        summary_path = os.path.join(output_dir, 'comparison_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ENCODER COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("ORIGINAL MODEL (ECGMotionEncoderVAE)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Matched samples: {metrics_original['matched_samples']}\n")
            f.write(f"mu MSE:          {metrics_original['mse_mu_overall']:.6f}\n")
            f.write(f"mu MAE:          {metrics_original['mae_mu_overall']:.6f}\n")
            f.write(f"mu correlation:  {metrics_original['mean_corr_mu']:.4f}\n")
            f.write(f"z MSE:           {metrics_original['mse_z_overall']:.6f}\n")
            f.write(f"z MAE:           {metrics_original['mae_z_overall']:.6f}\n")
            f.write(f"z correlation:   {metrics_original['mean_corr_z']:.4f}\n\n")

            f.write("HYBRID MODEL (ECGEncoder)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Matched samples: {metrics_hybrid['matched_samples']}\n")
            f.write(f"mu MSE:          {metrics_hybrid['mse_mu_overall']:.6f}\n")
            f.write(f"mu MAE:          {metrics_hybrid['mae_mu_overall']:.6f}\n")
            f.write(f"mu correlation:  {metrics_hybrid['mean_corr_mu']:.4f}\n")
            f.write(f"z MSE:           {metrics_hybrid['mse_z_overall']:.6f}\n")
            f.write(f"z MAE:           {metrics_hybrid['mae_z_overall']:.6f}\n")
            f.write(f"z correlation:   {metrics_hybrid['mean_corr_z']:.4f}\n\n")

            f.write("COMPARISON\n")
            f.write("-" * 80 + "\n")

            # Determine winner based on overall MSE
            mu_winner = "Original" if metrics_original['mse_mu_overall'] < metrics_hybrid['mse_mu_overall'] else "Hybrid"
            z_winner = "Original" if metrics_original['mse_z_overall'] < metrics_hybrid['mse_z_overall'] else "Hybrid"

            mu_diff = abs(metrics_original['mse_mu_overall'] - metrics_hybrid['mse_mu_overall'])
            z_diff = abs(metrics_original['mse_z_overall'] - metrics_hybrid['mse_z_overall'])

            f.write(f"mu MSE winner:   {mu_winner} (difference: {mu_diff:.6f})\n")
            f.write(f"z MSE winner:    {z_winner} (difference: {z_diff:.6f})\n")

            corr_mu_diff = abs(metrics_original['mean_corr_mu'] - metrics_hybrid['mean_corr_mu'])
            corr_z_diff = abs(metrics_original['mean_corr_z'] - metrics_hybrid['mean_corr_z'])

            corr_mu_winner = "Original" if metrics_original['mean_corr_mu'] > metrics_hybrid['mean_corr_mu'] else "Hybrid"
            corr_z_winner = "Original" if metrics_original['mean_corr_z'] > metrics_hybrid['mean_corr_z'] else "Hybrid"

            f.write(f"mu corr winner:  {corr_mu_winner} (difference: {corr_mu_diff:.4f})\n")
            f.write(f"z corr winner:   {corr_z_winner} (difference: {corr_z_diff:.4f})\n")
            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"✅ Summary report saved: {summary_path}")

        # Per-sample errors
        per_sample_path = os.path.join(output_dir, 'per_sample_errors.csv')
        df_samples = pd.DataFrame({
            'eid': metrics_original['eids'],
            'original_mu_mse': metrics_original['mse_mu_per_sample'],
            'hybrid_mu_mse': metrics_hybrid['mse_mu_per_sample'],
            'original_z_mse': metrics_original['mse_z_per_sample'],
            'hybrid_z_mse': metrics_hybrid['mse_z_per_sample'],
        })
        df_samples.to_csv(per_sample_path, index=False)
        logger.info(f"✅ Per-sample errors saved: {per_sample_path}")

        # Per-dimension analysis
        per_dim_path = os.path.join(output_dir, 'per_dimension_analysis.csv')
        df_dims = pd.DataFrame({
            'dimension': range(1, 65),
            'original_mu_mse': metrics_original['mse_mu_per_dim'],
            'hybrid_mu_mse': metrics_hybrid['mse_mu_per_dim'],
            'original_mu_corr': metrics_original['corr_mu_per_dim'],
            'hybrid_mu_corr': metrics_hybrid['corr_mu_per_dim'],
            'original_z_mse': metrics_original['mse_z_per_dim'],
            'hybrid_z_mse': metrics_hybrid['mse_z_per_dim'],
            'original_z_corr': metrics_original['corr_z_per_dim'],
            'hybrid_z_corr': metrics_hybrid['corr_z_per_dim'],
        })
        df_dims.to_csv(per_dim_path, index=False)
        logger.info(f"✅ Per-dimension analysis saved: {per_dim_path}")

        # Print summary to console
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL COMPARISON SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nOriginal Model:")
        logger.info(f"  mu MSE: {metrics_original['mse_mu_overall']:.6f}")
        logger.info(f"  z MSE:  {metrics_original['mse_z_overall']:.6f}")
        logger.info(f"  mu corr: {metrics_original['mean_corr_mu']:.4f}")
        logger.info(f"  z corr:  {metrics_original['mean_corr_z']:.4f}")
        logger.info(f"\nHybrid Model:")
        logger.info(f"  mu MSE: {metrics_hybrid['mse_mu_overall']:.6f}")
        logger.info(f"  z MSE:  {metrics_hybrid['mse_z_overall']:.6f}")
        logger.info(f"  mu corr: {metrics_hybrid['mean_corr_mu']:.4f}")
        logger.info(f"  z corr:  {metrics_hybrid['mean_corr_z']:.4f}")
        logger.info(f"\nWinner: {mu_winner} (mu MSE), {z_winner} (z MSE)")
        logger.info("=" * 80)

    def run(self):
        """Run complete comparison pipeline"""
        try:
            # Load dataloader
            val_loader = self.load_dataloader()

            # Load models
            original_model = self.load_original_encoder()
            hybrid_model = self.load_hybrid_encoder()

            # Run inference
            mu_orig, z_orig, eids_orig = self.run_inference_original(original_model, val_loader)
            mu_hybrid, z_hybrid, eids_hybrid = self.run_inference_hybrid(hybrid_model, val_loader)

            # Load ground truth
            gt_mu_dict, gt_z_dict = self.load_ground_truth()

            # Compute metrics
            metrics_original = self.match_and_compute_metrics(
                mu_orig, z_orig, eids_orig, gt_mu_dict, gt_z_dict, "Original"
            )

            metrics_hybrid = self.match_and_compute_metrics(
                mu_hybrid, z_hybrid, eids_hybrid, gt_mu_dict, gt_z_dict, "Hybrid"
            )

            # Generate report
            self.generate_report(metrics_original, metrics_hybrid)

            logger.info("\n✅ Comparison complete!")

        except Exception as e:
            logger.error(f"❌ Error during comparison: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    """Main entry point"""

    # Configuration
    checkpoint_path = 'jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
    train_csv_path = 'ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt64_12ecg64_71k/train_ecg_motion.csv'
    val_csv_path = 'ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt64_12ecg64_71k/val_ecg_motion.csv'
    preprocessed_ecg_path = 'ukb/jz_ecg/ecg_echonext_15Oct25/preprocessed_ecg_12x2500_v1_15Oct25_parallel.pt'
    ecg_phenotypes_path = 'cardiac/pi514/ukbb_ecg/Final/2_Factor_ECG/data/pt_data_ecg/ecg_phenotypes.csv'
    gt_mu_csv_path = 'jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/validation_debug_table.csv'
    gt_z_csv_path = 'jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/validation_best_table.csv'

    # Run comparison
    comparison = EncoderComparison(
        checkpoint_path=checkpoint_path,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        preprocessed_ecg_path=preprocessed_ecg_path,
        ecg_phenotypes_path=ecg_phenotypes_path,
        gt_mu_csv_path=gt_mu_csv_path,
        gt_z_csv_path=gt_z_csv_path,
        device='cuda:0',
        batch_size=4,
    )

    comparison.run()


if __name__ == "__main__":
    main()
