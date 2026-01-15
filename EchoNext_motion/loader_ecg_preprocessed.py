#!/usr/bin/env python
"""
Fast ECG loader for preprocessed 12×2500 ECG data

Loads preprocessed ECG data from a single PyTorch file for efficient training.
Replaces the slow CSV-based loading with instant dictionary lookup.
Modified version with configurable ecg_phenotypes_path.
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGMotionDatasetPreprocessed(Dataset):
    """Dataset for ECG-to-Motion with preprocessed 12×2500 ECG data"""

    def __init__(
        self,
        csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        motion_scaler: Optional[StandardScaler] = None,
        morphology_scaler: Optional[Pipeline] = None,
        is_train: bool = True,
    ):
        """
        Initialize dataset with preprocessed ECG data

        Args:
            csv_path: Path to CSV with demographics and motion embeddings
            preprocessed_ecg_path: Path to .pt file with preprocessed ECGs
            ecg_phenotypes_path: Path to CSV file with ECG morphology phenotypes
            motion_scaler: StandardScaler for motion embeddings (from training set)
            morphology_scaler: Pipeline for ECG morphology features (from training set)
            is_train: Whether this is training data (fit scaler) or val/test (use scaler)
        """
        self.csv_path = csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.is_train = is_train

        # ECG parameters
        self.ecg_num_leads = 12
        self.ecg_num_timepoints = 2500

        # Load preprocessed ECG data (ONCE during init)
        logger.info(f"Loading preprocessed ECG data from {preprocessed_ecg_path}...")
        self.ecg_data = torch.load(preprocessed_ecg_path)
        logger.info(f"✅ Loaded {len(self.ecg_data)} preprocessed ECG samples")

        # Get available (eid, instance) keys
        available_keys = set(self.ecg_data.keys())
        logger.info(f"   Available EID-instance pairs: {len(available_keys)}")

        # Sample key check
        sample_key = list(available_keys)[0]
        sample_ecg = self.ecg_data[sample_key]
        logger.info(f"   Sample ECG shape: {sample_ecg.shape}")
        assert sample_ecg.shape == (self.ecg_num_leads, self.ecg_num_timepoints), \
            f"Expected shape (12, 2500), got {sample_ecg.shape}"

        # Load ECG morphology phenotypes
        logger.info(f"Loading ECG morphology phenotypes from {ecg_phenotypes_path}...")
        self.df_morphology = pd.read_csv(ecg_phenotypes_path)
        logger.info(f"✅ Loaded {len(self.df_morphology)} ECG morphology samples")

        # Define morphology feature columns
        self.morphology_cols = [
            'VentricularRate', 'PQInterval', 'PDuration', 'QRSDuration',
            'QTInterval', 'QTCInterval', 'RRInterval', 'PPInterval',
            'PAxis', 'RAxis', 'TAxis', 'POnset', 'POffset',
            'QOnset', 'QOffset', 'TOffset'
        ]

        # Define atrial-related features (fill with 0 for atrial arrhythmia)
        self.atrial_features = ['PQInterval', 'PDuration', 'PPInterval', 'PAxis', 'POnset', 'POffset']

        # Load CSV with demographics and motion embeddings
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")

        # Store EID columns
        if 'eid_18545' in self.df.columns:
            self.eids = self.df['eid_18545'].values.astype(int)
        else:
            raise ValueError("'eid_18545' column not found in CSV")

        if 'eid_40616' in self.df.columns:
            self.eids_40616 = self.df['eid_40616'].values.astype(int)
        else:
            raise ValueError("'eid_40616' column not found in CSV")

        # Define column names
        self.demo_cols = ['Age', 'Sex', 'Weight', 'Height', 'DBP_at_MRI', 'SBP_at_MRI', 'BMI', 'BSA', 'Ethnic_background']
        self.motion_cols = [f'mesh_embed_{i}' for i in range(1, 65)]

        # Verify columns exist
        missing_cols = []
        for col in self.demo_cols + self.motion_cols:
            if col not in self.df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        # Extract features (NO normalization for demographics)
        self.demographics = self.df[self.demo_cols].values.astype(np.float32)
        self.motion_embeddings = self.df[self.motion_cols].values.astype(np.float32)

        # Handle missing values
        self.demographics = np.nan_to_num(self.demographics, nan=0.0)
        self.motion_embeddings = np.nan_to_num(self.motion_embeddings, nan=0.0)

        # Build EID → instance mapping (with priority: instance 2 > 3 > others)
        logger.info("Building EID → instance mapping...")
        self.eid_to_instance = self._build_eid_instance_mapping()

        # Build morphology lookup: (eid_40616, instance) → morphology_features
        logger.info("Building morphology lookup...")
        self.morphology_lookup = self._build_morphology_lookup()

        # Check coverage
        available_count = sum(1 for eid in self.eids_40616 if self.eid_to_instance.get(eid) is not None)
        logger.info(f"✅ ECG coverage: {available_count}/{len(self.eids_40616)} samples "
                   f"({100*available_count/len(self.eids_40616):.1f}%)")

        # Prepare morphology features for all samples
        logger.info("Loading morphology features for all samples...")
        morphology_features = []
        for eid_40616 in self.eids_40616:
            instance = self.eid_to_instance.get(eid_40616)
            if instance is not None:
                key = (eid_40616, instance)
                morph = self.morphology_lookup.get(key, np.full(16, np.nan, dtype=np.float32))
            else:
                morph = np.full(16, np.nan, dtype=np.float32)
            morphology_features.append(morph)

        self.morphology_features = np.array(morphology_features, dtype=np.float32)  # [N, 16]

        # Morphology normalization (StandardScaler + MedianImputer)
        if is_train:
            # Fit pipeline on training data
            self.morphology_scaler = Pipeline([
                ('scale', StandardScaler()),
                ('impute', SimpleImputer(strategy='median'))
            ])
            self.morphology_features = self.morphology_scaler.fit_transform(self.morphology_features)
            logger.info("Fitted morphology scaler (StandardScaler + MedianImputer) on training data")
        else:
            # Use provided scaler for validation/test
            if morphology_scaler is None:
                raise ValueError("Morphology scaler must be provided for non-training datasets")

            self.morphology_scaler = morphology_scaler
            self.morphology_features = self.morphology_scaler.transform(self.morphology_features)
            logger.info("Applied morphology scaler from training set")

        # Motion normalization (only motion, NOT demographics)
        if is_train:
            # Fit scaler on training data
            self.motion_scaler = StandardScaler()
            self.motion_embeddings = self.motion_scaler.fit_transform(self.motion_embeddings)
            logger.info("Fitted StandardScaler on training motion latents")
        else:
            # Use provided scaler for validation/test
            if motion_scaler is None:
                raise ValueError("Motion scaler must be provided for non-training datasets")

            self.motion_scaler = motion_scaler
            self.motion_embeddings = self.motion_scaler.transform(self.motion_embeddings)
            logger.info("Applied StandardScaler from training set")

        logger.info(f"Demographics shape: {self.demographics.shape}")
        logger.info(f"Morphology features shape: {self.morphology_features.shape}")
        logger.info(f"Motion embeddings shape: {self.motion_embeddings.shape}")

    def _build_eid_instance_mapping(self) -> Dict[int, Optional[int]]:
        """
        Build mapping from EID → instance with priority: 2 > 3 > others

        Returns:
            Dict mapping eid_40616 → best instance
        """
        eid_to_instance = {}

        # Collect all (eid, instance) pairs from preprocessed data
        eid_instances = {}
        for eid, instance in self.ecg_data.keys():
            if eid not in eid_instances:
                eid_instances[eid] = []
            eid_instances[eid].append(instance)

        # For each EID in our dataset, pick the best instance
        for eid in self.eids_40616:
            if eid in eid_instances:
                instances = eid_instances[eid]
                # Priority: instance 2 > 3 > others (lowest)
                if 2 in instances:
                    eid_to_instance[eid] = 2
                elif 3 in instances:
                    eid_to_instance[eid] = 3
                else:
                    eid_to_instance[eid] = min(instances)  # Take lowest instance
            else:
                eid_to_instance[eid] = None  # No ECG available

        return eid_to_instance

    def _build_morphology_lookup(self) -> Dict:
        """
        Build lookup dictionary: (eid_40616, instance) → morphology array [16]

        Returns:
            Dict mapping (eid, instance) → numpy array of morphology features
        """
        morphology_lookup = {}

        # Extract morphology features
        df_morph = self.df_morphology.copy()

        # Special handling: Fill atrial-related features with 0 (for AF/atrial arrhythmia)
        for col in self.atrial_features:
            if col in df_morph.columns:
                df_morph[col] = df_morph[col].fillna(0)

        # Group by (eid_40616, Instance) and store morphology array
        for _, row in df_morph.iterrows():
            eid = int(row['eid_40616'])
            instance = int(row['Instance'])

            # Extract morphology values
            morph_values = []
            for col in self.morphology_cols:
                val = row[col]
                # Convert to float, NaN becomes np.nan (will be imputed later)
                morph_values.append(float(val) if pd.notna(val) else np.nan)

            morphology_lookup[(eid, instance)] = np.array(morph_values, dtype=np.float32)

        logger.info(f"   Built morphology lookup with {len(morphology_lookup)} (eid, instance) pairs")
        return morphology_lookup

    def _load_ecg(self, eid: int) -> torch.Tensor:
        """
        Load preprocessed ECG for given EID

        Args:
            eid: eid_40616

        Returns:
            [12, 2500] tensor (zeros if not found)
        """
        instance = self.eid_to_instance.get(eid)

        if instance is None:
            # No ECG available for this EID
            return torch.zeros((self.ecg_num_leads, self.ecg_num_timepoints), dtype=torch.float32)

        # Look up preprocessed ECG
        key = (eid, instance)
        if key in self.ecg_data:
            return self.ecg_data[key]
        else:
            # Should not happen if mapping is correct
            logger.warning(f"ECG not found for key {key}")
            return torch.zeros((self.ecg_num_leads, self.ecg_num_timepoints), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                demographics: [9] tensor
                ecg_raw: [12, 2500] tensor (preprocessed)
                ecg_morphology: [16] tensor
                motion_latent: [64] tensor
                eid: int
        """
        eid = self.eids[idx]
        eid_40616 = self.eids_40616[idx]

        # Load preprocessed ECG (instant lookup!)
        ecg_preprocessed = self._load_ecg(eid_40616)  # [12, 2500]

        return {
            "demographics": torch.FloatTensor(self.demographics[idx]),  # [9]
            "ecg_raw": ecg_preprocessed,  # [12, 2500] - already preprocessed!
            "ecg_morphology": torch.FloatTensor(self.morphology_features[idx]),  # [16]
            "motion_latent": torch.FloatTensor(self.motion_embeddings[idx]),  # [64]
            "eid": eid
        }

    def get_scalers(self):
        """Return scalers for use with validation dataset"""
        return self.motion_scaler, self.morphology_scaler

    def inverse_transform_motion(self, motion):
        """Inverse transform motion embeddings back to original scale"""
        if self.motion_scaler is not None:
            if isinstance(motion, torch.Tensor):
                motion = motion.cpu().numpy()
            return self.motion_scaler.inverse_transform(motion)
        return motion

    def get_eids(self):
        """Return the EID values for all samples"""
        return self.eids


class MotionDataModulePreprocessed:
    """
    Data module for ECG-to-Motion with preprocessed ECG data.
    Manages train/val datasets and dataloaders.
    """

    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Initialize data module

        Args:
            train_csv_path: Path to training CSV
            val_csv_path: Path to validation CSV
            preprocessed_ecg_path: Path to preprocessed ECG .pt file
            ecg_phenotypes_path: Path to CSV file with ECG morphology phenotypes
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.motion_scaler = None
        self.morphology_scaler = None

    def setup(self, stage: str = "fit"):
        """Setup datasets for training"""
        if stage == "fit" or stage is None:
            print("\n=== Setting up training dataset ===")
            self.train_dataset = ECGMotionDatasetPreprocessed(
                csv_path=self.train_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                motion_scaler=None,
                morphology_scaler=None,
                is_train=True,
            )

            # Store scalers from training data
            self.motion_scaler, self.morphology_scaler = self.train_dataset.get_scalers()

            print("\n=== Setting up validation dataset ===")
            self.val_dataset = ECGMotionDatasetPreprocessed(
                csv_path=self.val_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                motion_scaler=self.motion_scaler,
                morphology_scaler=self.morphology_scaler,
                is_train=False,
            )

    def train_dataloader(self):
        """Return training dataloader"""
        if self.train_dataset is None:
            raise ValueError("Training dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


if __name__ == "__main__":
    # Test the loader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--preprocessed_ecg", type=str, required=True)
    parser.add_argument("--ecg_phenotypes_path", type=str, required=True)
    args = parser.parse_args()

    print("=== Testing MotionDataModulePreprocessed ===")

    data_module = MotionDataModulePreprocessed(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        batch_size=32,
        num_workers=0,
    )

    data_module.setup("fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Test loading one batch
    print("\nTesting batch loading...")
    for batch in train_loader:
        print(f"Batch demographics shape: {batch['demographics'].shape}")
        print(f"Batch ECG raw shape: {batch['ecg_raw'].shape}")
        print(f"Batch ECG morphology shape: {batch['ecg_morphology'].shape}")
        print(f"Batch motion latent shape: {batch['motion_latent'].shape}")
        print(f"Batch EIDs: {batch['eid'][:5]}")
        print(f"\nECG raw stats:")
        print(f"  Mean: {batch['ecg_raw'].mean():.4f}")
        print(f"  Std: {batch['ecg_raw'].std():.4f}")
        print(f"  Min: {batch['ecg_raw'].min():.4f}")
        print(f"  Max: {batch['ecg_raw'].max():.4f}")
        print(f"\nECG morphology stats:")
        print(f"  Mean: {batch['ecg_morphology'].mean():.4f}")
        print(f"  Std: {batch['ecg_morphology'].std():.4f}")
        print(f"  Min: {batch['ecg_morphology'].min():.4f}")
        print(f"  Max: {batch['ecg_morphology'].max():.4f}")
        break

    print("\n✓ Dataloader test completed!")
