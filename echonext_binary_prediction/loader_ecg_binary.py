#!/usr/bin/env python
"""
Fast ECG loader for binary disease prediction
Loads preprocessed 12×2500 ECG data + binary disease labels

Modified from echonext_motion_latent_generation for binary classification
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


class ECGBinaryDataset(Dataset):
    """Dataset for ECG-to-Binary Disease Prediction with preprocessed 12×2500 ECG data"""

    def __init__(
        self,
        csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        label_column: str = 'diseased',
        threshold: Optional[float] = None,
        threshold_direction: str = 'less_than',
        morphology_scaler: Optional[Pipeline] = None,
        is_train: bool = True,
    ):
        """
        Initialize dataset with preprocessed ECG data and binary labels

        Args:
            csv_path: Path to CSV with demographics and disease labels
            preprocessed_ecg_path: Path to .pt file with preprocessed ECGs
            ecg_phenotypes_path: Path to CSV file with ECG morphology phenotypes
            label_column: Name of label column (can be binary or continuous)
            threshold: Optional threshold for converting continuous to binary
            threshold_direction: Direction for threshold comparison ('less_than' or 'greater_than')
            morphology_scaler: Pipeline for ECG morphology features (from training set)
            is_train: Whether this is training data (fit scaler) or val/test (use scaler)
        """
        self.csv_path = csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.label_column = label_column
        self.threshold_direction = threshold_direction
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

        # Load CSV with demographics and disease labels
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

        # Define demographic column names
        self.demo_cols1 = ['Age', 'Sex', 'Weight', 'Height', 'DBP_at_MRI', 'SBP_at_MRI', 'BMI', 'BSA']
        self.demo_cols2 = ['age_at_MRI', 'Sex', 'weight', 'height', 'DBP_avg', 'SBP_avg', 'BMI_img', 'BSA']

        # Check which columns exist and fill missing ones with constants
        missing_cols = []
        self.demo_cols = []
        for col1, col2 in zip(self.demo_cols1, self.demo_cols2):
            find_col = False
            if col1 in self.df.columns:
                self.demo_cols.append(col1)
                find_col = True
            elif col2 in self.df.columns:
                self.demo_cols.append(col2)
                find_col = True
                
            if not find_col: 
                missing_cols.append(col1)
                # Fill missing column with constant value (0.0)
                self.df[col1] = 0.0
                logger.warning(f"Column '{col1}' missing from dataset - filled with constant value 0.0")

        if missing_cols:
            logger.warning(f"Missing demographic columns (filled with 0.0): {missing_cols}")

        # Extract demographics (NO normalization)
        self.demographics = self.df[self.demo_cols].values.astype(np.float32)

        # Handle missing values
        self.demographics = np.nan_to_num(self.demographics, nan=0.0)

        # Validate label column exists
        if label_column not in self.df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV. "
                           f"Available columns: {list(self.df.columns)}")

        # Extract raw label values
        raw_labels = self.df[label_column].values.astype(np.float32)

        # Convert to binary labels
        if threshold is not None:
            # Continuous variable: apply threshold with configurable direction
            logger.info(f"Converting continuous variable '{label_column}' to binary using threshold={threshold}")

            if self.threshold_direction == 'less_than':
                logger.info(f"  Rule: {label_column} < {threshold} → diseased (1), >= {threshold} → healthy (0)")
                logger.info(f"  Use case: LVEF (low values indicate disease)")
            elif self.threshold_direction == 'greater_than':
                logger.info(f"  Rule: {label_column} > {threshold} → diseased (1), <= {threshold} → healthy (0)")
                logger.info(f"  Use case: WT_MAX (high values indicate disease)")
            else:
                raise ValueError(f"Invalid threshold_direction: {self.threshold_direction}. "
                               f"Must be 'less_than' or 'greater_than'.")

            # Show statistics before conversion
            valid_values = raw_labels[~np.isnan(raw_labels)]
            logger.info(f"  Raw values - Min: {np.min(valid_values):.2f}, Max: {np.max(valid_values):.2f}, "
                       f"Mean: {np.mean(valid_values):.2f}, Median: {np.median(valid_values):.2f}")

            # Apply threshold based on direction
            if self.threshold_direction == 'less_than':
                self.disease_labels = (raw_labels < threshold).astype(np.float32)
            else:  # greater_than
                self.disease_labels = (raw_labels > threshold).astype(np.float32)

            # Handle NaN values (set to healthy by default)
            if np.isnan(raw_labels).any():
                logger.warning(f"Found {np.isnan(raw_labels).sum()} missing values in '{label_column}', filling with 0 (healthy)")
                self.disease_labels[np.isnan(raw_labels)] = 0.0
        else:
            # Binary variable: validate values are already 0 or 1
            logger.info(f"Using binary variable '{label_column}' (expecting values 0 or 1)")
            self.disease_labels = raw_labels

            # Validate labels are binary (0 or 1)
            unique_labels = np.unique(self.disease_labels[~np.isnan(self.disease_labels)])
            if not np.all(np.isin(unique_labels, [0, 1])):
                raise ValueError(f"Labels must be 0 or 1 when no threshold provided, found: {unique_labels}. "
                               f"Use --threshold argument if this is a continuous variable.")

            # Handle missing labels (convert NaN to 0)
            if np.isnan(self.disease_labels).any():
                logger.warning(f"Found {np.isnan(self.disease_labels).sum()} missing labels, filling with 0 (healthy)")
                self.disease_labels = np.nan_to_num(self.disease_labels, nan=0.0)

        # Log class distribution
        n_positive = np.sum(self.disease_labels == 1)
        n_negative = np.sum(self.disease_labels == 0)
        logger.info(f"✅ Class distribution: Healthy (0)={n_negative}, Diseased (1)={n_positive} "
                   f"(diseased ratio={n_positive/(n_positive+n_negative):.2%})")

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
            logger.info("✅ Fitted morphology scaler (StandardScaler + MedianImputer) on training data")
        else:
            # Use provided scaler for validation/test
            if morphology_scaler is None:
                raise ValueError("Morphology scaler must be provided for non-training datasets")

            self.morphology_scaler = morphology_scaler
            self.morphology_features = self.morphology_scaler.transform(self.morphology_features)
            logger.info("✅ Applied morphology scaler from training set")

        logger.info(f"Demographics shape: {self.demographics.shape}")
        logger.info(f"Morphology features shape: {self.morphology_features.shape}")
        logger.info(f"Disease labels shape: {self.disease_labels.shape}")

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
                demographics: [8] tensor
                ecg_raw: [12, 2500] tensor (preprocessed)
                ecg_morphology: [16] tensor
                disease_label: scalar float (0 or 1)
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
            "disease_label": torch.tensor(self.disease_labels[idx], dtype=torch.float32),  # scalar
            "eid": eid
        }

    def get_scalers(self):
        """Return only morphology scaler (no motion scaler needed for classification)"""
        return self.morphology_scaler

    def get_eids(self):
        """Return the EID values for all samples"""
        return self.eids


class BinaryDataModule:
    """
    Data module for ECG Binary Disease Prediction with preprocessed ECG data.
    Manages train/val datasets and dataloaders.
    """

    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        label_column: str = 'diseased',
        threshold: Optional[float] = None,
        threshold_direction: str = 'less_than',
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
            label_column: Name of label column (binary or continuous)
            threshold: Optional threshold for converting continuous to binary
            threshold_direction: Direction for threshold comparison ('less_than' or 'greater_than')
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
        """
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.label_column = label_column
        self.threshold = threshold
        self.threshold_direction = threshold_direction
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.morphology_scaler = None

    def setup(self, stage: str = "fit"):
        """Setup datasets for training"""
        if stage == "fit" or stage is None:
            print("\n=== Setting up training dataset ===")
            self.train_dataset = ECGBinaryDataset(
                csv_path=self.train_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                label_column=self.label_column,
                threshold=self.threshold,
                threshold_direction=self.threshold_direction,
                morphology_scaler=None,
                is_train=True,
            )

            # Store scaler from training data (only morphology, no motion scaler)
            self.morphology_scaler = self.train_dataset.get_scalers()

            print("\n=== Setting up validation dataset ===")
            self.val_dataset = ECGBinaryDataset(
                csv_path=self.val_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                label_column=self.label_column,
                threshold=self.threshold,
                threshold_direction=self.threshold_direction,
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
    parser.add_argument("--label_column", type=str, default="diseased")
    args = parser.parse_args()

    print("=== Testing BinaryDataModule ===")

    data_module = BinaryDataModule(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        preprocessed_ecg_path=args.preprocessed_ecg,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=args.label_column,
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
        print(f"Batch disease label shape: {batch['disease_label'].shape}")
        print(f"Batch EIDs: {batch['eid'][:5]}")
        print(f"\nECG raw stats:")
        print(f"  Mean: {batch['ecg_raw'].mean():.4f}")
        print(f"  Std: {batch['ecg_raw'].std():.4f}")
        print(f"  Min: {batch['ecg_raw'].min():.4f}")
        print(f"  Max: {batch['ecg_raw'].max():.4f}")
        print(f"\nDisease label stats:")
        print(f"  Unique values: {torch.unique(batch['disease_label'])}")
        print(f"  Diseased (1): {torch.sum(batch['disease_label'] == 1).item()}")
        print(f"  Healthy (0): {torch.sum(batch['disease_label'] == 0).item()}")
        break

    print("\n✓ Dataloader test completed!")
