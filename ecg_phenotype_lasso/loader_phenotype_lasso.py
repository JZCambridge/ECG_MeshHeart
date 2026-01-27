#!/usr/bin/env python
"""
Data loader for LASSO-based ECG phenotype classification
Loads ONLY 16 ECG morphology features (no raw ECG, no demographics)

Adapted from echonext_binary_prediction for LASSO regression
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhenotypeLassoDataModule:
    """
    Simplified data module for LASSO regression
    Loads only ECG morphology features (16 features), no raw ECG or demographics
    """

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        ecg_phenotypes_path: str,
        label_column: str = 'diseased',
        threshold: Optional[float] = None,
        threshold_direction: str = 'less_than',
    ):
        """
        Initialize data module for LASSO training

        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            ecg_phenotypes_path: Path to CSV file with ECG morphology phenotypes
            label_column: Name of label column (can be binary or continuous)
            threshold: Optional threshold for converting continuous to binary
            threshold_direction: Direction for threshold comparison ('less_than' or 'greater_than')
        """
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.label_column = label_column
        self.threshold = threshold
        self.threshold_direction = threshold_direction

        # Define morphology feature columns
        self.morphology_cols = [
            'VentricularRate', 'PQInterval', 'PDuration', 'QRSDuration',
            'QTInterval', 'QTCInterval', 'RRInterval', 'PPInterval',
            'PAxis', 'RAxis', 'TAxis', 'POnset', 'POffset',
            'QOnset', 'QOffset', 'TOffset'
        ]

        # Define atrial-related features (fill with 0 for atrial arrhythmia)
        self.atrial_features = ['PQInterval', 'PDuration', 'PPInterval', 'PAxis', 'POnset', 'POffset']

        # Storage for prepared data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.eids_train = None
        self.eids_val = None
        self.morphology_scaler = None

    def _build_eid_instance_mapping(self, df_morphology: pd.DataFrame) -> Dict[int, Optional[int]]:
        """
        Build mapping from EID → instance with priority: 2 > 3 > others

        Args:
            df_morphology: Morphology dataframe with eid_40616 and Instance columns

        Returns:
            Dict mapping eid_40616 → best instance
        """
        eid_instances = {}

        # Collect all (eid, instance) pairs from morphology data
        for _, row in df_morphology.iterrows():
            eid = int(row['eid_40616'])
            instance = int(row['Instance'])
            if eid not in eid_instances:
                eid_instances[eid] = []
            eid_instances[eid].append(instance)

        # For each EID, pick the best instance
        eid_to_instance = {}
        for eid, instances in eid_instances.items():
            # Priority: instance 2 > 3 > others (lowest)
            if 2 in instances:
                eid_to_instance[eid] = 2
            elif 3 in instances:
                eid_to_instance[eid] = 3
            else:
                eid_to_instance[eid] = min(instances)  # Take lowest instance

        return eid_to_instance

    def _build_morphology_lookup(self, df_morphology: pd.DataFrame) -> Dict:
        """
        Build lookup dictionary: (eid_40616, instance) → morphology array [16]

        Args:
            df_morphology: Morphology dataframe

        Returns:
            Dict mapping (eid, instance) → numpy array of morphology features
        """
        morphology_lookup = {}

        # Special handling: Fill atrial-related features with 0 (for AF/atrial arrhythmia)
        df_morph = df_morphology.copy()
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

    def _process_labels(self, df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        """
        Process labels: apply threshold if needed, convert to binary

        Args:
            df: DataFrame with label column
            dataset_name: Name for logging (e.g., 'train', 'val')

        Returns:
            Binary labels as numpy array (0 or 1)
        """
        # Validate label column exists
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in {dataset_name} CSV. "
                           f"Available columns: {list(df.columns)}")

        # Extract raw label values
        raw_labels = df[self.label_column].values.astype(np.float32)

        # Convert to binary labels
        if self.threshold is not None:
            # Continuous variable: apply threshold with configurable direction
            logger.info(f"Converting continuous variable '{self.label_column}' to binary using threshold={self.threshold}")

            if self.threshold_direction == 'less_than':
                logger.info(f"  Rule: {self.label_column} < {self.threshold} → diseased (1), >= {self.threshold} → healthy (0)")
                logger.info(f"  Use case: LVEF (low values indicate disease)")
            elif self.threshold_direction == 'greater_than':
                logger.info(f"  Rule: {self.label_column} > {self.threshold} → diseased (1), <= {self.threshold} → healthy (0)")
                logger.info(f"  Use case: WT_MAX (high values indicate disease)")
            else:
                raise ValueError(f"Invalid threshold_direction: {self.threshold_direction}. "
                               f"Must be 'less_than' or 'greater_than'.")

            # Show statistics before conversion
            valid_values = raw_labels[~np.isnan(raw_labels)]
            logger.info(f"  {dataset_name} - Raw values - Min: {np.min(valid_values):.2f}, Max: {np.max(valid_values):.2f}, "
                       f"Mean: {np.mean(valid_values):.2f}, Median: {np.median(valid_values):.2f}")

            # Apply threshold based on direction
            if self.threshold_direction == 'less_than':
                labels = (raw_labels < self.threshold).astype(np.float32)
            else:  # greater_than
                labels = (raw_labels > self.threshold).astype(np.float32)

            # Handle NaN values (set to healthy by default)
            if np.isnan(raw_labels).any():
                logger.warning(f"Found {np.isnan(raw_labels).sum()} missing values in '{self.label_column}', filling with 0 (healthy)")
                labels[np.isnan(raw_labels)] = 0.0
        else:
            # Binary variable: validate values are already 0 or 1
            logger.info(f"Using binary variable '{self.label_column}' (expecting values 0 or 1)")
            labels = raw_labels

            # Validate labels are binary (0 or 1)
            unique_labels = np.unique(labels[~np.isnan(labels)])
            if not np.all(np.isin(unique_labels, [0, 1])):
                raise ValueError(f"Labels must be 0 or 1 when no threshold provided, found: {unique_labels}. "
                               f"Use --threshold argument if this is a continuous variable.")

            # Handle missing labels (convert NaN to 0)
            if np.isnan(labels).any():
                logger.warning(f"Found {np.isnan(labels).sum()} missing labels, filling with 0 (healthy)")
                labels = np.nan_to_num(labels, nan=0.0)

        # Log class distribution
        n_positive = np.sum(labels == 1)
        n_negative = np.sum(labels == 0)
        logger.info(f"✅ {dataset_name} - Class distribution: Healthy (0)={n_negative}, Diseased (1)={n_positive} "
                   f"(diseased ratio={n_positive/(n_positive+n_negative):.2%})")

        return labels

    def _load_morphology_features(self, df: pd.DataFrame, df_morphology: pd.DataFrame,
                                   eid_to_instance: Dict, morphology_lookup: Dict,
                                   dataset_name: str) -> np.ndarray:
        """
        Load morphology features for all samples in dataframe

        Args:
            df: DataFrame with patient data (must have eid_40616 column)
            df_morphology: Morphology features dataframe
            eid_to_instance: Mapping from eid to instance
            morphology_lookup: Lookup dictionary for morphology features
            dataset_name: Name for logging

        Returns:
            Morphology features array [N, 16]
        """
        # Get eid_40616 column
        if 'eid_40616' in df.columns:
            eids_40616 = df['eid_40616'].values.astype(int)
        else:
            raise ValueError("'eid_40616' column not found in CSV")

        # Load morphology features for all samples
        logger.info(f"Loading morphology features for {dataset_name} samples...")
        morphology_features = []
        available_count = 0

        for eid_40616 in eids_40616:
            instance = eid_to_instance.get(eid_40616)
            if instance is not None:
                key = (eid_40616, instance)
                morph = morphology_lookup.get(key, np.full(16, np.nan, dtype=np.float32))
                if not np.all(np.isnan(morph)):
                    available_count += 1
            else:
                morph = np.full(16, np.nan, dtype=np.float32)
            morphology_features.append(morph)

        morphology_features = np.array(morphology_features, dtype=np.float32)  # [N, 16]

        logger.info(f"✅ {dataset_name} - Morphology coverage: {available_count}/{len(eids_40616)} samples "
                   f"({100*available_count/len(eids_40616):.1f}%)")

        return morphology_features

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess all data

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, eids_train, eids_val)
            where X is [N, 16] morphology features, y is [N] binary labels
        """
        logger.info("\n=== Loading ECG Morphology Phenotypes ===")
        df_morphology = pd.read_csv(self.ecg_phenotypes_path)
        logger.info(f"✅ Loaded {len(df_morphology)} ECG morphology samples")

        # Build EID → instance mapping
        logger.info("Building EID → instance mapping...")
        eid_to_instance = self._build_eid_instance_mapping(df_morphology)

        # Build morphology lookup
        logger.info("Building morphology lookup...")
        morphology_lookup = self._build_morphology_lookup(df_morphology)

        # Load training data
        logger.info("\n=== Loading Training Data ===")
        df_train = pd.read_csv(self.train_csv)
        logger.info(f"Loaded {len(df_train)} training samples")

        # Extract EIDs
        if 'eid_18545' in df_train.columns:
            self.eids_train = df_train['eid_18545'].values.astype(int)
        else:
            raise ValueError("'eid_18545' column not found in training CSV")

        # Process labels
        self.y_train = self._process_labels(df_train, 'train')

        # Load morphology features
        X_train_raw = self._load_morphology_features(
            df_train, df_morphology, eid_to_instance, morphology_lookup, 'train'
        )

        # Load validation data
        logger.info("\n=== Loading Validation Data ===")
        df_val = pd.read_csv(self.val_csv)
        logger.info(f"Loaded {len(df_val)} validation samples")

        # Extract EIDs
        if 'eid_18545' in df_val.columns:
            self.eids_val = df_val['eid_18545'].values.astype(int)
        else:
            raise ValueError("'eid_18545' column not found in validation CSV")

        # Process labels
        self.y_val = self._process_labels(df_val, 'val')

        # Load morphology features
        X_val_raw = self._load_morphology_features(
            df_val, df_morphology, eid_to_instance, morphology_lookup, 'val'
        )

        # Preprocess morphology features (StandardScaler + SimpleImputer)
        logger.info("\n=== Preprocessing Morphology Features ===")
        self.morphology_scaler = Pipeline([
            ('scale', StandardScaler()),
            ('impute', SimpleImputer(strategy='median'))
        ])

        # Fit on training data
        self.X_train = self.morphology_scaler.fit_transform(X_train_raw)
        logger.info("✅ Fitted morphology scaler (StandardScaler + MedianImputer) on training data")

        # Transform validation data
        self.X_val = self.morphology_scaler.transform(X_val_raw)
        logger.info("✅ Applied morphology scaler to validation data")

        logger.info(f"\nFinal shapes:")
        logger.info(f"  X_train: {self.X_train.shape}")
        logger.info(f"  y_train: {self.y_train.shape}")
        logger.info(f"  X_val: {self.X_val.shape}")
        logger.info(f"  y_val: {self.y_val.shape}")

        return self.X_train, self.y_train, self.X_val, self.y_val, self.eids_train, self.eids_val

    def get_feature_names(self):
        """Return list of 16 morphology feature names"""
        return self.morphology_cols

    def get_scaler(self):
        """Return fitted morphology scaler"""
        return self.morphology_scaler


if __name__ == "__main__":
    # Test the loader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--ecg_phenotypes_path", type=str, required=True)
    parser.add_argument("--label_column", type=str, default="diseased")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold_direction", type=str, default="less_than",
                       choices=['less_than', 'greater_than'])
    args = parser.parse_args()

    print("=== Testing PhenotypeLassoDataModule ===")

    data_module = PhenotypeLassoDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=args.label_column,
        threshold=args.threshold,
        threshold_direction=args.threshold_direction,
    )

    X_train, y_train, X_val, y_val, eids_train, eids_val = data_module.prepare_data()

    print(f"\n=== Data Preparation Summary ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Feature names: {data_module.get_feature_names()}")
    print(f"\nTraining data stats:")
    print(f"  X_train - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    print(f"  y_train - Diseased: {y_train.sum()}/{len(y_train)} ({100*y_train.mean():.1f}%)")
    print(f"\nValidation data stats:")
    print(f"  X_val - Mean: {X_val.mean():.4f}, Std: {X_val.std():.4f}")
    print(f"  y_val - Diseased: {y_val.sum()}/{len(y_val)} ({100*y_val.mean():.1f}%)")

    print("\n✓ Data loader test completed!")
