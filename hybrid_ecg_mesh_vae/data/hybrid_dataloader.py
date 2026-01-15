#!/usr/bin/env python
"""
Hybrid ECG-Mesh DataLoader for ECG-to-Mesh Generation VAE

This dataloader merges:
1. ECG signal loading from EchoNext (12×2500 ECG + demographics + morphology)
2. Mesh sequence loading from MeshHeart (50 frames of cardiac mesh)

Key alignment: Ensures ECG and mesh data correspond to the same patient via:
- Primary key: eid_18545 (subject ID for mesh)
- ECG lookup: eid_40616 (ECG recording ID mapped from patient info)
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, Dataset
import h5py

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridECGMeshDataset(Dataset):
    """
    Dataset that combines ECG signals + patient info with mesh sequences.

    Returns for each sample:
        - ECG raw: [12, 2500] - 12-lead ECG signals
        - Demographics: [9] - Age, Sex, Weight, Height, BP, BMI, BSA, Ethnicity
        - ECG morphology: [16] - ECG phenotype features
        - Mesh vertices: [seq_len, n_points, 3] - Cardiac mesh sequence
        - Mesh faces: [seq_len, n_faces, 3] - Face connectivity
        - Mesh edges: [seq_len, 2, n_edges] - Edge connectivity
        - Subject ID: int - For tracking/debugging
    """

    def __init__(
        self,
        csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        target_seg_dir: str,
        seq_len: int = 50,
        n_samples: int = 1412,
        surf_type: str = 'all',
        normalize: bool = True,
        motion_scaler: Optional[StandardScaler] = None,
        morphology_scaler: Optional[Pipeline] = None,
        is_train: bool = True,
    ):
        """
        Initialize hybrid dataset.

        Args:
            csv_path: Path to CSV with patient info (has both eid_18545 and eid_40616)
            preprocessed_ecg_path: Path to .pt file with preprocessed ECGs
            ecg_phenotypes_path: Path to CSV with ECG morphology phenotypes
            target_seg_dir: Directory containing mesh HDF5 files
            seq_len: Sequence length for meshes
            n_samples: Number of mesh vertices
            surf_type: Mesh resolution ('all' or 'sample')
            normalize: Whether to normalize inputs
            motion_scaler: StandardScaler for motion (not used but kept for compatibility)
            morphology_scaler: Pipeline for morphology features (from training set)
            is_train: Whether this is training data
        """
        self.csv_path = csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.target_seg_dir = target_seg_dir
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.surf_type = surf_type
        self.normalize = normalize
        self.is_train = is_train

        # ECG parameters
        self.ecg_num_leads = 12
        self.ecg_num_timepoints = 2500

        logger.info(f"🔧 Initializing Hybrid ECG-Mesh Dataset ({'train' if is_train else 'val/test'})")
        logger.info(f"   CSV path: {csv_path}")
        logger.info(f"   Preprocessed ECG path: {preprocessed_ecg_path}")
        logger.info(f"   ECG phenotypes path: {ecg_phenotypes_path}")
        logger.info(f"   Mesh directory: {target_seg_dir}")

        # Load preprocessed ECG data
        logger.info(f"Loading preprocessed ECG data...")
        self.ecg_data = torch.load(preprocessed_ecg_path)
        logger.info(f"✅ Loaded {len(self.ecg_data)} preprocessed ECG samples")

        # Load ECG morphology phenotypes
        logger.info(f"Loading ECG morphology phenotypes...")
        self.df_morphology = pd.read_csv(ecg_phenotypes_path)
        logger.info(f"✅ Loaded {len(self.df_morphology)} ECG morphology samples")

        # Define morphology feature columns
        self.morphology_cols = [
            'VentricularRate', 'PQInterval', 'PDuration', 'QRSDuration',
            'QTInterval', 'QTCInterval', 'RRInterval', 'PPInterval',
            'PAxis', 'RAxis', 'TAxis', 'POnset', 'POffset',
            'QOnset', 'QOffset', 'TOffset'
        ]

        # Atrial-related features (fill with 0 for atrial arrhythmia)
        self.atrial_features = ['PQInterval', 'PDuration', 'PPInterval', 'PAxis', 'POnset', 'POffset']

        # Load main CSV with patient info
        logger.info(f"Loading patient info CSV...")
        self.df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded {len(self.df)} samples from CSV")

        # Extract EID columns for alignment
        if 'eid_18545' in self.df.columns:
            self.eids_18545 = self.df['eid_18545'].values.astype(int)
        else:
            raise ValueError("'eid_18545' column not found in CSV")

        if 'eid_40616' in self.df.columns:
            self.eids_40616 = self.df['eid_40616'].values.astype(int)
        else:
            raise ValueError("'eid_40616' column not found in CSV")

        logger.info(f"📊 Alignment info:")
        logger.info(f"   eid_18545 (mesh subject ID): {len(self.eids_18545)} entries")
        logger.info(f"   eid_40616 (ECG recording ID): {len(self.eids_40616)} entries")

        # Define demographic columns
        self.demo_cols = ['Age', 'Sex', 'Weight', 'Height', 'DBP_at_MRI', 'SBP_at_MRI', 'BMI', 'BSA', 'Ethnic_background']

        # Verify columns exist
        missing_cols = []
        for col in self.demo_cols:
            if col not in self.df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        # Extract demographics (NO normalization as per EchoNext design)
        self.demographics = self.df[self.demo_cols].values.astype(np.float32)
        self.demographics = np.nan_to_num(self.demographics, nan=0.0)

        # Build EID → instance mapping for ECG
        logger.info("Building EID → instance mapping for ECG...")
        self.eid_to_instance = self._build_eid_instance_mapping()

        # Build morphology lookup
        logger.info("Building morphology lookup...")
        self.morphology_lookup = self._build_morphology_lookup()

        # Check ECG coverage
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

        self.morphology_features = np.array(morphology_features, dtype=np.float32)

        # Morphology normalization
        if is_train:
            self.morphology_scaler = Pipeline([
                ('scale', StandardScaler()),
                ('impute', SimpleImputer(strategy='median'))
            ])
            self.morphology_features = self.morphology_scaler.fit_transform(self.morphology_features)
            logger.info("✅ Fitted morphology scaler on training data")
        else:
            if morphology_scaler is None:
                raise ValueError("Morphology scaler must be provided for non-training datasets")
            self.morphology_scaler = morphology_scaler
            self.morphology_features = self.morphology_scaler.transform(self.morphology_features)
            logger.info("✅ Applied morphology scaler from training set")
        
        # Load motion embeddings (mesh latent vectors)
        self.motion_cols = [f'mesh_embed_{i}' for i in range(1, 65)]
        self.motion_embeddings = self.df[self.motion_cols].values.astype(np.float32)
        self.motion_embeddings = np.nan_to_num(self.motion_embeddings, nan=0.0)
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
            
        
        logger.info(f"📈 Final dataset statistics:")
        logger.info(f"   Demographics shape: {self.demographics.shape}")
        logger.info(f"   Morphology features shape: {self.morphology_features.shape}")
        logger.info(f"   Total samples: {len(self.df)}")
        logger.info(f"   Mesh sequence length: {self.seq_len}")
        logger.info(f"   Mesh points: {self.n_samples}")

    def _build_eid_instance_mapping(self) -> Dict[int, Optional[int]]:
        """Build mapping from eid_40616 → instance with priority: 2 > 3 > others"""
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
                    eid_to_instance[eid] = min(instances)
            else:
                eid_to_instance[eid] = None

        return eid_to_instance

    def _build_morphology_lookup(self) -> Dict:
        """Build lookup dictionary: (eid_40616, instance) → morphology array [16]"""
        morphology_lookup = {}

        df_morph = self.df_morphology.copy()

        # Special handling: Fill atrial-related features with 0
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
                morph_values.append(float(val) if pd.notna(val) else np.nan)

            morphology_lookup[(eid, instance)] = np.array(morph_values, dtype=np.float32)

        logger.info(f"   Built morphology lookup with {len(morphology_lookup)} (eid, instance) pairs")
        return morphology_lookup

    def _load_ecg(self, eid: int) -> torch.Tensor:
        """Load preprocessed ECG for given eid_40616"""
        instance = self.eid_to_instance.get(eid)

        if instance is None:
            # No ECG available for this EID
            return torch.zeros((self.ecg_num_leads, self.ecg_num_timepoints), dtype=torch.float32)

        # Look up preprocessed ECG
        key = (eid, instance)
        if key in self.ecg_data:
            return self.ecg_data[key]
        else:
            logger.warning(f"ECG not found for key {key}")
            return torch.zeros((self.ecg_num_leads, self.ecg_num_timepoints), dtype=torch.float32)

    def _load_mesh(self, subid: int):
        """Load mesh data for given eid_18545 (subject ID)"""
        mesh_path = f"{self.target_seg_dir}/{subid}/image_space_pipemesh"

        # Select mesh resolution based on surface type
        if self.surf_type == 'all':
            h5filepath = f'{mesh_path}/preprossed_vtk.hdf5'
        elif self.surf_type == 'sample':
            h5filepath = f'{mesh_path}/preprossed_decimate.hdf5'

        try:
            # Load mesh geometry from HDF5 file
            f = h5py.File(h5filepath, "r")
            mesh_verts = torch.Tensor(np.array(f['heart_v']))
            mesh_faces = torch.LongTensor(np.array(f['heart_f']))
            mesh_edges = torch.LongTensor(np.array(f['heart_e']))
            f.close()

            # Check for nan/inf in loaded mesh data
            if np.isnan(mesh_verts.numpy()).any():
                logger.warning(f"Detected nan in mesh vertices for subid {subid}")
            if np.isinf(mesh_verts.numpy()).any():
                logger.warning(f"Detected inf in mesh vertices for subid {subid}")

            return mesh_verts, mesh_faces, mesh_edges
        except FileNotFoundError:
            logger.warning(f"No mesh found for subid {subid} - file not found")
            return (
                torch.zeros((self.seq_len, self.n_samples, 3), dtype=torch.float32),
                torch.zeros((self.seq_len, 100, 3), dtype=torch.long),
                torch.zeros((self.seq_len, 2, 100), dtype=torch.long)
            )
        except Exception as e:
            logger.error(f"No mesh found for subid {subid} - {str(e)}")
            return (
                torch.zeros((self.seq_len, self.n_samples, 3), dtype=torch.float32),
                torch.zeros((self.seq_len, 100, 3), dtype=torch.long),
                torch.zeros((self.seq_len, 2, 100), dtype=torch.long)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns dictionary with all modalities for a single patient.

        CRITICAL: Ensures ECG and mesh correspond to the same patient via:
        - ECG loaded via eid_40616[idx]
        - Mesh loaded via eid_18545[idx]
        - Both indices correspond to same row in patient info CSV
        """
        # Get subject IDs for this sample
        eid_18545 = self.eids_18545[idx]  # Mesh subject ID
        eid_40616 = self.eids_40616[idx]  # ECG recording ID

        # Load ECG data via eid_40616
        ecg_preprocessed = self._load_ecg(eid_40616)

        # Load mesh data via eid_18545
        mesh_verts, mesh_faces, mesh_edges = self._load_mesh(eid_18545)

        return {
            "demographics": torch.FloatTensor(self.demographics[idx]),  # [9]
            "ecg_raw": ecg_preprocessed,  # [12, 2500]
            "ecg_morphology": torch.FloatTensor(self.morphology_features[idx]),  # [16]
            "heart_v": mesh_verts,  # [seq_len, n_points, 3]
            "heart_f": mesh_faces,  # [seq_len, n_faces, 3]
            "heart_e": mesh_edges,  # [seq_len, 2, n_edges]
            "eid": eid_18545  # Subject ID for tracking
        }

    def get_motion_scaler(self):
        """Return motion scaler for use with validation dataset"""
        return self.motion_scaler
    
    def get_scalers(self):
        """Return scalers for use with validation dataset"""
        return self.morphology_scaler


class HybridDataModule:
    """
    Data module for Hybrid ECG-Mesh dataset.
    Manages train/val datasets and dataloaders.
    """

    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: str,
        preprocessed_ecg_path: str,
        ecg_phenotypes_path: str,
        target_seg_dir: str,
        seq_len: int = 50,
        n_samples: int = 1412,
        surf_type: str = 'all',
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """Initialize data module"""
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.target_seg_dir = target_seg_dir
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.surf_type = surf_type
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.morphology_scaler = None

    def setup(self, stage: str = "fit"):
        """Setup datasets for training"""
        if stage == "fit" or stage is None:
            print("\n=== Setting up training dataset ===")
            self.train_dataset = HybridECGMeshDataset(
                csv_path=self.train_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                target_seg_dir=self.target_seg_dir,
                seq_len=self.seq_len,
                n_samples=self.n_samples,
                surf_type=self.surf_type,
                morphology_scaler=None,
                is_train=True,
            )

            # Store scaler from training data
            self.morphology_scaler = self.train_dataset.get_scalers()
            self.motion_scaler = self.train_dataset.get_motion_scaler()

            print("\n=== Setting up validation dataset ===")
            self.val_dataset = HybridECGMeshDataset(
                csv_path=self.val_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                target_seg_dir=self.target_seg_dir,
                seq_len=self.seq_len,
                n_samples=self.n_samples,
                surf_type=self.surf_type,
                morphology_scaler=self.morphology_scaler,
                motion_scaler=self.motion_scaler,
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

    def get_motion_scaler(self):
        """Return motion_scaler for use in model denormalization"""
        return self.train_dataset.get_motion_scaler()