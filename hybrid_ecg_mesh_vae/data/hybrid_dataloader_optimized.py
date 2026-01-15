#!/usr/bin/env python
"""
Optimized Hybrid ECG-Mesh DataLoader with Shared Memory Support

This dataloader is optimized for multi-worker training by:
1. Loading large ECG data (30GB+) and morphology tables ONCE into shared memory
2. Allowing multiple workers to access the shared data without redundant copies
3. Only loading per-patient mesh files in each worker (small, distributed files)

Key optimization: Reduces memory usage from (30GB × num_workers) to just 30GB total.
"""

import logging
import os
from typing import Dict, Optional
import multiprocessing as mp

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

# Global variables for shared memory access across workers
_SHARED_ECG_TENSOR = None  # Consolidated tensor: [N, 12, 2500]
_SHARED_ECG_KEY_TO_IDX = None  # Mapping: {(eid, instance): index}
_SHARED_MORPHOLOGY_DATA = None
_SHARED_DEMOGRAPHICS_DATA = None


def _worker_init_fn(worker_id):
    """
    Worker initialization function to reuse shared memory.
    This prevents each worker from loading the large ECG data separately.
    """
    global _SHARED_ECG_TENSOR, _SHARED_ECG_KEY_TO_IDX, _SHARED_MORPHOLOGY_DATA, _SHARED_DEMOGRAPHICS_DATA

    # Workers inherit the shared memory references from the main process
    # No need to reload anything here - just log that worker is ready
    logger.info(f"Worker {worker_id} initialized with shared memory access")


class OptimizedHybridECGMeshDataset(Dataset):
    """
    Memory-optimized dataset that uses shared memory for large ECG data.

    Architecture:
    - Main process: Loads ECG data (30GB), morphology, demographics into shared memory
    - Worker processes: Access shared memory directly, only load per-patient mesh files

    Returns for each sample:
        - ECG raw: [12, 2500] - 12-lead ECG signals (from shared memory)
        - Demographics: [9] - Age, Sex, Weight, Height, BP, BMI, BSA, Ethnicity (from shared memory)
        - ECG morphology: [16] - ECG phenotype features (from shared memory)
        - Mesh vertices: [seq_len, n_points, 3] - Cardiac mesh sequence (loaded per-sample)
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
        use_shared_memory: bool = True,
    ):
        """
        Initialize optimized hybrid dataset with shared memory support.

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
            use_shared_memory: Whether to use shared memory optimization (default: True)
        """
        global _SHARED_ECG_TENSOR, _SHARED_ECG_KEY_TO_IDX, _SHARED_MORPHOLOGY_DATA, _SHARED_DEMOGRAPHICS_DATA

        self.csv_path = csv_path
        self.preprocessed_ecg_path = preprocessed_ecg_path
        self.ecg_phenotypes_path = ecg_phenotypes_path
        self.target_seg_dir = target_seg_dir
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.surf_type = surf_type
        self.normalize = normalize
        self.is_train = is_train
        self.use_shared_memory = use_shared_memory

        # ECG parameters
        self.ecg_num_leads = 12
        self.ecg_num_timepoints = 2500

        logger.info(f"🔧 Initializing Optimized Hybrid ECG-Mesh Dataset ({'train' if is_train else 'val/test'})")
        logger.info(f"   Shared memory mode: {'ENABLED' if use_shared_memory else 'DISABLED'}")
        logger.info(f"   CSV path: {csv_path}")
        logger.info(f"   Preprocessed ECG path: {preprocessed_ecg_path}")
        logger.info(f"   ECG phenotypes path: {ecg_phenotypes_path}")
        logger.info(f"   Mesh directory: {target_seg_dir}")

        # Check if we're in the main process (should load data) or worker process (should reuse)
        is_main_process = _SHARED_ECG_TENSOR is None

        if is_main_process and use_shared_memory:
            logger.info("📊 Main process: Loading data into shared memory...")

            # Load preprocessed ECG data
            logger.info(f"Loading preprocessed ECG data (this may take a while for large files)...")
            ecg_data = torch.load(preprocessed_ecg_path)
            logger.info(f"✅ Loaded {len(ecg_data)} ECG samples from disk")

            # Consolidate ECG data into single tensor to avoid "too many open files" error
            # Instead of calling .share_memory_() on each tensor (thousands of file descriptors),
            # we consolidate into ONE large tensor and share that (only 1 file descriptor)
            logger.info(f"Consolidating {len(ecg_data)} ECG samples into single tensor...")
            ecg_list = []
            key_to_idx = {}

            for idx, (key, tensor) in enumerate(ecg_data.items()):
                ecg_list.append(tensor)
                key_to_idx[key] = idx

                # Progress logging for large datasets
                if (idx + 1) % 10000 == 0:
                    logger.info(f"   Processed {idx + 1}/{len(ecg_data)} samples...")

            # Stack all ECG tensors into single consolidated tensor [N, 12, 2500]
            logger.info(f"Stacking tensors into consolidated array...")
            ecg_consolidated = torch.stack(ecg_list)
            logger.info(f"✅ Consolidated tensor shape: {ecg_consolidated.shape}")

            # NOW call share_memory_() only ONCE - this uses only 1 file descriptor!
            logger.info(f"Moving consolidated tensor to shared memory...")
            ecg_consolidated.share_memory_()

            _SHARED_ECG_TENSOR = ecg_consolidated
            _SHARED_ECG_KEY_TO_IDX = key_to_idx
            logger.info(f"✅ ECG data in shared memory (1 file descriptor instead of {len(ecg_data)})")

            # Load ECG morphology phenotypes
            logger.info(f"Loading ECG morphology phenotypes...")
            _SHARED_MORPHOLOGY_DATA = pd.read_csv(ecg_phenotypes_path)
            logger.info(f"✅ Loaded {len(_SHARED_MORPHOLOGY_DATA)} ECG morphology samples")

        elif not use_shared_memory:
            # Non-shared mode: load data normally for this instance (dictionary format)
            logger.info(f"Loading preprocessed ECG data (non-shared mode)...")
            self.ecg_data_dict = torch.load(preprocessed_ecg_path)
            logger.info(f"✅ Loaded {len(self.ecg_data_dict)} preprocessed ECG samples")

            logger.info(f"Loading ECG morphology phenotypes...")
            self.df_morphology = pd.read_csv(ecg_phenotypes_path)
            logger.info(f"✅ Loaded {len(self.df_morphology)} ECG morphology samples")
        else:
            # Worker process: reuse shared memory
            logger.info(f"Worker process: Reusing shared ECG data and morphology from main process")

        # Set references to shared or local data
        if use_shared_memory:
            self.ecg_tensor = _SHARED_ECG_TENSOR
            self.ecg_key_to_idx = _SHARED_ECG_KEY_TO_IDX
            self.df_morphology = _SHARED_MORPHOLOGY_DATA
        else:
            # For non-shared mode, keep dictionary format
            self.ecg_tensor = None
            self.ecg_key_to_idx = None

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
        demographics = self.df[self.demo_cols].values.astype(np.float32)
        demographics = np.nan_to_num(demographics, nan=0.0)

        if is_main_process and use_shared_memory:
            # Convert demographics to shared tensor
            demographics_tensor = torch.from_numpy(demographics)
            demographics_tensor.share_memory_()
            _SHARED_DEMOGRAPHICS_DATA = demographics_tensor
            self.demographics = _SHARED_DEMOGRAPHICS_DATA
            logger.info(f"✅ Demographics moved to shared memory")
        elif use_shared_memory:
            self.demographics = _SHARED_DEMOGRAPHICS_DATA
        else:
            self.demographics = demographics

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

        # Motion scaler extraction (for encoder-decoder denormalization)
        # Extract motion latent columns from CSV if available
        logger.info("Loading motion latent features for scaler...")

        # Try both column naming conventions: z_* (validation) and mesh_embed_* (training)
        motion_latent_cols = [col for col in self.df.columns if col.startswith('z_')]
        if len(motion_latent_cols) == 0:
            motion_latent_cols = [col for col in self.df.columns if col.startswith('mesh_embed_')]

        if len(motion_latent_cols) > 0:
            # Extract motion embeddings from CSV
            motion_embeddings = self.df[motion_latent_cols].values.astype(np.float32)
            logger.info(f"   Found {len(motion_latent_cols)} motion latent columns: {motion_latent_cols[:3]}...")

            if is_train:
                # Fit scaler on training motion latents
                self.motion_scaler = StandardScaler()
                self.motion_scaler.fit(motion_embeddings)
                logger.info(f"✅ Fitted motion_scaler on {len(motion_latent_cols)} latent dimensions")
                logger.info(f"   Motion scaler mean: [{self.motion_scaler.mean_[:5]}...]")
                logger.info(f"   Motion scaler std: [{self.motion_scaler.scale_[:5]}...]")
            else:
                # Use scaler from training dataset
                if motion_scaler is None:
                    logger.warning("⚠️  Motion scaler not provided for validation dataset - will be None")
                    self.motion_scaler = None
                else:
                    self.motion_scaler = motion_scaler
                    logger.info("✅ Using motion_scaler from training set")
        else:
            # No motion latent columns in CSV
            logger.warning(f"⚠️  No motion latent columns (z_* or mesh_embed_*) found in CSV - motion_scaler will be None")
            self.motion_scaler = None

        logger.info(f"📈 Final dataset statistics:")
        logger.info(f"   Demographics shape: {self.demographics.shape if isinstance(self.demographics, np.ndarray) else tuple(self.demographics.shape)}")
        logger.info(f"   Morphology features shape: {self.morphology_features.shape}")
        logger.info(f"   Total samples: {len(self.df)}")
        logger.info(f"   Mesh sequence length: {self.seq_len}")
        logger.info(f"   Mesh points: {self.n_samples}")

    def _build_eid_instance_mapping(self) -> Dict[int, Optional[int]]:
        """Build mapping from eid_40616 → instance with priority: 2 > 3 > others"""
        eid_to_instance = {}

        # Collect all (eid, instance) pairs from preprocessed data
        # Handle both shared memory (key_to_idx dict) and non-shared (ecg_data_dict) modes
        eid_instances = {}

        if self.use_shared_memory:
            # Shared memory mode: use key_to_idx mapping
            keys = self.ecg_key_to_idx.keys()
        else:
            # Non-shared mode: use ecg_data_dict
            keys = self.ecg_data_dict.keys()

        for eid, instance in keys:
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
        """Load preprocessed ECG for given eid_40616 from shared memory (or dict in non-shared mode)"""
        instance = self.eid_to_instance.get(eid)

        if instance is None:
            # No ECG available for this EID
            return torch.zeros((self.ecg_num_leads, self.ecg_num_timepoints), dtype=torch.float32)

        key = (eid, instance)

        if self.use_shared_memory:
            # Shared memory mode: look up index and retrieve from consolidated tensor
            idx = self.ecg_key_to_idx.get(key)
            if idx is not None:
                return self.ecg_tensor[idx]  # [12, 2500]
            else:
                logger.warning(f"ECG not found for key {key}")
                return torch.zeros((self.ecg_num_leads, self.ecg_num_timepoints), dtype=torch.float32)
        else:
            # Non-shared mode: direct dictionary lookup
            if key in self.ecg_data_dict:
                return self.ecg_data_dict[key]
            else:
                logger.warning(f"ECG not found for key {key}")
                return torch.zeros((self.ecg_num_leads, self.ecg_num_timepoints), dtype=torch.float32)

    def _load_mesh(self, subid: int):
        """Load mesh data for given eid_18545 (subject ID) - loaded per-sample in worker"""
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
        - ECG loaded via eid_40616[idx] (from shared memory)
        - Mesh loaded via eid_18545[idx] (per-worker loading)
        - Both indices correspond to same row in patient info CSV
        """
        # Get subject IDs for this sample
        eid_18545 = self.eids_18545[idx]  # Mesh subject ID
        eid_40616 = self.eids_40616[idx]  # ECG recording ID

        # Load ECG data via eid_40616 (from shared memory - no disk I/O!)
        ecg_preprocessed = self._load_ecg(eid_40616)

        # Load mesh data via eid_18545 (per-worker - small files, parallelizable)
        mesh_verts, mesh_faces, mesh_edges = self._load_mesh(eid_18545)

        # Get demographics (from shared memory)
        if isinstance(self.demographics, torch.Tensor):
            demographics = self.demographics[idx]
        else:
            demographics = torch.FloatTensor(self.demographics[idx])

        return {
            "demographics": demographics,  # [9] - from shared memory
            "ecg_raw": ecg_preprocessed,  # [12, 2500] - from shared memory
            "ecg_morphology": torch.FloatTensor(self.morphology_features[idx]),  # [16]
            "heart_v": mesh_verts,  # [seq_len, n_points, 3] - loaded per-worker
            "heart_f": mesh_faces,  # [seq_len, n_faces, 3]
            "heart_e": mesh_edges,  # [seq_len, 2, n_edges]
            "eid": eid_18545  # Subject ID for tracking
        }

    def get_scalers(self):
        """Return scalers for use with validation dataset"""
        return self.morphology_scaler, self.motion_scaler


class OptimizedHybridDataModule:
    """
    Memory-optimized data module for Hybrid ECG-Mesh dataset.
    Manages train/val datasets and dataloaders with shared memory support.
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
        use_shared_memory: bool = True,
    ):
        """Initialize optimized data module"""
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
        self.use_shared_memory = use_shared_memory

        self.train_dataset = None
        self.val_dataset = None
        self.morphology_scaler = None
        self.motion_scaler = None

        logger.info(f"🚀 Initializing OptimizedHybridDataModule")
        logger.info(f"   Shared memory: {'ENABLED' if use_shared_memory else 'DISABLED'}")
        logger.info(f"   Num workers: {num_workers}")
        logger.info(f"   Expected memory usage: ~30GB (shared) vs ~{30 * num_workers}GB (without optimization)")

    def setup(self, stage: str = "fit"):
        """Setup datasets for training"""
        if stage == "fit" or stage is None:
            print("\n=== Setting up optimized training dataset ===")
            self.train_dataset = OptimizedHybridECGMeshDataset(
                csv_path=self.train_csv_path,
                preprocessed_ecg_path=self.preprocessed_ecg_path,
                ecg_phenotypes_path=self.ecg_phenotypes_path,
                target_seg_dir=self.target_seg_dir,
                seq_len=self.seq_len,
                n_samples=self.n_samples,
                surf_type=self.surf_type,
                morphology_scaler=None,
                is_train=True,
                use_shared_memory=self.use_shared_memory,
            )

            # Store scalers from training data
            self.morphology_scaler, self.motion_scaler = self.train_dataset.get_scalers()

            print("\n=== Setting up optimized validation dataset ===")
            self.val_dataset = OptimizedHybridECGMeshDataset(
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
                use_shared_memory=self.use_shared_memory,
            )

    def train_dataloader(self):
        """Return training dataloader with shared memory worker init"""
        if self.train_dataset is None:
            raise ValueError("Training dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            worker_init_fn=_worker_init_fn if self.use_shared_memory else None,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive
        )

    def val_dataloader(self):
        """Return validation dataloader with shared memory worker init"""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_worker_init_fn if self.use_shared_memory else None,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive
        )

    def get_motion_scaler(self):
        """Return motion_scaler for use in model denormalization"""
        return self.motion_scaler
