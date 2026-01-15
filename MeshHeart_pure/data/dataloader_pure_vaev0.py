from torch.utils.data import Dataset
import torch
import numpy as np
import csv
import pandas as pd
import pyvista as pv
import h5py
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

class PureGeometricMesh(Dataset):
    """
    Pure geometric mesh dataset that loads only mesh data without any conditioning.
    
    This is a simplified version of UKbiobankMesh that focuses entirely on geometric
    learning. All patient metadata (age, sex, weight, height) and ECG embeddings 
    have been removed to create a pure transformer VAE that learns only from 
    geometric patterns.
    
    Why remove conditioning?
    1. Simplifies the learning objective to focus on geometric patterns
    2. Reduces model complexity and potential overfitting to patient metadata
    3. Creates a more generalizable model that captures pure cardiac motion
    4. Enables studying geometric variations independent of patient characteristics
    """

    def __init__(self, config, data_usage='train'):
        """
        Initialize dataset with only geometric data loading.
        
        Key changes from original UKbiobankMesh:
        - Removed use_ecg parameter and all ECG processing
        - Removed patient metadata loading (age, sex, weight, height)
        - Simplified data_list to only contain subject IDs
        - Focus purely on mesh geometry loading
        """
        
        data_list = []
        csvpath = f"{config.label_dir}/mesh_{data_usage}.csv"
        
        print(f"🔧 Loading {data_usage} dataset in PURE GEOMETRIC mode")
        print(f"   CSV path: {csvpath}")
        print(f"   Loading ONLY subject IDs - no patient conditions or ECG data")
        
        # Simplified CSV reading - only extract subject IDs
        try:
            df = pd.read_csv(csvpath)
            print(f"   Found {len(df)} samples in dataset")
            
            # Only store subject IDs for mesh loading
            for _, row in df.iterrows():
                subid = row['Unnamed: 0']  # Subject ID for mesh file lookup
                data_list.append(subid)
                
            print(f"✅ Successfully loaded {len(data_list)} subject IDs")
                        
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            # Fallback to simple CSV reading
            with open(csvpath, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data_list.append(row['Unnamed: 0'])

        self.data_list = data_list
        self.data_dir = config.target_seg_dir
        self.device = config.device
        self.seq_len = config.seq_len
        self.label_dir = config.label_dir
        self.normalize = config.normalize
        self.n_samples = config.n_samples
        self.surf_type = config.surf_type
        
        print(f"📊 Dataset configuration:")
        print(f"   Sequence length: {self.seq_len}")
        print(f"   Points per mesh: {self.n_samples}")
        print(f"   Surface type: {self.surf_type}")
        print(f"   Total samples: {len(self.data_list)}")

    def __getitem__(self, index):
        """
        Get a single sample containing only geometric mesh data.
        
        Returns:
            - mesh_verts: Vertex coordinates [seq_len, n_points, 3]
            - mesh_faces: Face connectivity [seq_len, n_faces, 3] 
            - mesh_edges: Edge connectivity [seq_len, 2, n_edges]
            - subid: Subject ID for tracking/debugging
            
        Key changes from original:
        - REMOVED: All patient condition processing (age, sex, weight, height)
        - REMOVED: All ECG embedding loading and processing
        - REMOVED: Condition normalization and transformation
        - Focus purely on geometric data loading
        """
        subid = self.data_list[index]
        subid = int(subid)  # Ensure subid is an integer
        
        # Load mesh data (this part remains the same as it's pure geometry)
        data_dir = self.data_dir
        mesh_path = f"{data_dir}/{subid}/image_space_pipemesh"
        
        # Select mesh resolution based on surface type
        if self.surf_type == 'all':
            h5filepath = f'{mesh_path}/preprossed_vtk.hdf5'
        elif self.surf_type == 'sample':
            h5filepath = f'{mesh_path}/preprossed_decimate.hdf5'
            
        # Load mesh geometry from HDF5 file
        f = h5py.File(h5filepath, "r")
        mesh_verts = torch.Tensor(np.array(f['heart_v']))  # Vertex coordinates
        mesh_faces = torch.LongTensor(np.array(f['heart_f']))  # Face connectivity
        mesh_edges = torch.LongTensor(np.array(f['heart_e']))  # Edge connectivity
        
        # Return only geometric data - no conditions!
        return (mesh_verts, mesh_faces, mesh_edges, subid)

    def __len__(self):
        return len(self.data_list)

# Create alias for backward compatibility with existing training scripts
UKbiobankMesh = PureGeometricMesh