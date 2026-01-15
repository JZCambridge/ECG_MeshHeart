import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.data import Batch
from pytorch3d.structures import Meshes
import trimesh
import pyvista as pv
from torch_geometric.data import Data, Batch

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequences.
    This adds position information to help the transformer understand sequence order.
    """
    def __init__(self, d_model, dropout=0.1, max_len=60):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class Encoder_TRANSFORMER(nn.Module):
    """
    Pure transformer encoder that processes mesh sequences without any conditioning.
    
    Key changes from original:
    1. Removed all condition-related parameters (c_dim removed)
    2. Removed condition concatenation in forward pass
    3. FC layer now only processes GCN features directly
    4. Simplified architecture focuses purely on geometric learning
    """
    def __init__(self, dim_in=3, points=10000, seq_len=50, z_dim=32, 
                 ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()

        self.points = points
        self.dim_in = dim_in
        self.num_frames = seq_len
        self.latent_dim = z_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_bias = True

        # GCN layers for processing mesh geometry
        # These extract local geometric features from vertices
        self.skelEmbedding = nn.Sequential(
            nn.Conv1d(in_channels=dim_in, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),
        )
        self.gcn1 = GCNConv(64, 128)
        self.gcn2 = GCNConv(128, 256)

        # REMOVED: Condition processing and concatenation
        # OLD: self.fc = nn.Sequential(nn.Linear(256 + c_dim, self.latent_dim, bias=True), ...)
        # NEW: Direct mapping from GCN features to latent space
        self.fc = nn.Sequential(
            nn.Linear(256, self.latent_dim, bias=True),  # Only GCN features, no conditions
            nn.ReLU(inplace=True)
        )

        # Learnable queries for mu and sigma estimation
        # These help the transformer learn to estimate distribution parameters
        self.muQuery = nn.Parameter(torch.randn(self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(self.latent_dim))

        # Transformer components for sequence modeling
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

    def forward(self, v, f, edge_list, debug=False):
        """
        Forward pass processes only geometric information.
        
        Args:
            v: Vertices [batch, seq_len, points, 3]
            f: Faces [batch, seq_len, points, 3] 
            edge_list: Edge connectivity [batch, seq_len, 2, edges]
            debug: Print intermediate shapes for debugging
            
        Key changes:
        - REMOVED: combined_con parameter (no more conditioning)
        - Process only geometric features through GCN + Transformer
        """
        batch_size = v.shape[0]
        se_length = v.shape[1]
        nodes = v.shape[2]
        
        # Reshape for GCN processing
        v = v.reshape(batch_size * se_length, nodes, -1).permute((0, 2, 1))
        if debug:
            print(f"v reshaped: {v.shape}")

        # Initial embedding of vertex features
        v = self.skelEmbedding(v)
        if debug:
            print(f"v after embedding: {v.shape}")
        
        v = v.permute((0, 2, 1)).reshape((batch_size, se_length, nodes, -1))
        edge_list = edge_list.permute((0, 1, 3, 2))

        # Process each time frame through GCN
        v_all = []
        for seq_len in range(se_length):
            data_list = []
            for b in range(batch_size):
                data = Data(x=v[b, seq_len], edge_index=edge_list[b, seq_len])
                data_list.append(data)
            batch = Batch.from_data_list(data_list)
            
            # Apply GCN layers to extract geometric features
            temp = F.leaky_relu(self.gcn1(batch.x, batch.edge_index), 0.15)
            temp = F.leaky_relu(self.gcn2(temp, batch.edge_index), 0.15).reshape(batch_size, 1, nodes, -1)
            v_all.append(temp)
            
        v = torch.cat(v_all, dim=1).permute((0, 1, 3, 2))
        if debug:
            print(f"v after GCN: {v.shape}")

        # Global max pooling across vertices to get frame-level features
        v = v.max(dim=3)[0]
        if debug:
            print(f"v after max pooling: {v.shape}")
        
        # REMOVED: Condition concatenation
        # OLD: combined_con = combined_con.unsqueeze(1).repeat(1, v.shape[1], 1)
        # OLD: x = self.fc(torch.cat((v, combined_con), dim=2))
        # NEW: Process only geometric features
        x = self.fc(v)
        if debug:
            print(f"x after fc: {v.shape}")

        # Prepare sequence for transformer with mu/sigma queries
        xseq = torch.cat((
            self.muQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
            self.sigmaQuery.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1), 
            x), axis=1).permute(1, 0, 2)
        
        # Apply positional encoding and transformer
        xseq = self.sequence_pos_encoder(xseq)
        xseq = self.seqTransEncoder(xseq)
        
        # Extract distribution parameters
        mu = xseq[0]     # Mean of latent distribution
        logvar = xseq[1] # Log variance of latent distribution
        
        return mu, logvar, xseq

class Decoder_TRANSFORMER(nn.Module):
    """
    Pure transformer decoder that generates mesh sequences from latent codes only.
    
    Key changes from original:
    1. Removed all condition-related parameters
    2. Removed condition concatenation with latent code
    3. Direct processing of latent codes through transformer
    4. Simplified architecture focuses on latent-to-geometry mapping
    """
    def __init__(self, dim_in=3, points=10000, seq_len=50, z_dim=32, 
                 ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"):
        super().__init__()

        self.njoints = points
        self.nfeats = dim_in
        self.num_frames = seq_len
        self.latent_dim = z_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.njoints * self.nfeats
        self.use_bias = True

        # REMOVED: Condition processing 
        # OLD: self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)
        # NEW: Direct latent processing (no condition concatenation needed)
        self.ztimelinear = nn.Linear(self.latent_dim, self.latent_dim)
        
        # Transformer components
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation
        )
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

        # Final layers to generate vertex coordinates
        # This maps from latent features back to 3D vertex positions
        self.finallayer = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=points * 3, bias=self.use_bias),  # Output 3D coordinates
        )

    def forward(self, z, debug=False):
        """
        Forward pass generates mesh sequence from latent code only.
        
        Args:
            z: Latent code [batch, latent_dim]
            debug: Print intermediate shapes
            
        Key changes:
        - REMOVED: combined_con parameter (no conditioning)
        - Direct processing of latent code without condition concatenation
        """
        bs = z.shape[0]
        nframes = self.num_frames
        njoints, nfeats = self.njoints, self.nfeats
        
        # REMOVED: Condition concatenation
        # OLD: z = torch.cat((z, combined_con), axis=1)
        # NEW: Process latent code directly
        z = self.ztimelinear(z)
        if debug:
            print(f"z shape after linear transformation: {z.shape}")
            
        z = z[None]  # Add sequence dimension
        if debug:
            print(f"z shape after adding sequence dimension: {z.shape}")

        # Create time queries for sequence generation
        timequeries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        if debug:
            print(f"timequeries shape: {timequeries.shape}")

        # Generate sequence through transformer decoder
        output_seq = self.seqTransDecoder(tgt=timequeries, memory=z)
        if debug:
            print(f"output_seq shape: {output_seq.shape}")

        # Convert to final vertex coordinates
        output = self.finallayer(torch.squeeze(output_seq, 1)).reshape(nframes, bs, njoints, nfeats)
        if debug:
            print(f"output shape: {output.shape}")
            
        return output.permute(1, 0, 2, 3), output_seq

class PureTransformerVAE(nn.Module):
    """
    Pure Transformer-based Variational Autoencoder for mesh sequences.
    
    This model learns to encode and decode mesh sequences using only geometric information,
    without any patient or ECG conditioning. It focuses purely on learning the underlying
    geometric patterns and variations in heart mesh dynamics.
    
    Key changes from original CAE:
    1. Removed all condition mapping MLPs
    2. Removed ECG processing entirely  
    3. Simplified encoder/decoder to process only geometric data
    4. Maintains VAE structure with reparameterization trick
    """
    def __init__(self, dim_in=3, dim_h=128, z_dim=32, points=15000, seq_len=50,
                 ff_size=1024, num_heads=4, activation="gelu", num_layers=4):
        super().__init__()
        self.latent_dim = z_dim
        
        print(f"🔧 Pure Transformer VAE Configuration:")
        print(f"   Input dimensions: {dim_in}")
        print(f"   Latent dimension: {z_dim}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Points per mesh: {points}")
        print(f"   NO CONDITIONING - Pure geometric learning")
            
        # Pure geometric encoder (no conditioning)
        self.encoder = Encoder_TRANSFORMER(
            dim_in=dim_in, 
            points=points,
            seq_len=seq_len, 
            z_dim=z_dim, 
            ff_size=ff_size,
            num_layers=num_layers, 
            num_heads=num_heads, 
            dropout=0.1,
            activation=activation
        )
        
        # Pure geometric decoder (no conditioning)
        self.decoder = Decoder_TRANSFORMER(
            dim_in=dim_in, 
            points=points,
            seq_len=seq_len, 
            z_dim=z_dim, 
            ff_size=ff_size,
            num_layers=num_layers, 
            num_heads=num_heads, 
            dropout=0.1,
            activation=activation
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE sampling.
        This allows gradients to flow through the stochastic latent variable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, v, f, edge_list, debug=False):
        """
        Forward pass for pure geometric VAE.
        
        Args:
            v: Vertices [batch, time_frames, points, 3]
            f: Faces [batch, time_frames, points, 3]
            edge_list: Edge connectivity
            debug: Print debug information
            
        Key changes:
        - REMOVED: All condition parameters (c, ecg_c)
        - Process only geometric inputs
        - Pure latent space learning without bias from patient data
        """
        if debug:
            print(f"🔍 Pure VAE Forward pass:")
            print(f"   Input vertices: {v.shape}")
            print(f"   Input faces: {f.shape}")
            print(f"   NO CONDITIONS - Pure geometric processing")
        
        # Encode geometry to latent distribution
        mu, logvar, xseq = self.encoder(v, f, edge_list, debug=debug)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode latent to geometry
        v_all, v_all_latent = self.decoder(z, debug=debug)
        
        return v_all, logvar, mu

    def encode(self, v, f, edge_list):
        """
        Encode mesh sequence to latent representation.
        Useful for analysis and interpolation experiments.
        """
        mu, logvar, xseq = self.encoder(v, f, edge_list)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        """
        Decode latent code to mesh sequence.
        Useful for generation from random latent codes.
        """
        v_all, v_all_latent = self.decoder(z)
        return v_all

    def generate_random(self, batch_size, device):
        """
        Generate random mesh sequences by sampling from prior.
        This demonstrates the generative capability of the pure VAE.
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z)

# Convenience alias for backward compatibility
CAE = PureTransformerVAE