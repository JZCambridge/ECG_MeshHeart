"""
Mesh Decoder for ECG-to-Mesh Generation VAE

Extracted from MeshHeart's Transformer decoder.
Takes latent code z and generates mesh sequences (50 frames of cardiac motion).
"""

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer sequences.
    Adds position information to help the transformer understand sequence order.
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


class Decoder_TRANSFORMER(nn.Module):
    """
    Transformer decoder that generates mesh sequences from latent codes.

    Takes a latent code z [B, latent_dim] and generates a sequence of
    3D mesh vertices [B, seq_len, points, 3] representing cardiac motion.
    """

    def __init__(
        self,
        dim_in=3,
        points=1412,
        seq_len=50,
        z_dim=64,
        ff_size=1024,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        activation="gelu"
    ):
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

        # Direct latent processing (no condition concatenation)
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
        self.finallayer = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=points * 3, bias=self.use_bias),
        )

    def forward(self, z, debug=False):
        """
        Forward pass generates mesh sequence from latent code.

        Args:
            z: Latent code [B, latent_dim]
            debug: Print intermediate shapes

        Returns:
            output: Mesh vertices [B, seq_len, points, 3]
            output_seq: Latent sequence [seq_len, B, latent_dim]
        """
        bs = z.shape[0]
        nframes = self.num_frames
        njoints, nfeats = self.njoints, self.nfeats

        # Process latent code directly
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


class MeshDecoder(nn.Module):
    """
    Mesh Decoder wrapper for standalone use.

    Takes latent code and generates cardiac mesh sequences.

    Input:
        - z: [B, latent_dim] - Latent code

    Output:
        - v_out: [B, seq_len, points, 3] - Mesh vertex sequences
    """

    def __init__(
        self,
        latent_dim: int = 64,
        seq_len: int = 50,
        points: int = 1412,
        ff_size: int = 1024,
        num_layers: int = 2,
        num_heads: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.points = points

        print(f"🔧 Mesh Decoder Configuration:")
        print(f"   Latent dimension: {latent_dim}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Points per mesh: {points}")
        print(f"   Transformer: {num_layers} layers, {num_heads} heads")

        self.decoder = Decoder_TRANSFORMER(
            dim_in=3,
            points=points,
            seq_len=seq_len,
            z_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )

    def forward(self, z: torch.Tensor, debug: bool = False):
        """
        Generate mesh sequence from latent code.

        Args:
            z: [B, latent_dim] - Latent code
            debug: Print intermediate shapes

        Returns:
            v_out: [B, seq_len, points, 3] - Mesh vertices
        """
        v_out, _ = self.decoder(z, debug=debug)
        return v_out


if __name__ == "__main__":
    # Test Mesh Decoder
    print("=== Testing Mesh Decoder ===")

    latent_dim = 64
    seq_len = 50
    points = 1412
    batch_size = 8

    decoder = MeshDecoder(
        latent_dim=latent_dim,
        seq_len=seq_len,
        points=points
    )
    print(f"\nDecoder created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test forward pass
    dummy_z = torch.randn(batch_size, latent_dim)

    v_out = decoder(dummy_z)

    print(f"\nForward pass test:")
    print(f"  Input latent shape: {dummy_z.shape}")
    print(f"  Output mesh shape: {v_out.shape}")
    print(f"  Expected shape: [{batch_size}, {seq_len}, {points}, 3]")

    print("\n✓ Mesh Decoder test completed!")
