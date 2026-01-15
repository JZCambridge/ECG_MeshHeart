"""
Hybrid ECG-to-Mesh VAE Model

Combines:
1. ECG Encoder (from EchoNext): ECG + demographics + morphology → latent (mu, logvar)
2. Mesh Decoder (from MeshHeart): latent z → mesh sequence

This is a generative beta-VAE that takes ECG signals and patient information as input
and generates 50-frame cardiac mesh sequences as output.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .ecg_encoder import ECGEncoder
from .mesh_decoder import MeshDecoder


class HybridECGMeshVAE(nn.Module):
    """
    Hybrid ECG-to-Mesh Variational Autoencoder.

    Architecture:
        Input: ECG [B,12,2500] + Demographics [B,9] + Morphology [B,16]
               ↓
        ECG Encoder (ResNet1D + Tabular fusion)
               ↓
        Latent Distribution (mu, logvar) [B, latent_dim]
               ↓
        Reparameterization: z = mu + sigma * epsilon
               ↓
        Mesh Decoder (Transformer)
               ↓
        Output: Mesh Sequence [B, seq_len, points, 3]

    Loss:
        - Reconstruction loss: Chamfer distance + Laplacian smoothing
        - KL divergence: Regularize latent space

    The model can be initialized with pretrained weights from:
    - EchoNext ECG encoder checkpoint
    - MeshHeart mesh decoder checkpoint
    """

    def __init__(
        self,
        latent_dim: int = 64,
        seq_len: int = 50,
        points: int = 1412,
        # ECG Encoder parameters
        ecg_filter_size: int = 64,
        ecg_dropout: float = 0.5,
        ecg_conv1_kernel_size: int = 15,
        ecg_conv1_stride: int = 2,
        ecg_padding: int = 7,
        # Mesh Decoder parameters
        ff_size: int = 1024,
        num_layers: int = 2,
        num_heads: int = 4,
        activation: str = "gelu",
        decoder_dropout: float = 0.1,
        # Motion scaler for denormalization
        motion_scaler = None,
        use_motion_denorm: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.points = points
        self.motion_scaler = motion_scaler
        self.use_motion_denorm = use_motion_denorm

        print(f"🔧 Hybrid ECG-to-Mesh VAE Configuration:")
        print(f"   Latent dimension: {latent_dim}")
        print(f"   Sequence length: {seq_len} frames")
        print(f"   Mesh points: {points}")
        print(f"   Architecture: ECG Encoder → Latent Space → Mesh Decoder")
        print(f"   Motion denormalization: {'ENABLED' if use_motion_denorm and motion_scaler is not None else 'DISABLED'}")

        # ECG Encoder: 12-lead ECG + demographics + morphology → (mu, logvar)
        self.encoder = ECGEncoder(
            latent_dim=latent_dim,
            filter_size=ecg_filter_size,
            input_channels=12,
            dropout_value=ecg_dropout,
            conv1_kernel_size=ecg_conv1_kernel_size,
            conv1_stride=ecg_conv1_stride,
            padding=ecg_padding,
        )
        print(f"✅ ECG Encoder initialized")

        # Mesh Decoder: latent z → mesh sequence
        self.decoder = MeshDecoder(
            latent_dim=latent_dim,
            seq_len=seq_len,
            points=points,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            activation=activation,
            dropout=decoder_dropout,
        )
        print(f"✅ Mesh Decoder initialized")

        print(f"🎯 Total model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE sampling.

        z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)

        This allows gradients to flow through the stochastic latent variable.

        Args:
            mu: [B, latent_dim] - Mean of latent distribution
            logvar: [B, latent_dim] - Log variance of latent distribution

        Returns:
            z: [B, latent_dim] - Sampled latent code
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        ecg_raw: torch.Tensor,
        demographics: torch.Tensor,
        morphology: torch.Tensor,
        debug: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid VAE.

        Args:
            ecg_raw: [B, 12, 2500] - 12-lead ECG signals
            demographics: [B, 9] - Patient demographics
            morphology: [B, 16] - ECG morphology features
            debug: Print intermediate shapes for debugging

        Returns:
            v_out: [B, seq_len, points, 3] - Generated mesh sequence
            mu: [B, latent_dim] - Mean of latent distribution
            logvar: [B, latent_dim] - Log variance of latent distribution
        """
        if debug:
            print(f"🔍 Hybrid VAE Forward pass:")
            print(f"   Input ECG: {ecg_raw.shape}")
            print(f"   Input demographics: {demographics.shape}")
            print(f"   Input morphology: {morphology.shape}")

        # Encode: ECG + patient info → latent distribution parameters
        mu, logvar = self.encoder(ecg_raw, demographics, morphology)
        if debug:
            print(f"   Encoded mu: {mu.shape}")
            print(f"   Encoded logvar: {logvar.shape}")

        # Sample from latent distribution via reparameterization
        z = self.reparameterize(mu, logvar)
        if debug:
            print(f"   Sampled z (normalized): {z.shape}, range: [{z.min():.3f}, {z.max():.3f}]")

        # Denormalize latent before passing to decoder (if enabled)
        # Decoder was trained on denormalized latents, so we need to match that scale
        if self.use_motion_denorm and self.motion_scaler is not None:
            # Convert to numpy, apply inverse transform, convert back
            z_np = z.cpu().detach().numpy()
            z_denorm_np = self.motion_scaler.inverse_transform(z_np)
            z_denorm = torch.from_numpy(z_denorm_np).to(z.device).float()

            if debug:
                print(f"   Denormalized z: {z_denorm.shape}, range: [{z_denorm.min():.3f}, {z_denorm.max():.3f}]")

            z_for_decoder = z_denorm
        else:
            z_for_decoder = z

        # Decode: latent → mesh sequence
        v_out = self.decoder(z_for_decoder, debug=debug)
        if debug:
            print(f"   Output mesh: {v_out.shape}")

        return v_out, mu, logvar

    def encode(
        self,
        ecg_raw: torch.Tensor,
        demographics: torch.Tensor,
        morphology: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode ECG and patient info to latent representation.

        Useful for analysis and interpolation experiments.

        Args:
            ecg_raw: [B, 12, 2500] - ECG signals
            demographics: [B, 9] - Demographics
            morphology: [B, 16] - Morphology

        Returns:
            z: [B, latent_dim] - Sampled latent code
            mu: [B, latent_dim] - Mean
            logvar: [B, latent_dim] - Log variance
        """
        mu, logvar = self.encoder(ecg_raw, demographics, morphology)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to mesh sequence.

        Useful for generation from arbitrary latent codes.

        Args:
            z: [B, latent_dim] - Latent code

        Returns:
            v_out: [B, seq_len, points, 3] - Generated mesh
        """
        return self.decoder(z)

    def generate(
        self,
        ecg_raw: torch.Tensor,
        demographics: torch.Tensor,
        morphology: torch.Tensor
    ) -> torch.Tensor:
        """
        Generation mode: use mu directly (no sampling).

        This produces deterministic outputs from the same inputs,
        useful for inference/deployment.

        Args:
            ecg_raw: [B, 12, 2500] - ECG signals
            demographics: [B, 9] - Demographics
            morphology: [B, 16] - Morphology

        Returns:
            v_out: [B, seq_len, points, 3] - Generated mesh
        """
        mu, _ = self.encoder(ecg_raw, demographics, morphology)
        v_out = self.decoder(mu)
        return v_out

    def generate_random(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate random mesh sequences by sampling from prior N(0, I).

        This demonstrates the generative capability of the VAE.

        Args:
            batch_size: Number of samples to generate
            device: Device to generate on

        Returns:
            v_out: [batch_size, seq_len, points, 3] - Random mesh sequences
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decoder(z)


def initialize_from_pretrained(
    model: HybridECGMeshVAE,
    ecg_ckpt_path: str,
    mesh_ckpt_path: str,
    device: torch.device,
    strict: bool = False
):
    """
    Load pretrained weights for encoder and decoder separately.

    This enables transfer learning from individually trained models
    for improved training stability.

    Args:
        model: HybridECGMeshVAE instance
        ecg_ckpt_path: Path to EchoNext encoder checkpoint
        mesh_ckpt_path: Path to MeshHeart decoder checkpoint
        device: torch device
        strict: Whether to strictly match state dict keys

    Returns:
        model: Model with pretrained weights loaded
    """
    import os

    # Load ECG encoder checkpoint
    if ecg_ckpt_path and os.path.exists(ecg_ckpt_path):
        print(f"📥 Loading ECG encoder from: {ecg_ckpt_path}")
        ecg_checkpoint = torch.load(ecg_ckpt_path, map_location=device)

        # Extract encoder weights (handle key mismatches)
        ecg_state_dict = ecg_checkpoint.get('state_dict', ecg_checkpoint)

        # Map encoder keys from VAE checkpoint: model.* → encoder.resnet.*
        encoder_state_dict = {}
        for k, v in ecg_state_dict.items():
            # Handle different checkpoint formats
            # New VAE checkpoint format: model.* → encoder.resnet.*
            if 'model.' in k:
                new_key = k.replace('model.', 'encoder.resnet.')
                encoder_state_dict[new_key] = v
            elif 'encoder.' in k:
                # Already in correct format
                encoder_state_dict[k] = v

        if encoder_state_dict:
            # Load with strict=False to allow partial loading
            missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
            print(f"✅ ECG encoder weights loaded: {len(encoder_state_dict)} parameters")
            if missing:
                # Filter out decoder-related missing keys for cleaner output
                encoder_missing = [k for k in missing if 'encoder' in k]
                if encoder_missing:
                    print(f"   ⚠️  Missing encoder keys: {len(encoder_missing)}")
                    for k in encoder_missing[:5]:
                        print(f"      {k}")
                    if len(encoder_missing) > 5:
                        print(f"      ... and {len(encoder_missing) - 5} more")
            if unexpected:
                print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
        else:
            print(f"⚠️  No matching encoder weights found in checkpoint")
    else:
        print(f"⚠️  ECG encoder checkpoint not found: {ecg_ckpt_path}")

    # Load Mesh decoder checkpoint
    if mesh_ckpt_path and os.path.exists(mesh_ckpt_path):
        print(f"📥 Loading mesh decoder from: {mesh_ckpt_path}")
        mesh_checkpoint = torch.load(mesh_ckpt_path, map_location=device)

        # Extract decoder weights
        mesh_state_dict = mesh_checkpoint.get('state_dict', mesh_checkpoint)

        decoder_state_dict = {}
        for k, v in mesh_state_dict.items():
            if 'decoder.' in k:
                # HybridECGMeshVAE has structure: decoder.decoder.*
                # MeshHeart checkpoint has: decoder.*
                # Need to add extra 'decoder.' prefix to match nested structure
                new_key = k.replace('decoder.', 'decoder.decoder.', 1)
                decoder_state_dict[new_key] = v

        if decoder_state_dict:
            missing, unexpected = model.load_state_dict(decoder_state_dict, strict=False)
            print(f"✅ Mesh decoder weights loaded: {len(decoder_state_dict)} parameters")
            if missing:
                # Filter out encoder-related missing keys for cleaner output
                decoder_missing = [k for k in missing if 'decoder' in k]
                if decoder_missing:
                    print(f"   ⚠️  Missing decoder keys: {len(decoder_missing)}")
                    for k in decoder_missing[:5]:
                        print(f"      {k}")
                    if len(decoder_missing) > 5:
                        print(f"      ... and {len(decoder_missing) - 5} more")
            if unexpected:
                print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
        else:
            print(f"⚠️  No matching decoder weights found in checkpoint")
    else:
        print(f"⚠️  Mesh decoder checkpoint not found: {mesh_ckpt_path}")

    print("🎯 Model initialized with pretrained weights")
    return model


if __name__ == "__main__":
    # Test Hybrid VAE
    print("=== Testing Hybrid ECG-to-Mesh VAE ===\n")

    latent_dim = 64
    seq_len = 50
    points = 1412
    batch_size = 4

    # Create model
    model = HybridECGMeshVAE(
        latent_dim=latent_dim,
        seq_len=seq_len,
        points=points
    )

    print(f"\n=== Forward Pass Test ===")
    # Test forward pass
    dummy_ecg = torch.randn(batch_size, 12, 2500)
    dummy_demographics = torch.randn(batch_size, 9)
    dummy_morphology = torch.randn(batch_size, 16)

    v_out, mu, logvar = model(dummy_ecg, dummy_demographics, dummy_morphology)

    print(f"  Input ECG shape: {dummy_ecg.shape}")
    print(f"  Input demographics shape: {dummy_demographics.shape}")
    print(f"  Input morphology shape: {dummy_morphology.shape}")
    print(f"  Output mesh shape: {v_out.shape}")
    print(f"  Latent mu shape: {mu.shape}")
    print(f"  Latent logvar shape: {logvar.shape}")

    print(f"\n=== Generation Test ===")
    # Test generation mode
    v_gen = model.generate(dummy_ecg, dummy_demographics, dummy_morphology)
    print(f"  Generated mesh shape: {v_gen.shape}")

    print(f"\n=== Random Generation Test ===")
    # Test random generation
    v_random = model.generate_random(batch_size, device=dummy_ecg.device)
    print(f"  Random mesh shape: {v_random.shape}")

    print("\n✓ Hybrid VAE test completed!")
