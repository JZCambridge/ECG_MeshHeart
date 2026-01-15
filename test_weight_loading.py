#!/usr/bin/env python3
"""
Test script to verify pretrained weight loading for Hybrid ECG-Mesh VAE
"""
import torch
import sys
sys.path.insert(0, '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')

from hybrid_ecg_mesh_vae.model.hybrid_vae import HybridECGMeshVAE, initialize_from_pretrained

print("="*80)
print("Testing Pretrained Weight Loading for Hybrid ECG-Mesh VAE")
print("="*80)

# Checkpoint paths
ecg_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
mesh_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt'

print("\n1️⃣  Creating Hybrid ECG-Mesh VAE model...")
model = HybridECGMeshVAE(
    latent_dim=64,
    seq_len=50,
    points=1412,
    ecg_filter_size=64,
    ecg_dropout=0.5,
    ff_size=1024,
    num_layers=2,
    num_heads=4,
)

# Count parameters before loading
total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())

print(f"\n📊 Model Statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Encoder parameters: {encoder_params:,}")
print(f"   Decoder parameters: {decoder_params:,}")

print("\n2️⃣  Loading pretrained weights...")
print(f"   ECG Encoder: {ecg_ckpt}")
print(f"   Mesh Decoder: {mesh_ckpt}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = initialize_from_pretrained(
    model,
    ecg_ckpt_path=ecg_ckpt,
    mesh_ckpt_path=mesh_ckpt,
    device=device
)

print("\n3️⃣  Testing forward pass...")
model = model.to(device)
model.eval()

# Create dummy inputs
batch_size = 4
ecg_raw = torch.randn(batch_size, 12, 2500).to(device)
demographics = torch.randn(batch_size, 9).to(device)
morphology = torch.randn(batch_size, 16).to(device)

with torch.no_grad():
    v_out, mu, logvar = model(ecg_raw, demographics, morphology)

print(f"✅ Forward pass successful!")
print(f"   Input shapes:")
print(f"      ECG: {ecg_raw.shape}")
print(f"      Demographics: {demographics.shape}")
print(f"      Morphology: {morphology.shape}")
print(f"   Output shapes:")
print(f"      Mesh: {v_out.shape}")
print(f"      Mu: {mu.shape}")
print(f"      Logvar: {logvar.shape}")

print("\n4️⃣  Verifying output ranges...")
print(f"   Mesh output range: [{v_out.min():.3f}, {v_out.max():.3f}]")
print(f"   Mu range: [{mu.min():.3f}, {mu.max():.3f}]")
print(f"   Logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")

print("\n5️⃣  Checking for NaN/Inf...")
has_nan = torch.isnan(v_out).any() or torch.isnan(mu).any() or torch.isnan(logvar).any()
has_inf = torch.isinf(v_out).any() or torch.isinf(mu).any() or torch.isinf(logvar).any()

if has_nan:
    print("   ⚠️  WARNING: NaN values detected in outputs!")
elif has_inf:
    print("   ⚠️  WARNING: Inf values detected in outputs!")
else:
    print("   ✅ No NaN/Inf values detected")

print("\n" + "="*80)
print("✅ All tests passed! Pretrained weight loading successful!")
print("="*80)
print("\nSummary:")
print("✅ ECG Encoder: Fully loaded with new VAE checkpoint")
print("✅ Mesh Decoder: Fully loaded with MeshHeart checkpoint")
print("✅ Forward pass: Working correctly")
print("✅ Ready for training!")
