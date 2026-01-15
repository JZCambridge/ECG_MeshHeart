#!/usr/bin/env python3
"""
Check: Latent code distribution from encoder vs expected by decoder
"""
import torch
import sys
import os
import pandas as pd
sys.path.insert(0, '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')
os.chdir('/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')

from hybrid_ecg_mesh_vae.model.hybrid_vae import HybridECGMeshVAE, initialize_from_pretrained

print("="*80)
print("LATENT DISTRIBUTION ANALYSIS")
print("="*80)

# Load validation latents that the decoder was trained on
print("\n1️⃣  Loading MeshHeart training latents (what decoder expects)...")
val_latents_csv = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/validation_best_table.csv'
df = pd.read_csv(val_latents_csv)
latent_cols = [col for col in df.columns if col.startswith('latent_')]
mesh_latents = torch.tensor(df[latent_cols].values, dtype=torch.float32)

print(f"   Shape: {mesh_latents.shape}")
print(f"   Mean: {mesh_latents.mean():.6f}, Std: {mesh_latents.std():.6f}")
print(f"   Min: {mesh_latents.min():.6f}, Max: {mesh_latents.max():.6f}")
print(f"   Per-dimension stats:")
print(f"     Mean range: [{mesh_latents.mean(dim=0).min():.4f}, {mesh_latents.mean(dim=0).max():.4f}]")
print(f"     Std range: [{mesh_latents.std(dim=0).min():.4f}, {mesh_latents.std(dim=0).max():.4f}]")

# Create hybrid model and load weights
print("\n2️⃣  Creating HybridECGMeshVAE and loading weights...")
model = HybridECGMeshVAE(latent_dim=64, seq_len=50, points=1412)
ecg_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
mesh_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt'
device = torch.device('cpu')
model = initialize_from_pretrained(model, ecg_ckpt, mesh_ckpt, device)
model.eval()

# Generate latents from random ECG inputs
print("\n3️⃣  Generating latents from ECG encoder with random inputs...")
batch_size = 100
ecg_raw = torch.randn(batch_size, 12, 2500)
demographics = torch.randn(batch_size, 9)
morphology = torch.randn(batch_size, 16)

with torch.no_grad():
    mu, logvar = model.encoder(ecg_raw, demographics, morphology)
    z = model.reparameterize(mu, logvar)

print(f"   Generated latents shape: {z.shape}")
print(f"   Mu - Mean: {mu.mean():.6f}, Std: {mu.std():.6f}")
print(f"   Mu - Min: {mu.min():.6f}, Max: {mu.max():.6f}")
print(f"   LogVar - Mean: {logvar.mean():.6f}, Std: {logvar.std():.6f}")
print(f"   LogVar - Min: {logvar.min():.6f}, Max: {logvar.max():.6f}")
print(f"   Z (sampled) - Mean: {z.mean():.6f}, Std: {z.std():.6f}")
print(f"   Z (sampled) - Min: {z.min():.6f}, Max: {z.max():.6f}")

# Check for NaN/Inf
has_nan_mu = torch.isnan(mu).any()
has_inf_mu = torch.isinf(mu).any()
has_nan_z = torch.isnan(z).any()
has_inf_z = torch.isinf(z).any()

print(f"\n4️⃣  Checking for NaN/Inf...")
if has_nan_mu or has_inf_mu:
    print(f"   ❌ ERROR: Mu has NaN={has_nan_mu} or Inf={has_inf_mu}")
if has_nan_z or has_inf_z:
    print(f"   ❌ ERROR: Z has NaN={has_nan_z} or Inf={has_inf_z}")
if not (has_nan_mu or has_inf_mu or has_nan_z or has_inf_z):
    print(f"   ✅ No NaN/Inf detected in encoder outputs")

# Compare distributions
print("\n5️⃣  DISTRIBUTION COMPARISON:")
print("="*80)
print(f"Expected (MeshHeart training latents):")
print(f"   Mean: {mesh_latents.mean():.6f}, Std: {mesh_latents.std():.6f}")
print(f"   Range: [{mesh_latents.min():.6f}, {mesh_latents.max():.6f}]")
print(f"\nGenerated (ECG encoder mu):")
print(f"   Mean: {mu.mean():.6f}, Std: {mu.std():.6f}")
print(f"   Range: [{mu.min():.6f}, {mu.max():.6f}]")
print(f"\nGenerated (ECG encoder z - with sampling):")
print(f"   Mean: {z.mean():.6f}, Std: {z.std():.6f}")
print(f"   Range: [{z.min():.6f}, {z.max():.6f}]")

# Test decoder with expected vs generated latents
print("\n6️⃣  Testing decoder with different latent inputs...")
with torch.no_grad():
    # Test with expected latents (from MeshHeart training)
    mesh_sample = mesh_latents[:16]  # Take 16 samples
    v_out_expected = model.decoder(mesh_sample)
    print(f"   Decoder output (expected latents): {v_out_expected.shape}")
    print(f"     Range: [{v_out_expected.min():.3f}, {v_out_expected.max():.3f}]")
    print(f"     Has NaN: {torch.isnan(v_out_expected).any()}")
    print(f"     Has Inf: {torch.isinf(v_out_expected).any()}")

    # Test with generated latents (from ECG encoder)
    v_out_generated = model.decoder(z[:16])
    print(f"   Decoder output (generated latents): {v_out_generated.shape}")
    print(f"     Range: [{v_out_generated.min():.3f}, {v_out_generated.max():.3f}]")
    print(f"     Has NaN: {torch.isnan(v_out_generated).any()}")
    print(f"     Has Inf: {torch.isinf(v_out_generated).any()}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
if has_nan_z or has_inf_z:
    print("❌ PROBLEM: Encoder producing NaN/Inf values!")
elif abs(z.std() - mesh_latents.std()) > 2.0:
    print(f"⚠️  WARNING: Large distribution mismatch!")
    print(f"   Encoder std ({z.std():.3f}) vs Expected std ({mesh_latents.std():.3f})")
    print(f"   Difference: {abs(z.std() - mesh_latents.std()):.3f}")
else:
    print("✅ Latent distributions look reasonable")
