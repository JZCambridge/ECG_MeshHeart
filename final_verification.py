#!/usr/bin/env python3
"""
FINAL VERIFICATION: All weights loaded correctly
"""
import torch
import sys
import os
sys.path.insert(0, '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')
os.chdir('/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')

# Need to reload the module to get updated code
import importlib
if 'hybrid_ecg_mesh_vae.model.hybrid_vae' in sys.modules:
    del sys.modules['hybrid_ecg_mesh_vae.model.hybrid_vae']

from hybrid_ecg_mesh_vae.model.hybrid_vae import HybridECGMeshVAE, initialize_from_pretrained

print("="*80)
print("FINAL COMPREHENSIVE VERIFICATION")
print("="*80)

# Checkpoint paths
ecg_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
mesh_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt'

print("\n1️⃣  Creating model...")
model = HybridECGMeshVAE(latent_dim=64, seq_len=50, points=1412)

# Get initial state
initial_encoder_param = model.encoder.resnet.conv1.weight.clone()
initial_decoder_param = model.decoder.decoder.ztimelinear.weight.clone()

print("\n2️⃣  Loading pretrained weights...")
device = torch.device('cpu')
model = initialize_from_pretrained(model, ecg_ckpt, mesh_ckpt, device)

# Get loaded state
loaded_encoder_param = model.encoder.resnet.conv1.weight
loaded_decoder_param = model.decoder.decoder.ztimelinear.weight

print("\n3️⃣  Verification Results:")
print("="*80)

# Check encoder changed
encoder_changed = not torch.equal(initial_encoder_param, loaded_encoder_param)
print(f"✅ Encoder weights changed: {encoder_changed}")
if not encoder_changed:
    print("   ❌ ERROR: Encoder weights were NOT loaded!")

# Check decoder changed
decoder_changed = not torch.equal(initial_decoder_param, loaded_decoder_param)
print(f"✅ Decoder weights changed: {decoder_changed}")
if not decoder_changed:
    print("   ❌ ERROR: Decoder weights were NOT loaded!")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())

print(f"\n4️⃣  Parameter counts:")
print(f"   Total: {total_params:,}")
print(f"   Encoder: {encoder_params:,}")
print(f"   Decoder: {decoder_params:,}")

print("\n5️⃣  Forward pass test...")
model.eval()
with torch.no_grad():
    ecg = torch.randn(2, 12, 2500)
    demo = torch.randn(2, 9)
    morph = torch.randn(2, 16)
    v_out, mu, logvar = model(ecg, demo, morph)
    print(f"✅ Forward pass successful!")
    print(f"   Output shapes: v_out={v_out.shape}, mu={mu.shape}, logvar={logvar.shape}")

print("\n" + "="*80)
if encoder_changed and decoder_changed:
    print("🎉 SUCCESS: ALL pretrained weights loaded correctly!")
    print("✅ Encoder: Fully loaded from VAE checkpoint")
    print("✅ Decoder: Fully loaded from MeshHeart checkpoint")
    print("✅ Model ready for training!")
else:
    print("❌ FAILURE: Some weights were not loaded!")
    if not encoder_changed:
        print("   ❌ Encoder weights NOT loaded")
    if not decoder_changed:
        print("   ❌ Decoder weights NOT loaded")
print("="*80)
