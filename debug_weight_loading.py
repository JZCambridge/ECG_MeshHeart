#!/usr/bin/env python
"""
Debug script to check if encoder weights are being loaded correctly.
"""
import sys
sys.path.insert(0, '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')

import torch
from hybrid_ecg_mesh_vae.model.hybrid_vae import HybridECGMeshVAE, initialize_from_pretrained

# Paths
ecg_ckpt_path = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
mesh_ckpt_path = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("WEIGHT LOADING DEBUG")
print("="*80)

# Create model
print("\n1. Creating hybrid model...")
model = HybridECGMeshVAE(
    latent_dim=64,
    ecg_filter_size=64,
    ecg_dropout=0.5,
)

# Get encoder parameter before loading
encoder_params_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
print(f"   Encoder has {len(encoder_params_before)} parameters")

# Sample a specific parameter to track
sample_param_name = 'resnet.conv1.weight'
sample_param_before = encoder_params_before[sample_param_name].clone()
print(f"   Sample param '{sample_param_name}' mean: {sample_param_before.mean().item():.6f}")

# Load pretrained weights
print("\n2. Loading pretrained weights...")
model = initialize_from_pretrained(
    model,
    ecg_ckpt_path,
    mesh_ckpt_path,
    device
)

# Check encoder parameter after loading
encoder_params_after = model.encoder.state_dict()
sample_param_after = encoder_params_after[sample_param_name]
print(f"\n3. Checking if weights changed...")
print(f"   Sample param '{sample_param_name}' mean after: {sample_param_after.mean().item():.6f}")

# Check if parameters actually changed
params_changed = not torch.allclose(sample_param_before, sample_param_after.cpu())
print(f"   Parameters changed: {params_changed}")

if params_changed:
    print("   ✅ SUCCESS: Weights were loaded!")
else:
    print("   ❌ FAILURE: Weights were NOT loaded!")

# Check how many parameters match
print("\n4. Detailed parameter check...")
changed_count = 0
for name in encoder_params_before.keys():
    if not torch.allclose(encoder_params_before[name], encoder_params_after[name].cpu()):
        changed_count += 1

print(f"   Total encoder parameters: {len(encoder_params_before)}")
print(f"   Parameters that changed: {changed_count}")
print(f"   Parameters unchanged: {len(encoder_params_before) - changed_count}")

print("\n" + "="*80)
if changed_count > 0:
    print("✅ VERDICT: Pretrained weights ARE being loaded")
else:
    print("❌ VERDICT: Pretrained weights are NOT being loaded!")
print("="*80)
