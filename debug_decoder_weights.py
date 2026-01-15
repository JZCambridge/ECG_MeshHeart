#!/usr/bin/env python3
"""
Debug: Compare decoder weights before and after loading
"""
import torch
import sys
import os
sys.path.insert(0, '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')
os.chdir('/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')

from hybrid_ecg_mesh_vae.model.hybrid_vae import HybridECGMeshVAE, initialize_from_pretrained

print("="*80)
print("DEBUGGING DECODER WEIGHT LOADING")
print("="*80)

mesh_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt'

# Load the checkpoint directly
print("\n1️⃣  Loading MeshHeart checkpoint directly...")
checkpoint = torch.load(mesh_ckpt, map_location='cpu', weights_only=False)
ckpt_state_dict = checkpoint.get('state_dict', checkpoint)

# Extract a sample decoder weight from checkpoint
sample_key_in_ckpt = 'decoder.ztimelinear.weight'
if sample_key_in_ckpt in ckpt_state_dict:
    ckpt_weight = ckpt_state_dict[sample_key_in_ckpt]
    print(f"   Checkpoint '{sample_key_in_ckpt}': shape={ckpt_weight.shape}")
    print(f"   First 5 values: {ckpt_weight.flatten()[:5]}")
    print(f"   Mean: {ckpt_weight.mean():.6f}, Std: {ckpt_weight.std():.6f}")

# Create hybrid model
print("\n2️⃣  Creating HybridECGMeshVAE model...")
model = HybridECGMeshVAE(latent_dim=64, seq_len=50, points=1412)

# Get initial decoder weight (random initialization)
model_key = 'decoder.decoder.ztimelinear.weight'
initial_weight = model.decoder.decoder.ztimelinear.weight.clone()
print(f"\n3️⃣  Initial (random) weight in model:")
print(f"   Shape: {initial_weight.shape}")
print(f"   First 5 values: {initial_weight.flatten()[:5]}")
print(f"   Mean: {initial_weight.mean():.6f}, Std: {initial_weight.std():.6f}")

# Load pretrained
print("\n4️⃣  Loading pretrained weights...")
ecg_ckpt = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
device = torch.device('cpu')
model = initialize_from_pretrained(model, ecg_ckpt, mesh_ckpt, device)

# Get loaded decoder weight
loaded_weight = model.decoder.decoder.ztimelinear.weight
print(f"\n5️⃣  After loading pretrained:")
print(f"   Shape: {loaded_weight.shape}")
print(f"   First 5 values: {loaded_weight.flatten()[:5]}")
print(f"   Mean: {loaded_weight.mean():.6f}, Std: {loaded_weight.std():.6f}")

# Compare
print("\n6️⃣  Comparison:")
print("="*80)

weights_match_checkpoint = torch.allclose(loaded_weight, ckpt_weight, rtol=1e-5, atol=1e-8)
weights_changed_from_initial = not torch.equal(loaded_weight, initial_weight)

if weights_match_checkpoint:
    print("✅ PERFECT: Loaded weights MATCH checkpoint exactly!")
else:
    print("❌ ERROR: Loaded weights DO NOT match checkpoint!")

if weights_changed_from_initial:
    print("✅ Weights changed from random initialization")
else:
    print("❌ ERROR: Weights still random (NOT loaded)!")

# Check other decoder layers
print("\n7️⃣  Checking all decoder layers...")
all_match = True
mismatches = []

for name, param in model.decoder.named_parameters():
    full_name = f"decoder.{name}"
    # The checkpoint has keys without the extra 'decoder.' prefix
    ckpt_key = full_name.replace('decoder.decoder.', 'decoder.')

    if ckpt_key in ckpt_state_dict:
        ckpt_param = ckpt_state_dict[ckpt_key]
        if not torch.allclose(param, ckpt_param, rtol=1e-5, atol=1e-8):
            all_match = False
            mismatches.append(full_name)

if all_match:
    print(f"✅ ALL {len(model.decoder.state_dict())} decoder parameters match checkpoint!")
else:
    print(f"❌ {len(mismatches)} parameters DO NOT match:")
    for k in mismatches[:10]:
        print(f"   - {k}")
    if len(mismatches) > 10:
        print(f"   ... and {len(mismatches) - 10} more")

print("\n" + "="*80)
if all_match and weights_match_checkpoint:
    print("🎉 SUCCESS: Decoder weights loaded perfectly!")
else:
    print("❌ PROBLEM: Decoder weights NOT loaded correctly!")
    print("   This explains the poor validation performance!")
print("="*80)
