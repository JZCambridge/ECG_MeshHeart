#!/usr/bin/env python3
"""
Verify that ALL parts of ECGEncoder are covered by pretrained weights
"""
import torch
import sys
import os
sys.path.insert(0, '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')
os.chdir('/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')

from hybrid_ecg_mesh_vae.model.hybrid_vae import HybridECGMeshVAE

print("="*80)
print("WEIGHT MAPPING VERIFICATION")
print("="*80)

# Create the hybrid model
print("\n1️⃣  Creating HybridECGMeshVAE model...")
model = HybridECGMeshVAE(
    latent_dim=64,
    seq_len=50,
    points=1412,
)

# Get all parameter keys from the model
print("\n2️⃣  Listing ALL parameter keys in HybridECGMeshVAE.encoder:")
print("-"*80)
encoder_keys = []
for name, param in model.encoder.named_parameters():
    full_name = f"encoder.{name}"
    encoder_keys.append(full_name)
    print(f"  {full_name}: {param.shape}")

print(f"\n   Total encoder parameters: {len(encoder_keys)}")

# Load the VAE checkpoint and check what's available
print("\n3️⃣  Loading VAE checkpoint to check available weights...")
ckpt_path = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt.get('state_dict', ckpt)

print("\n4️⃣  Mapping checkpoint keys to model keys...")
print("-"*80)

# Simulate the mapping we do in initialize_from_pretrained
mapped_keys = {}
for k, v in state_dict.items():
    if 'model.' in k:
        new_key = k.replace('model.', 'encoder.resnet.')
        mapped_keys[new_key] = v.shape

print(f"Keys in checkpoint (mapped): {len(mapped_keys)}")
for k, shape in list(mapped_keys.items())[:20]:
    print(f"  {k}: {shape}")
if len(mapped_keys) > 20:
    print(f"  ... and {len(mapped_keys) - 20} more")

# Check coverage
print("\n5️⃣  CHECKING COVERAGE - Are all model parameters covered?")
print("="*80)

covered = []
missing = []

for model_key in encoder_keys:
    if model_key in mapped_keys:
        covered.append(model_key)
    else:
        missing.append(model_key)

print(f"\n✅ COVERED parameters: {len(covered)}/{len(encoder_keys)}")
if len(covered) <= 30:
    for k in covered:
        print(f"  ✓ {k}")
else:
    for k in covered[:15]:
        print(f"  ✓ {k}")
    print(f"  ... and {len(covered) - 15} more")

if missing:
    print(f"\n❌ MISSING parameters (NOT in checkpoint): {len(missing)}")
    for k in missing:
        print(f"  ✗ {k}")
    print("\n⚠️  WARNING: Some encoder parameters are NOT covered by pretrained weights!")
else:
    print(f"\n🎉 SUCCESS: ALL encoder parameters are covered by pretrained weights!")

# Also check for extra keys in checkpoint that don't match model
print("\n6️⃣  Checking for UNUSED checkpoint keys...")
unused = []
for ckpt_key in mapped_keys.keys():
    if ckpt_key not in encoder_keys:
        unused.append(ckpt_key)

if unused:
    print(f"⚠️  UNUSED keys in checkpoint: {len(unused)}")
    for k in unused[:10]:
        print(f"  ! {k}")
    if len(unused) > 10:
        print(f"  ... and {len(unused) - 10} more")
else:
    print("✅ All checkpoint keys are used")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Model encoder parameters: {len(encoder_keys)}")
print(f"Checkpoint parameters (mapped): {len(mapped_keys)}")
print(f"Covered: {len(covered)}")
print(f"Missing: {len(missing)}")
print(f"Unused in checkpoint: {len(unused)}")

if len(missing) == 0:
    print("\n✅ VERIFICATION PASSED: All encoder weights will be loaded from pretrained checkpoint!")
else:
    print("\n❌ VERIFICATION FAILED: Some encoder weights are missing from checkpoint!")
