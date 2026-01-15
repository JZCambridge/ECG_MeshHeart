#!/usr/bin/env python3
"""
Verify that ALL parts of MeshDecoder are covered by pretrained weights
"""
import torch
import sys
import os
sys.path.insert(0, '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')
os.chdir('/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/ECG_MeshHeart')

from hybrid_ecg_mesh_vae.model.hybrid_vae import HybridECGMeshVAE

print("="*80)
print("DECODER WEIGHT MAPPING VERIFICATION")
print("="*80)

# Create the hybrid model
print("\n1️⃣  Creating HybridECGMeshVAE model...")
model = HybridECGMeshVAE(
    latent_dim=64,
    seq_len=50,
    points=1412,
)

# Get all parameter keys from the model decoder
print("\n2️⃣  Listing ALL parameter keys in HybridECGMeshVAE.decoder:")
print("-"*80)
decoder_keys = []
for name, param in model.decoder.named_parameters():
    full_name = f"decoder.{name}"
    decoder_keys.append(full_name)
    print(f"  {full_name}: {param.shape}")

print(f"\n   Total decoder parameters: {len(decoder_keys)}")

# Load the MeshHeart checkpoint and check what's available
print("\n3️⃣  Loading MeshHeart decoder checkpoint...")
ckpt_path = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt.get('state_dict', ckpt)

print("\n4️⃣  Extracting decoder keys from checkpoint...")
print("-"*80)

# Get decoder keys from checkpoint
decoder_checkpoint_keys = {}
for k, v in state_dict.items():
    if 'decoder.' in k:
        decoder_checkpoint_keys[k] = v.shape

print(f"Keys in checkpoint: {len(decoder_checkpoint_keys)}")
for k, shape in list(decoder_checkpoint_keys.items())[:20]:
    print(f"  {k}: {shape}")
if len(decoder_checkpoint_keys) > 20:
    print(f"  ... and {len(decoder_checkpoint_keys) - 20} more")

# Check coverage
print("\n5️⃣  CHECKING COVERAGE - Are all decoder parameters covered?")
print("="*80)

covered = []
missing = []

for model_key in decoder_keys:
    if model_key in decoder_checkpoint_keys:
        covered.append(model_key)
    else:
        missing.append(model_key)

print(f"\n✅ COVERED parameters: {len(covered)}/{len(decoder_keys)}")
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
    print("\n⚠️  WARNING: Some decoder parameters are NOT covered by pretrained weights!")
else:
    print(f"\n🎉 SUCCESS: ALL decoder parameters are covered by pretrained weights!")

# Also check for extra keys in checkpoint that don't match model
print("\n6️⃣  Checking for UNUSED checkpoint decoder keys...")
unused = []
for ckpt_key in decoder_checkpoint_keys.keys():
    if ckpt_key not in decoder_keys:
        unused.append(ckpt_key)

if unused:
    print(f"⚠️  UNUSED keys in checkpoint: {len(unused)}")
    for k in unused[:10]:
        print(f"  ! {k}")
    if len(unused) > 10:
        print(f"  ... and {len(unused) - 10} more")
else:
    print("✅ All checkpoint decoder keys are used")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Model decoder parameters: {len(decoder_keys)}")
print(f"Checkpoint decoder parameters: {len(decoder_checkpoint_keys)}")
print(f"Covered: {len(covered)}")
print(f"Missing: {len(missing)}")
print(f"Unused in checkpoint: {len(unused)}")

if len(missing) == 0:
    print("\n✅ VERIFICATION PASSED: All decoder weights will be loaded from pretrained checkpoint!")
else:
    print("\n❌ VERIFICATION FAILED: Some decoder weights are missing from checkpoint!")
