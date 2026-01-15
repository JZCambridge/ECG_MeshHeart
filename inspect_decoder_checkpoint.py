#!/usr/bin/env python3
"""
Inspect the MeshHeart decoder checkpoint structure
"""
import torch

# Path to the MeshHeart decoder checkpoint
ckpt_path = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt'

print(f"Loading MeshHeart decoder checkpoint from: {ckpt_path}\n")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("="*80)
print("CHECKPOINT TOP-LEVEL KEYS")
print("="*80)
for key in ckpt.keys():
    if isinstance(ckpt[key], dict):
        print(f"  {key}: dict with {len(ckpt[key])} keys")
    elif isinstance(ckpt[key], (int, float, str)):
        print(f"  {key}: {ckpt[key]}")
    else:
        print(f"  {key}: {type(ckpt[key])}")

print("\n" + "="*80)
print("STATE_DICT STRUCTURE (first 50 keys)")
print("="*80)
state_dict = ckpt.get('state_dict', ckpt)
for i, key in enumerate(sorted(state_dict.keys())[:50]):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"  {key}: {shape}")

if len(state_dict.keys()) > 50:
    print(f"  ... and {len(state_dict.keys()) - 50} more keys")

print(f"\n" + "="*80)
print(f"SUMMARY")
print("="*80)
print(f"Total parameters in state_dict: {len(state_dict.keys())}")

# Check for decoder-specific keys
decoder_keys = [k for k in state_dict.keys() if 'decoder' in k.lower()]
print(f"Keys containing 'decoder': {len(decoder_keys)}")
if decoder_keys:
    print("Sample decoder keys:")
    for k in decoder_keys[:10]:
        print(f"  {k}")
