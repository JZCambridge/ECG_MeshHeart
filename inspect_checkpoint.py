#!/usr/bin/env python3
"""
Inspect the new VAE checkpoint structure
"""
import torch

# Path to the new VAE checkpoint
ckpt_path = '/home/jzheng12@isd.csc.mrc.ac.uk/jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt'

print(f"Loading checkpoint from: {ckpt_path}\n")
ckpt = torch.load(ckpt_path, map_location='cpu')

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
print("STATE_DICT STRUCTURE (all keys)")
print("="*80)
state_dict = ckpt.get('state_dict', ckpt)
for key in sorted(state_dict.keys()):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"  {key}: {shape}")

print(f"\n" + "="*80)
print(f"SUMMARY")
print("="*80)
print(f"Total parameters in state_dict: {len(state_dict.keys())}")
print(f"Checkpoint metadata:")
if 'epoch' in ckpt:
    print(f"  Epoch: {ckpt['epoch']}")
if 'val_loss' in ckpt:
    print(f"  Val Loss: {ckpt['val_loss']}")
if 'val_corr' in ckpt:
    print(f"  Val Correlation: {ckpt['val_corr']}")
if 'beta' in ckpt:
    print(f"  Beta: {ckpt['beta']}")
