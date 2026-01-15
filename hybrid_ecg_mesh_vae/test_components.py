#!/usr/bin/env python
"""
Test script for Hybrid ECG-to-Mesh VAE components

Tests:
1. ECG Encoder
2. Mesh Decoder
3. Hybrid VAE
4. Individual component forward passes

Run this to verify all components are working correctly before training.
"""

import torch
import sys
sys.path.append('.')

print("="*80)
print("Testing Hybrid ECG-to-Mesh VAE Components")
print("="*80)

# Test parameters
batch_size = 4
latent_dim = 64
seq_len = 50
points = 1412

print(f"\nTest Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Latent dimension: {latent_dim}")
print(f"  Sequence length: {seq_len}")
print(f"  Mesh points: {points}")

# ============================================================================
# Test 1: ECG Encoder
# ============================================================================
print("\n" + "="*80)
print("Test 1: ECG Encoder")
print("="*80)

try:
    from model.ecg_encoder import ECGEncoder

    encoder = ECGEncoder(latent_dim=latent_dim)
    print(f"✅ ECG Encoder created")
    print(f"   Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test forward pass
    dummy_ecg = torch.randn(batch_size, 12, 2500)
    dummy_demographics = torch.randn(batch_size, 9)
    dummy_morphology = torch.randn(batch_size, 16)

    mu, logvar = encoder(dummy_ecg, dummy_demographics, dummy_morphology)

    print(f"✅ Forward pass successful:")
    print(f"   Input ECG: {dummy_ecg.shape}")
    print(f"   Input demographics: {dummy_demographics.shape}")
    print(f"   Input morphology: {dummy_morphology.shape}")
    print(f"   Output mu: {mu.shape}")
    print(f"   Output logvar: {logvar.shape}")

    assert mu.shape == (batch_size, latent_dim), f"Expected mu shape {(batch_size, latent_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Expected logvar shape {(batch_size, latent_dim)}, got {logvar.shape}"

    print("✅ ECG Encoder test PASSED")

except Exception as e:
    print(f"❌ ECG Encoder test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 2: Mesh Decoder
# ============================================================================
print("\n" + "="*80)
print("Test 2: Mesh Decoder")
print("="*80)

try:
    from model.mesh_decoder import MeshDecoder

    decoder = MeshDecoder(latent_dim=latent_dim, seq_len=seq_len, points=points)
    print(f"✅ Mesh Decoder created")
    print(f"   Parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test forward pass
    dummy_z = torch.randn(batch_size, latent_dim)

    v_out = decoder(dummy_z)

    print(f"✅ Forward pass successful:")
    print(f"   Input latent: {dummy_z.shape}")
    print(f"   Output mesh: {v_out.shape}")

    expected_shape = (batch_size, seq_len, points, 3)
    assert v_out.shape == expected_shape, f"Expected output shape {expected_shape}, got {v_out.shape}"

    print("✅ Mesh Decoder test PASSED")

except Exception as e:
    print(f"❌ Mesh Decoder test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 3: Hybrid VAE
# ============================================================================
print("\n" + "="*80)
print("Test 3: Hybrid VAE")
print("="*80)

try:
    from model.hybrid_vae import HybridECGMeshVAE

    model = HybridECGMeshVAE(latent_dim=latent_dim, seq_len=seq_len, points=points)
    print(f"✅ Hybrid VAE created")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    v_out, mu, logvar = model(dummy_ecg, dummy_demographics, dummy_morphology)

    print(f"✅ Forward pass successful:")
    print(f"   Input ECG: {dummy_ecg.shape}")
    print(f"   Input demographics: {dummy_demographics.shape}")
    print(f"   Input morphology: {dummy_morphology.shape}")
    print(f"   Output mesh: {v_out.shape}")
    print(f"   Latent mu: {mu.shape}")
    print(f"   Latent logvar: {logvar.shape}")

    expected_mesh_shape = (batch_size, seq_len, points, 3)
    assert v_out.shape == expected_mesh_shape, f"Expected mesh shape {expected_mesh_shape}, got {v_out.shape}"
    assert mu.shape == (batch_size, latent_dim), f"Expected mu shape {(batch_size, latent_dim)}, got {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"Expected logvar shape {(batch_size, latent_dim)}, got {logvar.shape}"

    print("✅ Hybrid VAE forward pass test PASSED")

    # Test generation mode
    v_gen = model.generate(dummy_ecg, dummy_demographics, dummy_morphology)
    print(f"✅ Generation mode successful:")
    print(f"   Generated mesh: {v_gen.shape}")
    assert v_gen.shape == expected_mesh_shape, f"Expected shape {expected_mesh_shape}, got {v_gen.shape}"

    # Test random generation
    v_random = model.generate_random(batch_size, device=dummy_ecg.device)
    print(f"✅ Random generation successful:")
    print(f"   Random mesh: {v_random.shape}")
    assert v_random.shape == expected_mesh_shape, f"Expected shape {expected_mesh_shape}, got {v_random.shape}"

    # Test encode/decode
    z, mu_enc, logvar_enc = model.encode(dummy_ecg, dummy_demographics, dummy_morphology)
    v_dec = model.decode(z)
    print(f"✅ Encode/decode successful:")
    print(f"   Encoded z: {z.shape}")
    print(f"   Decoded mesh: {v_dec.shape}")

    print("✅ Hybrid VAE all tests PASSED")

except Exception as e:
    print(f"❌ Hybrid VAE test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 4: Reparameterization Trick
# ============================================================================
print("\n" + "="*80)
print("Test 4: Reparameterization Trick")
print("="*80)

try:
    # Test that reparameterization produces different samples
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)

    print(f"✅ Reparameterization produces stochastic samples:")
    print(f"   Sample 1 mean: {z1.mean().item():.4f}, std: {z1.std().item():.4f}")
    print(f"   Sample 2 mean: {z2.mean().item():.4f}, std: {z2.std().item():.4f}")

    # They should be different (stochastic)
    assert not torch.allclose(z1, z2), "Reparameterization should produce different samples"

    # But centered around mu
    z_mean = torch.stack([model.reparameterize(mu, logvar) for _ in range(100)]).mean(dim=0)
    print(f"   Mean of 100 samples close to mu: {torch.allclose(z_mean, mu, atol=0.5)}")

    print("✅ Reparameterization test PASSED")

except Exception as e:
    print(f"❌ Reparameterization test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 5: Gradient Flow
# ============================================================================
print("\n" + "="*80)
print("Test 5: Gradient Flow")
print("="*80)

try:
    # Test that gradients flow through the model
    model.train()
    dummy_ecg.requires_grad = True

    v_out, mu, logvar = model(dummy_ecg, dummy_demographics, dummy_morphology)

    # Dummy loss
    loss = v_out.sum() + mu.sum() + logvar.sum()
    loss.backward()

    # Check gradients exist
    assert dummy_ecg.grad is not None, "Gradients should flow to input"

    # Check model parameters have gradients
    encoder_has_grad = any(p.grad is not None for p in model.encoder.parameters())
    decoder_has_grad = any(p.grad is not None for p in model.decoder.parameters())

    assert encoder_has_grad, "Encoder should have gradients"
    assert decoder_has_grad, "Decoder should have gradients"

    print("✅ Gradient flow test PASSED")
    print("   Encoder has gradients: ✓")
    print("   Decoder has gradients: ✓")
    print("   Input has gradients: ✓")

except Exception as e:
    print(f"❌ Gradient flow test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nComponent Summary:")
print(f"  ✅ ECG Encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
print(f"  ✅ Mesh Decoder: {sum(p.numel() for p in decoder.parameters()):,} parameters")
print(f"  ✅ Hybrid VAE: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"\n  Input → Output:")
print(f"  [ECG: {dummy_ecg.shape}] + [Demographics: {dummy_demographics.shape}] + [Morphology: {dummy_morphology.shape}]")
print(f"  → [Mesh: {v_out.shape}]")
print(f"\n  Latent Space:")
print(f"  [mu, logvar: {mu.shape}] → [z: {z.shape}]")
print("\nThe model is ready for training!")
print("="*80)
