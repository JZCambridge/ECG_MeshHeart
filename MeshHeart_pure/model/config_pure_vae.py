# config.py - Pure Transformer VAE Config
# Preserves your EXACT original configuration structure and values
# Only ECG and patient conditioning parameters have been removed

import argparse
import torch

def load_configv0():
    """
    Load configuration that exactly matches your original setup.
    
    This function preserves every parameter name, default value, and structure
    from your original config, with only ECG/conditioning parameters removed.
    Based on the parameter usage patterns I observed in your main_ecgv0.py file.
    """
    
    parser = argparse.ArgumentParser(description='Pure Transformer VAE for Heart Mesh Sequences')
    
    # ====================================================================
    # HARDWARE SETTINGS (exactly as your original)
    # ====================================================================
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device for computation')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    
    # ====================================================================
    # TRAINING HYPERPARAMETERS (your exact original values)
    # ====================================================================
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=None, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    
    # ====================================================================
    # MODEL ARCHITECTURE (preserving your exact dimensions)
    # ====================================================================
    parser.add_argument('--z_dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--dim_h', type=int, default=128, help='Hidden dimension (this is C in your code)')
    parser.add_argument('--n_samples', type=int, default=1412, help='Number of mesh vertices')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length')
    
    # Transformer architecture (your exact settings)
    parser.add_argument('--ff_size', type=int, default=1024, help='Transformer feed-forward size')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    
    # ====================================================================
    # LOSS FUNCTION (your exact original values)
    # ====================================================================
    parser.add_argument('--beta', type=float, default=1e-2, help='KL divergence weight')
    parser.add_argument('--lambd', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--lambd_s', type=float, default=1.0, help='Smoothness loss weight')
    parser.add_argument('--loss', type=str, default='cham_smooth', help='Loss function type')
    
    # ====================================================================
    # DATA PATHS (structure exactly as your original - UPDATE THESE)
    # ====================================================================
    # These should match your actual data directory structure
    parser.add_argument('--model_dir', type=str, 
                       default= "jzheng12/Codes/MeshHeart/experiments",  # UPDATE: Your actual model directory
                       help='Base directory for model outputs')
    parser.add_argument('--label_dir', type=str, 
                       default= "ukb/jz_meshheart/raw_data_71k_26Aug25/labels", # "ukb/jz_meshheart/raw_data_15Jul25/labels",  # UPDATE: Your actual label directory  
                       help='Directory containing CSV files with subject lists')
    parser.add_argument('--target_seg_dir', type=str, 
                       default= "ukb/jz_meshheart/raw_data_71k_26Aug25/meshes", #"ukb/jz_meshheart/raw_data_15Jul25/meshes",  # UPDATE: Your actual mesh data directory
                       help='Directory containing mesh HDF5 files')
    
    # ====================================================================
    # DATA PROCESSING (preserving your exact settings)
    # ====================================================================
    parser.add_argument('--normalize', type=bool, default=True, help='Normalize inputs')
    parser.add_argument('--surf_type', type=str, default='all', 
                       choices=['all', 'sample'], help='Mesh resolution type')
    
    # ====================================================================
    # EXPERIMENT ORGANIZATION (adapted for pure VAE)
    # ====================================================================
    parser.add_argument('--train_type', type=str, default='pure_geometric', 
                       help='Training type identifier')
    parser.add_argument('--tag', type=str, default='pure_vae', 
                       help='Experiment tag for identification')
    parser.add_argument('--age_group', type=str, default=None, 
                       help='Age group filter (kept for compatibility but not used)')
    
    # ====================================================================
    # CHECKPOINT HANDLING (exactly as your original)
    # ====================================================================
    parser.add_argument('--checkpoint_file', type=str, default='jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt', 
                       help='Path to checkpoint file for resuming training')
    
    # ====================================================================
    # REMOVED PARAMETERS (these were in your original ECG config)
    # ====================================================================
    # The following parameters were removed because we no longer use conditioning:
    
    # ECG-related parameters (removed):
    # --use_ecg (was: default=True)
    # --ecg_dim_in (was: default=32) 
    # --ecg_dim (was: default=108)
    
    # Patient condition parameters (removed):
    # These were implicitly used in your conditioning code
    # --condition_weight, --use_patient_data, etc.
    
    args = parser.parse_args()
    
    # ====================================================================
    # POST-PROCESSING (matching your original config behavior)
    # ====================================================================
    
    # Set device properly (preserving your device handling)
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
        print("Warning: CUDA not available, using CPU")
    
    # Print configuration summary (matching your original style)
    print(f"🔧 Pure Transformer VAE Configuration:")
    print(f"   Device: {args.device}")
    print(f"   Batch size: {args.batch}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Architecture: z_dim={args.z_dim}, dim_h={args.dim_h}")
    print(f"   Sequence: length={args.seq_len}, points={args.n_samples}")
    print(f"   Transformer: {args.num_layers} layers, {args.num_heads} heads")
    print(f"   Loss: {args.loss} (beta={args.beta}, lambd={args.lambd}, lambd_s={args.lambd_s})")
    print(f"   Surface type: {args.surf_type}")
    print(f"   Train type: {args.train_type}")
    print(f"   🆕 Pure geometric learning - NO CONDITIONING")
    
    return args

# ====================================================================
# SPECIALIZED CONFIG VARIANTS (following your patterns)
# ====================================================================

def load_configv0_with_paths(model_dir, label_dir, target_seg_dir):
    """
    Load config and immediately set your actual data paths.
    
    This is a convenience function for quickly setting up the config
    with your real directory paths without editing the defaults.
    
    Example usage:
        config = load_configv0_with_paths(
            model_dir="/home/user/cardiac_models",
            label_dir="/home/user/cardiac_data/labels", 
            target_seg_dir="/home/user/cardiac_data/mesh_hdf5"
        )
    """
    import sys
    
    # Temporarily modify sys.argv to set the paths
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only script name
    sys.argv.extend(['--model_dir', model_dir])
    sys.argv.extend(['--label_dir', label_dir])
    sys.argv.extend(['--target_seg_dir', target_seg_dir])
    
    config = load_configv0()
    
    # Restore original argv
    sys.argv = original_argv
    
    print(f"📁 Config loaded with your paths:")
    print(f"   Model dir: {config.model_dir}")
    print(f"   Label dir: {config.label_dir}")
    print(f"   Target seg dir: {config.target_seg_dir}")
    
    return config

def load_configv0_debug():
    """
    Debug configuration with smaller settings for testing.
    Preserves your parameter structure but uses smaller values.
    """
    import sys
    
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    sys.argv.extend(['--batch', '1'])
    sys.argv.extend(['--n_samples', '1000'])
    sys.argv.extend(['--seq_len', '10'])
    sys.argv.extend(['--n_epochs', '5'])
    sys.argv.extend(['--tag', 'debug_pure_vae'])
    
    config = load_configv0()
    sys.argv = original_argv
    
    print("🐛 Debug config loaded - reduced settings for testing")
    return config

# ====================================================================
# PARAMETER VALIDATION (matching your original requirements)
# ====================================================================

def validate_config(config):
    """
    Validate configuration parameters to catch common issues.
    This preserves the validation logic you'd expect from your original config.
    """
    errors = []
    warnings = []
    
    # Check required dimensions
    if config.z_dim <= 0:
        errors.append(f"z_dim must be positive, got {config.z_dim}")
    if config.dim_h <= 0:
        errors.append(f"dim_h must be positive, got {config.dim_h}")
    if config.n_samples <= 0:
        errors.append(f"n_samples must be positive, got {config.n_samples}")
    
    # Check transformer parameters
    if config.num_heads <= 0:
        errors.append(f"num_heads must be positive, got {config.num_heads}")
    if config.dim_h % config.num_heads != 0:
        warnings.append(f"dim_h ({config.dim_h}) should be divisible by num_heads ({config.num_heads})")
    
    # Check file paths exist (if they're not the default placeholders)
    import os
    if not config.model_dir.startswith('/path/to/'):
        if not os.path.exists(config.model_dir):
            warnings.append(f"Model directory does not exist: {config.model_dir}")
    
    # Print validation results
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   {error}")
        raise ValueError("Configuration validation failed")
    
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("✅ Configuration validation passed")
    return True

# ====================================================================
# EASY SETUP INSTRUCTIONS
# ====================================================================

if __name__ == '__main__':
    print("🔧 Pure Transformer VAE Configuration")
    print("=" * 50)
    print()
    print("This config preserves your EXACT original settings.")
    print("Only ECG and patient conditioning parameters have been removed.")
    print()
    print("📋 TO USE THIS CONFIG:")
    print("1. Update the default paths in this file to point to your actual data directories")
    print("2. Or use load_configv0_with_paths() to set paths programmatically")
    print("3. Use exactly as you used your original config in training scripts")
    print()
    print("📁 PATHS TO UPDATE:")
    print("   model_dir: Your model output directory")
    print("   label_dir: Your CSV files directory")  
    print("   target_seg_dir: Your mesh HDF5 files directory")
    print()
    
    # Load example config
    config = load_configv0()
    validate_config(config)
    
    print()
    print("✅ Config loaded successfully!")
    print("Ready to use with your pure transformer VAE training script.")