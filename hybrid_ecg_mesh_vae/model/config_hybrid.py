"""
Configuration for Hybrid ECG-to-Mesh VAE

Merges parameters from:
1. EchoNext (ECG encoder configuration)
2. MeshHeart (Mesh decoder configuration)

Adds new parameters for:
- Data paths for both ECG and mesh data
- Pretrained checkpoint paths for transfer learning
"""

import argparse
import torch


def load_config():
    """
    Load configuration for Hybrid ECG-to-Mesh VAE training.

    This config combines parameters from both EchoNext and MeshHeart projects,
    ensuring all necessary settings for the hybrid model.
    """

    parser = argparse.ArgumentParser(description='Hybrid ECG-to-Mesh VAE Training')
    # Notes
    parser.add_argument('--key_notes', type=str, default='Fine tune beta 0.001 + no weight decay',
                        help='Key notes for this experiment')
    
    # ====================================================================
    # HARDWARE SETTINGS
    # ====================================================================
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')
    parser.add_argument('--num_workers', type=int, default=1, #16,
                       help='DataLoader workers (0 for main process only, reduces memory overhead)')

    # ====================================================================
    # TRAINING HYPERPARAMETERS
    # ====================================================================
    parser.add_argument('--batch', type=int, default=256, help='Batch size (per GPU)')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps (effective_batch = batch * accumulation_steps)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0, #1e-4,
                        help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')

    # Chunked training for large datasets
    parser.add_argument('--chunk_size', type=int, default=1024,
                       help='Number of samples per training chunk (e.g., 10000). '
                            'If None, trains on full epoch. Useful for very large datasets '
                            'to get more frequent validation feedback.')

    # Gradient stability
    parser.add_argument('--grad_clip_value', type=float, default=1e6, #1e6,
                       help='Gradient clipping max norm to prevent exploding gradients. '
                            'Set to None to disable gradient clipping.')

    # Metrics computation frequency
    parser.add_argument('--compute_train_metrics_freq', type=int, default=0,
                       help='Frequency of computing HD/ASSD metrics during training. '
                            '0 = disabled (recommended for speed), '
                            'N = compute every N batches, '
                            '-1 = compute every batch (slowest). '
                            'Validation metrics are always computed.')

    # Learning rate scheduler (chunk-based for chunked training)
    parser.add_argument('--use_scheduler', type=bool, default=False,
                       help='Use learning rate scheduler (chunk-based)')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'linear_warmup'],
                       help='Type of LR scheduler')
    parser.add_argument('--warmup_chunks', type=int, default=50,
                       help='Number of chunks for LR warmup (not epochs!)')
    parser.add_argument('--total_chunks_for_cosine', type=int, default=50,
                       help='Total chunks for cosine annealing cycle. '
                            'Default: n_epochs * chunks_per_epoch')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='Minimum learning rate for scheduler')

    # ====================================================================
    # MODEL ARCHITECTURE - LATENT SPACE
    # ====================================================================
    parser.add_argument('--z_dim', type=int, default=64,
                       help='Latent dimension (configurable, default 64)')
    parser.add_argument('--use_motion_denorm', type=bool, default=True,
                       help='Apply motion scaler inverse transform to denormalize latents before decoder. '
                            'This should match how the pretrained decoder was trained. Default: True')

    # ====================================================================
    # ECG ENCODER PARAMETERS
    # ====================================================================
    parser.add_argument('--ecg_filter_size', type=int, default=64,
                       help='Filter size for ResNet1D encoder')
    parser.add_argument('--ecg_dropout', type=float, default=0.5,
                       help='Dropout for ECG encoder')
    parser.add_argument('--ecg_conv1_kernel_size', type=int, default=15,
                       help='First conv layer kernel size')
    parser.add_argument('--ecg_conv1_stride', type=int, default=2,
                       help='First conv layer stride')
    parser.add_argument('--ecg_padding', type=int, default=7,
                       help='Conv layer padding')

    # ====================================================================
    # MESH DECODER PARAMETERS
    # ====================================================================
    parser.add_argument('--n_samples', type=int, default=1412,
                       help='Number of mesh vertices')
    parser.add_argument('--seq_len', type=int, default=50,
                       help='Sequence length (number of frames)')
    parser.add_argument('--ff_size', type=int, default=1024,
                       help='Transformer feed-forward size')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--activation', type=str, default='gelu',
                       help='Activation function')
    parser.add_argument('--decoder_dropout', type=float, default=0.1,
                       help='Dropout for mesh decoder')

    # ====================================================================
    # LOSS FUNCTION WEIGHTS
    # ====================================================================
    parser.add_argument('--beta', type=float, default= 0.001, # 0.01
                       help='KL divergence weight (beta-VAE parameter)')
    parser.add_argument('--lambd', type=float, default=1.0,
                       help='Reconstruction loss weight')
    parser.add_argument('--lambd_s', type=float, default=1.0, # 1.0
                       help='Smoothness loss weight')
    parser.add_argument('--loss', type=str, default= 'cham_smooth', #'cham_smooth', Note cham_smooth and mse_smooth are set inversely
                       help='Loss function type (cham_smooth or mse_smooth)')

    # ====================================================================
    # DATA PATHS - ECG DATA
    # ====================================================================
    parser.add_argument('--train_csv', type=str,
                       default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt64_12ecg64_71k/train_ecg_motion.csv",
                       help='Path to training CSV with patient info (eid_18545 and eid_40616)')
    parser.add_argument('--val_csv', type=str,
                       default="ukb/jz_meshheart/flow_match_raw/MeshHeart_ltnt64_12ecg64_71k/val_ecg_motion.csv",
                       help='Path to validation CSV with patient info')
    parser.add_argument('--preprocessed_ecg_path', type=str,
                       default="ukb/jz_ecg/ecg_echonext_15Oct25/preprocessed_ecg_12x2500_v1_15Oct25_parallel.pt",
                       help='Path to preprocessed ECG .pt file')
    parser.add_argument('--ecg_phenotypes_path', type=str,
                       default="cardiac/pi514/ukbb_ecg/Final/2_Factor_ECG/data/pt_data_ecg/ecg_phenotypes.csv",
                       help='Path to ECG morphology phenotypes CSV')

    # ====================================================================
    # DATA PATHS - MESH DATA
    # ====================================================================
    parser.add_argument('--target_seg_dir', type=str,
                       default="ukb/jz_meshheart/raw_data_71k_26Aug25/meshes",
                       help='Directory containing mesh HDF5 files')
    parser.add_argument('--surf_type', type=str, default='all',
                       choices=['all', 'sample'],
                       help='Mesh resolution type')

    # ====================================================================
    # DATA PATHS - MODEL OUTPUTS
    # ====================================================================
    parser.add_argument('--model_dir', type=str,
                       default="jzheng12/Codes/ECG_MeshHeart/output/init_16Oct25",
                       help='Base directory for model outputs')

    # ====================================================================
    # PRETRAINED CHECKPOINT INITIALIZATION
    # ====================================================================
    parser.add_argument('--load_pretrained', type=bool, default=True,
                       help='Load pretrained weights from individual models')
    parser.add_argument('--ecg_encoder_checkpoint', type=str,
                       default='jzheng12/Codes/FactorECG/output/echonext_preprocess_motion_vae/checkpoints_20251022_092526/best_checkpoint.ckpt',
                       help='Path to EchoNext ECG encoder VAE checkpoint (.ckpt)')
    parser.add_argument('--mesh_decoder_checkpoint', type=str,
                       default='jzheng12/Codes/MeshHeart/experiments/model/pure_geometric_z_dim64_loss_cham_smooth_beta0.01_lambd1.0_lambds1.0_lr0.0001_wdNone_batch16_20250723_133038/best_model.pt',
                       help='Path to MeshHeart mesh decoder checkpoint (.pt)')

    # ====================================================================
    # CHECKPOINT RESUMPTION
    # ====================================================================
    parser.add_argument('--checkpoint_file', type=str, default=None,
                       help='Path to checkpoint file for resuming training')

    # ====================================================================
    # DATA PROCESSING
    # ====================================================================
    parser.add_argument('--normalize', type=bool, default=True,
                       help='Normalize inputs')

    # ====================================================================
    # EXPERIMENT ORGANIZATION
    # ====================================================================
    parser.add_argument('--train_type', type=str, default='hybrid_ecg_mesh',
                       help='Training type identifier')
    parser.add_argument('--tag', type=str, default='hybrid_vae',
                       help='Experiment tag for identification')

    args = parser.parse_args()

    # ====================================================================
    # POST-PROCESSING
    # ====================================================================

    # Set device properly
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
        print("⚠️  Warning: CUDA not available, using CPU")

    # Print configuration summary
    print("=" * 80)
    print("🔧 Hybrid ECG-to-Mesh VAE Configuration")
    print("=" * 80)
    print(f"\n📊 Model Architecture:")
    print(f"   Latent dimension: {args.z_dim}")
    print(f"   Motion denormalization: {args.use_motion_denorm}")
    print(f"   Sequence length: {args.seq_len} frames")
    print(f"   Mesh vertices: {args.n_samples}")
    print(f"\n🧠 ECG Encoder:")
    print(f"   Filter size: {args.ecg_filter_size}")
    print(f"   Dropout: {args.ecg_dropout}")
    print(f"\n🎨 Mesh Decoder:")
    print(f"   Transformer layers: {args.num_layers}")
    print(f"   Attention heads: {args.num_heads}")
    print(f"   Feed-forward size: {args.ff_size}")
    print(f"\n📉 Loss Configuration:")
    print(f"   Loss type: {args.loss}")
    print(f"   Beta (KL weight): {args.beta}")
    print(f"   Lambda (recon weight): {args.lambd}")
    print(f"   Lambda_s (smooth weight): {args.lambd_s}")
    print(f"\n🎯 Training:")
    print(f"   Device: {args.device}")
    print(f"   Batch size: {args.batch}")
    print(f"   Accumulation steps: {args.accumulation_steps}")
    print(f"   Effective batch size: {args.batch * args.accumulation_steps}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Epochs: {args.n_epochs}")
    print(f"   Weight decay: {args.wd}")
    print(f"\n💾 Pretrained Initialization:")
    print(f"   Load pretrained: {args.load_pretrained}")
    if args.load_pretrained:
        print(f"   ECG encoder checkpoint: {args.ecg_encoder_checkpoint}")
        print(f"   Mesh decoder checkpoint: {args.mesh_decoder_checkpoint}")
    print("=" * 80)

    return args


def load_config_with_paths(
    train_csv: str,
    val_csv: str,
    preprocessed_ecg_path: str,
    ecg_phenotypes_path: str,
    target_seg_dir: str,
    model_dir: str,
    ecg_encoder_checkpoint: str = None,
    mesh_decoder_checkpoint: str = None
):
    """
    Convenience function to load config with specific paths.

    Example usage:
        config = load_config_with_paths(
            train_csv="/data/train.csv",
            val_csv="/data/val.csv",
            preprocessed_ecg_path="/data/ecg.pt",
            ecg_phenotypes_path="/data/phenotypes.csv",
            target_seg_dir="/data/meshes/",
            model_dir="./experiments",
            ecg_encoder_checkpoint="./echonext/best.ckpt",
            mesh_decoder_checkpoint="./meshheart/best.pt"
        )
    """
    import sys

    # Temporarily modify sys.argv
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    sys.argv.extend(['--train_csv', train_csv])
    sys.argv.extend(['--val_csv', val_csv])
    sys.argv.extend(['--preprocessed_ecg_path', preprocessed_ecg_path])
    sys.argv.extend(['--ecg_phenotypes_path', ecg_phenotypes_path])
    sys.argv.extend(['--target_seg_dir', target_seg_dir])
    sys.argv.extend(['--model_dir', model_dir])

    if ecg_encoder_checkpoint:
        sys.argv.extend(['--ecg_encoder_checkpoint', ecg_encoder_checkpoint])
    if mesh_decoder_checkpoint:
        sys.argv.extend(['--mesh_decoder_checkpoint', mesh_decoder_checkpoint])

    config = load_config()

    # Restore original argv
    sys.argv = original_argv

    print(f"\n📁 Config loaded with specified paths")
    return config


def validate_config(config):
    """
    Validate configuration parameters to catch common issues.
    """
    import os

    errors = []
    warnings = []

    # Check model parameters
    if config.z_dim <= 0:
        errors.append(f"z_dim must be positive, got {config.z_dim}")
    if config.n_samples <= 0:
        errors.append(f"n_samples must be positive, got {config.n_samples}")

    # Check transformer parameters
    if config.num_heads <= 0:
        errors.append(f"num_heads must be positive, got {config.num_heads}")

    # Check data paths
    if not os.path.exists(config.train_csv):
        warnings.append(f"Training CSV not found: {config.train_csv}")
    if not os.path.exists(config.val_csv):
        warnings.append(f"Validation CSV not found: {config.val_csv}")
    if not os.path.exists(config.preprocessed_ecg_path):
        warnings.append(f"Preprocessed ECG not found: {config.preprocessed_ecg_path}")
    if not os.path.exists(config.ecg_phenotypes_path):
        warnings.append(f"ECG phenotypes not found: {config.ecg_phenotypes_path}")
    if not os.path.exists(config.target_seg_dir):
        warnings.append(f"Mesh directory not found: {config.target_seg_dir}")

    # Check pretrained checkpoints if requested
    if config.load_pretrained:
        if config.ecg_encoder_checkpoint and not os.path.exists(config.ecg_encoder_checkpoint):
            warnings.append(f"ECG encoder checkpoint not found: {config.ecg_encoder_checkpoint}")
        if config.mesh_decoder_checkpoint and not os.path.exists(config.mesh_decoder_checkpoint):
            warnings.append(f"Mesh decoder checkpoint not found: {config.mesh_decoder_checkpoint}")

    # Print validation results
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"   {error}")
        raise ValueError("Configuration validation failed")

    if warnings:
        print("\n⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   {warning}")

    print("\n✅ Configuration validation passed")
    return True


if __name__ == '__main__':
    print("Testing Hybrid VAE Configuration\n")

    # Load config
    config = load_config()

    # Validate
    try:
        validate_config(config)
    except ValueError as e:
        print(f"\n{e}")
        print("\nPlease update the configuration paths before training.")
