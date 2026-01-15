import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import datetime
import uuid

from data.dataloader_pure_vaev0 import PureGeometricMesh  # Our new pure dataloader
import util.utils as util
import loss.loss as Loss
from config_pure_vae import load_configv0
import timeit
from torch.utils.tensorboard import SummaryWriter
import model.transformer_pure_vae_batchv0 as PureVAE  # Our new pure model
import time
import pyvista as pv
import h5py

# Import the metrics module (unchanged)
from loss.metrics import (
    compute_mesh_metrics_for_sequence, 
    compute_batch_metrics, 
    EpochMetricsAccumulator,
    print_metrics_summary
)
from util.utils import load_checkpoint_and_setup_paths, setup_training_directories

def save_model(mesh_vae, optimizer, epoch, train_loss, val_loss, checkpoint_name, model_type="best"):
    """
    Save model checkpoint with additional metadata.
    This function remains unchanged as it's not related to conditioning.
    """
    checkpoint = {
        'state_dict': mesh_vae.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch_num': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_type': model_type,
        'save_time': datetime.datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_name)
    print(f"✅ {model_type.capitalize()} model saved: {checkpoint_name}")
    print(f"   Epoch: {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

def compute_batch_mesh_metrics(v_out, v_gt, patient_ids):
    """
    Compute mesh metrics for a batch - unchanged from original.
    This function works the same way since it only evaluates geometric reconstruction quality.
    """
    batch_size = v_out.shape[0]
    hd_values = []
    assd_values = []
    
    for b in range(batch_size):
        try:
            pred_seq = v_out[b]  # [seq_len, num_vertices, 3]
            gt_seq = v_gt[b]     # [seq_len, num_vertices, 3]
            
            avg_hd, avg_assd = compute_mesh_metrics_for_sequence(pred_seq, gt_seq)
            
            hd_values.append(avg_hd.item())
            assd_values.append(avg_assd.item())
            
        except Exception as e:
            print(f"Warning: Error computing metrics for patient {patient_ids[b] if patient_ids else b}: {e}")
            continue
    
    return hd_values, assd_values

def train(mesheart, trainloader, optimizer, device, config, writer, epoch):
    """
    Training function modified for pure geometric learning.
    
    Key changes:
    1. Removed use_ecg parameter - no longer needed
    2. Simplified data unpacking - only geometric data
    3. Removed all condition processing and passing
    4. Model forward pass takes only geometric inputs
    
    The training loop is now much cleaner and focuses purely on learning
    geometric patterns without any external bias from patient conditions.
    """
    mesheart.train()
    avg_loss = []
    batch_losses = []
    
    # Initialize metrics accumulator
    metrics_accumulator = EpochMetricsAccumulator()

    for idx, data in enumerate(trainloader):
        try:
            # SIMPLIFIED: Only unpack geometric data, no conditions
            # OLD: heart_v, heart_f, heart_e, con, ecg_con, subid = data (with conditions)
            # NEW: Only geometric data
            heart_v, heart_f, heart_e, subid = data
            
            optimizer.zero_grad()
            heart_v = heart_v.to(device)
            heart_f = heart_f.to(device) 
            heart_e = heart_e.to(device)
            
            # SIMPLIFIED: Forward pass with only geometric inputs
            # OLD: v_out, logvar, mu = mesheart(heart_v, heart_f, heart_e, con, ecg_con)
            # NEW: Pure geometric processing
            v_out, logvar, mu = mesheart(heart_v, heart_f, heart_e)

            # Loss computation remains the same (it's about reconstruction quality)
            loss, loss_recon = Loss.VAECELoss(v_out, heart_v, heart_f, logvar,
                                                  mu, beta=config.beta, lambd=config.lambd, 
                                                  lambd_s=config.lambd_s, loss=config.loss)
            avg_loss.append(loss.item())
            batch_losses.append(loss.item())
            
            # Compute mesh metrics for this batch
            try:
                with torch.no_grad():
                    hd_values, assd_values = compute_batch_mesh_metrics(v_out, heart_v, subid)
                    metrics_accumulator.update_raw_values(hd_values, assd_values)
            except Exception as e:
                print(f"Warning: Could not compute mesh metrics for batch {idx}: {e}")
            
            loss.backward()
            optimizer.step()
            
            # Log batch-level metrics occasionally
            if idx % 10 == 0 and writer is not None:
                global_step = epoch * len(trainloader) + idx
                writer.add_scalar('Train_Loss_Batch', loss.item(), global_step)
                writer.add_scalar('Train_Loss_Recon_Batch', loss_recon.item(), global_step)
                
                if hd_values and assd_values:
                    batch_hd = np.mean(hd_values)
                    batch_assd = np.mean(assd_values)
                    writer.add_scalar('Train_HD_Batch', batch_hd, global_step)
                    writer.add_scalar('Train_ASSD_Batch', batch_assd, global_step)
                
        except Exception as e:
            print(f"Error in training batch {idx}: {e}")
            continue

    epoch_loss = np.mean(avg_loss)
    epoch_metrics = metrics_accumulator.compute_epoch_stats()
    
    print(f'Epoch {epoch}, Loss: {epoch_loss:.6f}')
    print_metrics_summary(epoch_metrics, "  Train ")
    
    # Enhanced tensorboard logging
    if writer is not None:
        writer.add_scalar('Train_Loss_Std', np.std(batch_losses), epoch)
        writer.add_scalar('Train_Batches_Processed', len(batch_losses), epoch)
        writer.add_scalar('Train_HD_Mean', epoch_metrics['hd_mean'], epoch)
        writer.add_scalar('Train_HD_Median', epoch_metrics['hd_median'], epoch)
        writer.add_scalar('Train_ASSD_Mean', epoch_metrics['assd_mean'], epoch)
        writer.add_scalar('Train_ASSD_Median', epoch_metrics['assd_median'], epoch)
    
    return epoch_loss, epoch_metrics

def val(mesheart, validloader, optimizer, device, config, writer, epoch):
    """
    Validation function modified for pure geometric learning.
    
    Same key changes as training function:
    - Simplified data unpacking
    - Removed condition processing
    - Pure geometric forward pass
    """
    print('-------------validation--------------')
    mesheart.eval()
    
    metrics_accumulator = EpochMetricsAccumulator()
    
    with torch.no_grad():
        valid_error = []
        for idx, data in enumerate(validloader):
            try:
                # SIMPLIFIED: Only geometric data
                myo_v, myo_f, myo_e, subid = data

                myo_v = myo_v.to(device)
                myo_f = myo_f.to(device) 
                myo_e = myo_e.to(device)

                # SIMPLIFIED: Pure geometric forward pass
                v_out, logvar, mu = mesheart(myo_v, myo_f, myo_e)

                # Loss computation (unchanged)
                loss, loss_recon = Loss.VAECELoss(v_out, myo_v, myo_f, logvar,
                                                  mu, beta=config.beta, lambd=config.lambd, 
                                                  lambd_s=config.lambd_s, loss=config.loss)
                valid_error.append(loss_recon)
                
                # Compute mesh metrics
                try:
                    hd_values, assd_values = compute_batch_mesh_metrics(v_out, myo_v, subid)
                    metrics_accumulator.update_raw_values(hd_values, assd_values)
                except Exception as e:
                    print(f"Warning: Could not compute mesh metrics for validation batch {idx}: {e}")
                
            except Exception as e:
                print(f"Error in validation batch {idx}: {e}")
                continue

        if valid_error:
            this_val_error = torch.mean(torch.stack(valid_error))
        else:
            print("Warning: No valid validation batches processed")
            this_val_error = float('inf')

        epoch_metrics = metrics_accumulator.compute_epoch_stats()

        print(f'Epoch {epoch}, Validation Error: {this_val_error:.6f}')
        print_metrics_summary(epoch_metrics, "  Val ")
        print('-------------------------------------')
        
        # Log validation metrics
        if writer is not None:
            writer.add_scalar('Val_HD_Mean', epoch_metrics['hd_mean'], epoch)
            writer.add_scalar('Val_HD_Median', epoch_metrics['hd_median'], epoch)
            writer.add_scalar('Val_ASSD_Mean', epoch_metrics['assd_mean'], epoch)
            writer.add_scalar('Val_ASSD_Median', epoch_metrics['assd_median'], epoch)
        
        return this_val_error, epoch_metrics

def log_hparams(writer, config, train_type, tag, final_train_loss=None, final_val_loss=None, 
                best_val_loss=None):
    """
    Log hyperparameters to TensorBoard - simplified for pure VAE.
    
    Removed ECG-related hyperparameters since we no longer use conditioning.
    """
    hparams = {
        'train_type': train_type,
        'tag': tag,
        'learning_rate': config.lr,
        'batch_size': config.batch,
        'z_dim': config.z_dim,
        'n_epochs': config.n_epochs,
        'beta': config.beta,
        'lambd': config.lambd,
        'lambd_s': config.lambd_s,
        'loss_type': config.loss,
        'dim_h': config.dim_h,
        'n_samples': config.n_samples,
        'seq_len': getattr(config, 'seq_len', 50),
        'ff_size': getattr(config, 'ff_size', 1024),
        'num_heads': getattr(config, 'num_heads', 4),
        'num_layers': getattr(config, 'num_layers', 2),
        'activation': getattr(config, 'activation', 'gelu'),
        'weight_decay': getattr(config, 'wd', 0.0),
        'surf_type': config.surf_type,
        'val_freq': getattr(config, 'val_freq', 1),
        'model_type': 'pure_transformer_vae',  # Indicate this is the pure version
    }
    
    metrics = {
        'hparam/final_train_loss': final_train_loss if final_train_loss is not None else 0.0,
        'hparam/final_val_loss': final_val_loss if final_val_loss is not None else 0.0,
        'hparam/best_val_loss': best_val_loss if best_val_loss is not None else 0.0,
    }
    
    writer.add_hparams(hparams, metrics)
    
    print("✅ Pure VAE HParams logged to TensorBoard:")
    print(f"   Model type: Pure Transformer VAE (no conditioning)")
    print(f"   Hyperparameters: {len(hparams)} items")
    print(f"   Final train loss: {final_train_loss}")
    print(f"   Final val loss: {final_val_loss}")
    print(f"   Best val loss: {best_val_loss}")

def main(config):
    """
    Main training function for pure transformer VAE.
    
    Key architectural changes:
    1. Removed all ECG and condition configuration
    2. Simplified model creation with only geometric parameters
    3. Streamlined training loop without condition handling
    4. Focus on pure latent space learning from geometry
    """
    # Basic configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    
    # REMOVED: All ECG and conditioning configuration
    # No more use_ecg, ecg_dim_in, ecg_dim parameters needed
    
    age_group = getattr(config, 'age_group', None)
    tag = getattr(config, 'tag', 'pure_vae')
    
    print('🔧 Pure Transformer VAE Configuration:')
    print('   Model Type: Pure Geometric Learning (NO CONDITIONING)')
    print('   Focus: Learning cardiac motion patterns from geometry only')
    print('   Benefits: Simpler model, faster training, pure latent representations')
    
    model_dir = config.model_dir
    device = config.device
    train_type = config.train_type
    surf_type = config.surf_type

    n_epochs = config.n_epochs
    n_samples = config.n_samples
    lr = config.lr

    z_dim = config.z_dim
    channal = 3  # xyz coordinates
    C = config.dim_h

    # REMOVED: Condition-related parameters
    # No more con_num_in, con_num_out, ecg processing

    print("\n=== PURE VAE PARAMETERS ===")
    print(f"channal (input dim): {channal}")
    print(f"C (hidden dim): {C}")
    print(f"z_dim (latent dim): {z_dim}")
    print(f"n_samples (points): {n_samples}")
    print(f"train_type: {train_type}")
    print(f"Model focus: PURE GEOMETRIC LEARNING")
    print("==============================\n")

    # Set defaults
    seq_len = getattr(config, 'seq_len', 50)
    ff_size = getattr(config, 'ff_size', 1024)
    num_heads = getattr(config, 'num_heads', 4)
    activation = getattr(config, 'activation', 'gelu')
    num_layers = getattr(config, 'num_layers', 2)

    start = timeit.default_timer()

    # Create pure geometric datasets
    try:
        print("📊 Creating pure geometric datasets...")
        trainset = PureGeometricMesh(config, 'train')
        validset = PureGeometricMesh(config, 'val')
        print(f"✅ Datasets created successfully")
            
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return

    trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True, num_workers=config.num_workers)
    validloader = DataLoader(validset, batch_size=config.batch, shuffle=False, num_workers=config.num_workers)

    print(f"📈 Dataset Statistics:")
    print(f"   Training samples: {len(trainset)}")
    print(f"   Validation samples: {len(validset)}")
    print(f"   Batch size: {config.batch}")
    print(f"   Data type: Pure geometric (mesh vertices, faces, edges only)")

    # Create pure transformer VAE model
    try:
        print("\n🏗️  Creating Pure Transformer VAE...")
        print(f"   Architecture: Transformer-based encoder-decoder")
        print(f"   Input: Mesh sequences (vertices, faces, edges)")
        print(f"   Output: Reconstructed mesh sequences")
        print(f"   Latent space: {z_dim}D continuous representation")
        print(f"   Learning objective: Geometric pattern reconstruction")
        
        mesheart = PureVAE.PureTransformerVAE(
            dim_in=channal,          # 3D coordinates
            dim_h=C,                 # Hidden dimensions
            z_dim=z_dim,             # Latent dimensions
            points=n_samples,        # Points per mesh
            seq_len=seq_len,         # Sequence length
            ff_size=ff_size,         # Transformer feed-forward size
            num_heads=num_heads,     # Attention heads
            activation=activation,   # Activation function
            num_layers=num_layers    # Transformer layers
        ).to(device)
        
        print("✅ Pure Transformer VAE created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Setup optimizer
    if config.wd:
        optimizer = optim.Adam(mesheart.parameters(), lr=lr, weight_decay=config.wd)
    else:
        optimizer = optim.Adam(mesheart.parameters(), lr=lr)

    # Setup training directories and checkpoints
    checkpoint_info = load_checkpoint_and_setup_paths(config, mesheart, optimizer, device, use_ecg=False)
    
    model_name = checkpoint_info['model_name']
    start_epoch = checkpoint_info['start_epoch']
    best_val_loss = checkpoint_info['best_val_loss']
    logdir = checkpoint_info['logdir']
    cp_path = checkpoint_info['cp_path']
    best_model_path = checkpoint_info['best_model_path']
    intermediate_model_path = checkpoint_info['intermediate_model_path']
    is_resume = checkpoint_info['is_resume']
    
    setup_training_directories(logdir, cp_path)
    writer = SummaryWriter(logdir)
    
    mesheart.to(device)

    # Test model with sample data
    print("\n🧪 Testing Pure VAE with sample data...")
    try:
        sample_batch = next(iter(trainloader))
        heart_v, heart_f, heart_e, subid = sample_batch  # Only geometric data
        
        print(f"✅ Sample batch loaded:")
        print(f"   Vertices: {heart_v.shape}")
        print(f"   Faces: {heart_f.shape}")
        print(f"   Edges: {heart_e.shape}")
        print(f"   Subject IDs: {len(subid)}")
        
        heart_v = heart_v.to(device)
        heart_f = heart_f.to(device)
        heart_e = heart_e.to(device)
        
        with torch.no_grad():
            # Test pure geometric forward pass
            v_out, logvar, mu = mesheart(heart_v, heart_f, heart_e)
            print(f"✅ Model test successful:")
            print(f"   Input shape: {heart_v.shape}")
            print(f"   Output shape: {v_out.shape}")
            print(f"   Latent mu: {mu.shape}")
            print(f"   Latent logvar: {logvar.shape}")
            
            # Test mesh metrics
            hd_values, assd_values = compute_batch_mesh_metrics(v_out, heart_v, subid)
            print(f"✅ Metrics test successful:")
            print(f"   Hausdorff Distance: {np.mean(hd_values):.4f}")
            print(f"   ASSD: {np.mean(assd_values):.4f}")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Training loop
    print(f"\n🚀 Starting Pure Transformer VAE Training...")
    if is_resume:
        print(f"📊 Resuming from epoch {start_epoch} to {n_epochs}")
    else:
        print(f"📊 New training from epoch {start_epoch} to {n_epochs}")
    print(f"🎯 Current best validation loss: {best_val_loss:.6f}")
    print(f"🎯 Objective: Learn pure geometric cardiac motion patterns")
    print("-" * 80)
    
    final_train_loss = None
    final_val_loss = None
    val_freq = getattr(config, 'val_freq', 1)
    
    for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training Pure VAE"):
        epoch_start_time = time.time()
        
        # Training with pure geometric data
        train_loss, train_metrics = train(mesheart, trainloader, optimizer, device, config, writer, epoch)
        final_train_loss = train_loss
        
        # Validation and model saving
        if epoch % val_freq == 0:
            val_loss, val_metrics = val(mesheart, validloader, optimizer, device, config, writer, epoch)
            final_val_loss = val_loss
            
            # Save best model
            if val_loss < best_val_loss:
                print(f"🎉 New best validation loss! {best_val_loss:.6f} -> {val_loss:.6f}")
                best_val_loss = val_loss
                save_model(mesheart, optimizer, epoch, train_loss, val_loss, 
                          best_model_path, model_type="best")
            else:
                print(f"📈 Validation loss: {val_loss:.6f} (best: {best_val_loss:.6f})")
            
            # Save intermediate checkpoint
            if epoch % 10 == 0 and epoch > 0:
                save_model(mesheart, optimizer, epoch, train_loss, val_loss, 
                          intermediate_model_path, model_type="intermediate")
            
            # Tensorboard logging
            try:
                writer.add_scalar('Train_Loss', train_loss, epoch)
                writer.add_scalar('Val_Loss', val_loss, epoch)
                writer.add_scalar('Best_Val_Loss', best_val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                
            except Exception as e:
                print(f"Error writing to tensorboard: {e}")
            
        # Timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_duration // 60:.0f}:{epoch_duration % 60:.2f}")

    # Training completion
    print("\n" + "="*80)
    print("🎊 Pure Transformer VAE Training Completed!")
    print(f"📈 Final train loss: {final_train_loss:.6f}")
    print(f"📉 Final validation loss: {final_val_loss:.6f}")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    print(f"💾 Best model saved at: {best_model_path}")
    print(f"🎯 Achievement: Pure geometric learning without conditioning bias!")
    print("="*80)
    
    # Log final results
    log_hparams(writer, config, train_type, tag, final_train_loss, final_val_loss, best_val_loss)
    
    writer.add_scalar('Summary/Best_Val_Loss', best_val_loss, n_epochs)
    writer.add_scalar('Summary/Final_Train_Loss', final_train_loss, n_epochs)
    writer.add_scalar('Summary/Final_Val_Loss', final_val_loss, n_epochs)
    writer.add_scalar('Summary/Total_Epochs', n_epochs, n_epochs)
    
    writer.close()
    print("✅ Pure Transformer VAE training and logging complete!")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    config = load_configv0()
    main(config)