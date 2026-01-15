import os
import torch
import math
import pickle

def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def age_transform(age):
    ## divide the age into 6 groups
    # <=50
    # years in[50,75], 5 years intervals
    # >=75
    if age < 55:
        return 0
    else:
        return min(math.floor((age - 50) // 5), 6 - 1)

def condition_normal_batch(age, gender,
                           normaltype=0, samesize=True,
                           NUM_AGES=6, NUM_GENDERS=2):
    ##normalization 0 is one-hot
    ##normalization 1 is as AgeHeart modified from str_to_tensor
    ##normal means the normalization is True

    batchsize= 1
    # initialize
    if normaltype==0:
        age_tensor = torch.zeros([batchsize, NUM_AGES])
        gender_tensor = torch.zeros([batchsize, NUM_GENDERS])
    else:
        age_tensor = -torch.ones([batchsize, NUM_AGES])
        gender_tensor = -torch.ones([batchsize, NUM_GENDERS])
    if samesize:
        gender_tensor = age_tensor


    for i in range(batchsize):
        if normaltype == 0:
            age_temp = torch.zeros(NUM_AGES)
            age_temp[int(age[i])] += 1
            gender_temp = torch.zeros(NUM_GENDERS)
            gender_temp[int(gender[i])] += 1
        else:
            age_temp = -torch.ones(NUM_AGES)
            age_temp[int(age[i])] *= -1
            gender_temp = -torch.ones(NUM_GENDERS)
            gender_temp[int(gender[i])] *= -1
        if samesize:#gender have the same weight as age
            gender_temp = gender_temp.repeat(NUM_AGES // NUM_GENDERS)#TODO: modify it, only be int,(6) at non-singleton dimension 0.  Target sizes: [7].  Tensor sizes: [6]

        age_tensor[i] = age_temp
        gender_tensor[i] = gender_temp
    return age_tensor, gender_tensor

def condition_normal(age, gender, dise, hypert,
                     normaltype=0, samesize=False,
                     NUM_AGES=6, NUM_GENDERS=2):
    ##normalization 0 is one-hot
    ##normalization 1 is as AgeHeart modified from str_to_tensor
    ##normal means the normalization is True

    if normaltype == 0:
        age_temp = torch.zeros(NUM_AGES)
        age_temp[int(age)] += 1
        gender_temp = torch.zeros(NUM_GENDERS)
        gender_temp[int(gender)] += 1
        # dise_temp = torch.zeros(NUM_GENDERS)
        # dise_temp[int(dise)] += 1
        # hypert_temp = torch.zeros(NUM_GENDERS)
        # hypert_temp[int(hypert)] += 1
    else:
        age_temp = -torch.ones(NUM_AGES)
        age_temp[int(age)] *= -1
        gender_temp = -torch.ones(NUM_GENDERS)
        gender_temp[int(gender)] *= -1
        # dise_temp = torch.ones(NUM_GENDERS)
        # dise_temp[int(dise)] *= -1
        # hypert_temp = torch.ones(NUM_GENDERS)
        # hypert_temp[int(hypert)] *= -1
    if samesize:
        ##gender have the same weight as age
        gender_temp = gender_temp.repeat(NUM_AGES // NUM_GENDERS)#TODO: modify it, only be int,(6) at non-singleton dimension 0.  Target sizes: [7].  Tensor sizes: [6]
        # dise_temp = dise_temp.repeat(NUM_AGES // NUM_GENDERS)
        # hypert_temp = hypert_temp.repeat(NUM_AGES // NUM_GENDERS)

    return age_temp,\
           gender_temp,\
           dise, \
           hypert

def pet_save(pet, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pet, f, pickle.HIGHEST_PROTOCOL)

def pet_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import numpy as np
def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

import os
import torch
import datetime

def extract_model_name_from_checkpoint(checkpoint_path):
    """
    Extract model name from checkpoint path for resuming training
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        str: Model name extracted from path
    """
    # Extract from path like: /path/to/model/MODEL_NAME/best_model.pt
    model_dir = os.path.dirname(checkpoint_path)
    model_name = os.path.basename(model_dir)
    return model_name

def load_checkpoint_and_setup_paths(config, model, optimizer, device, use_ecg=False):
    """
    Load checkpoint and setup model paths for training continuation or new training
    
    Args:
        config: Configuration object
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer to load state into
        device: Device to load tensors onto
        use_ecg (bool): Whether ECG features are being used
        
    Returns:
        dict: Contains all necessary information for training continuation
            - 'model_name': Model name for paths
            - 'start_epoch': Epoch to start/resume from
            - 'best_val_loss': Best validation loss so far
            - 'logdir': TensorBoard log directory
            - 'cp_path': Checkpoint save directory
            - 'best_model_path': Path to best model file
            - 'intermediate_model_path': Path to intermediate checkpoint file
            - 'is_resume': Boolean indicating if this is a resume operation
    """
    
    # Extract basic model parameters
    train_type = config.train_type
    z_dim = config.z_dim
    model_dir = config.model_dir
    checkpoint_file = getattr(config, 'checkpoint_file', None)
    
    # Initialize default values
    start_epoch = 0
    best_val_loss = float('inf')
    is_resume = False
    
    # --------------------------
    # Handle model naming for resume vs new training
    # --------------------------
    if checkpoint_file and os.path.exists(checkpoint_file):
        # RESUMING: Extract model name from existing checkpoint
        model_name = extract_model_name_from_checkpoint(checkpoint_file)
        is_resume = True
        print(f"🔄 RESUMING training from existing checkpoint")
        print(f"   Checkpoint: {checkpoint_file}")
        print(f"   Model name: {model_name}")
    else:
        # NEW TRAINING: Generate new model name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}"
        
        base_model_name = f"{train_type}_z_dim{z_dim}_loss_{config.loss}_beta{config.beta}_" \
                          f"lambd{config.lambd}_lambds{config.lambd_s}_lr{config.lr}_wd{config.wd}_batch{config.batch}"
        
        if use_ecg:
            ecg_dim = getattr(config, 'ecg_dim', 108)
            base_model_name += f"_ecg{ecg_dim}"
        
        model_name = f"{base_model_name}_{run_id}"
        print(f"🆕 STARTING new training")
        print(f"   Model name: {model_name}")

    # --------------------------
    # Setup paths
    # --------------------------
    logdir = f"{model_dir}/tb/{model_name}"
    cp_path = f"{model_dir}/model/{model_name}"
    best_model_path = f"{cp_path}/best_model.pt"
    intermediate_model_path = f"{cp_path}/intermediate_checkpoint.pt"
    
    # --------------------------
    # Load checkpoint if resuming
    # --------------------------
    if checkpoint_file:
        try:
            if os.path.exists(checkpoint_file):
                # Use the provided checkpoint path directly
                checkpoint_path = checkpoint_file
                print(f"📂 Loading checkpoint: {checkpoint_path}")
            elif os.path.exists(best_model_path):
                # Fallback to best model in current directory
                checkpoint_path = best_model_path
                print(f"📂 Loading best model checkpoint: {checkpoint_path}")
            elif os.path.exists(intermediate_model_path):
                # Fallback to intermediate checkpoint
                checkpoint_path = intermediate_model_path
                print(f"📂 Loading intermediate checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch_num'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            # Load model and optimizer states
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Ensure optimizer state is on correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            
            print(f"✅ Successfully loaded checkpoint from epoch {checkpoint['epoch_num']}")
            print(f"   Best validation loss so far: {best_val_loss:.6f}")
            print(f"   Resuming training from epoch {start_epoch}")
            print(f"   TensorBoard will continue in: {logdir}")
            print(f"   Models will be saved to: {cp_path}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_epoch = 0
            best_val_loss = float('inf')
            is_resume = False

    return {
        'model_name': model_name,
        'start_epoch': start_epoch,
        'best_val_loss': best_val_loss,
        'logdir': logdir,
        'cp_path': cp_path,
        'best_model_path': best_model_path,
        'intermediate_model_path': intermediate_model_path,
        'is_resume': is_resume
    }

def setup_training_directories(logdir, cp_path):
    """
    Setup training directories for tensorboard and checkpoints
    
    Args:
        logdir (str): TensorBoard log directory
        cp_path (str): Checkpoint save directory
    """
    import util.utils as util  # Assuming you have setup_dir in utils
    
    util.setup_dir(logdir)
    util.setup_dir(cp_path)
    
    print(f"📁 Directories setup:")
    print(f"   TensorBoard logs: {logdir}")
    print(f"   Model checkpoints: {cp_path}")