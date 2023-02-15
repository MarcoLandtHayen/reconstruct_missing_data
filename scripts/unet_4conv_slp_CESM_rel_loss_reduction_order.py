# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.
# Opposed to their net, only have 4 instead of 5 convolutional layers.
# Work with sea level pressure (slp) fields from Earth System Models, either FOCI or CESM.
#
# So far, we look for the patch (or grid point), that leads to strongest loss reduction, when added. 
# This patch (or grid points) is then fixed, before we successively add more patches (or grid points). 
# But the question remains: Is that optimal? To answer this question, we start with large patch sizes (48 and 24) 
# and compute loss reduction for ALL possible combination of patches.


import os
import sys
sys.path.append(
    "GitHub/MarcoLandtHayen/reconstruct_missing_data/reconstruct_missing_data"
)

from pathlib import Path
from json import dump, load

import numpy as np
import pandas as pd
from itertools import permutations
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from data_loading import find_data_files, load_data_set, get_anomalies, clone_data, create_missing_mask, split_and_scale_data
from models import build_unet_4conv
from relevance import compute_single_relevance_map

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, Conv1D, Conv2D, MaxPool2D, UpSampling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.initializers as tfi
import tensorflow.keras.regularizers as tfr
from tensorflow.keras.utils import plot_model

# Suppress Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



## Reload parameters and pre-trained model:

# Specify experiment:
model_config = 'unet_4conv'
feature = 'sea-level-pressure' # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = 'slp' # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = 'CESM' # Choose Earth System Model, either 'FOCI' or 'CESM'.
mask_type = 'variable'
missing_type = 'range'
range_string = '_50_999'
augmentation_factor = 3
run = '_final'

# Get path to parameters:
path_to_parameters = Path('GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/'+model_config+'_'+feature_short+'_'+source+'_'
                      +mask_type+'_'+missing_type+range_string+'_factor_'+str(augmentation_factor)+run)

# Reload parameters relevant for data pre-processing for this experiment:
with open(path_to_parameters / 'parameters.json', 'r') as f:
    parameters=load(f)

seed = parameters['seed']
train_val_split = parameters['train_val_split']
scale_to = parameters['scale_to']
missing_values = parameters['missing_values']

# Get path to pre-trained model:
path_to_model = Path('GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/'+model_config+'_'+feature_short+'_'+source+'_'
                      +mask_type+'_'+missing_type+range_string+'_factor_'+str(augmentation_factor)+run+'/missing'+range_string+'/model')

# Reload final model, trained on range:
model = tf.keras.models.load_model(path_to_model)

## Prepare input samples:

# Path to full data:
path_to_data = 'climate_index_collection/data/raw/2022-08-22/'

# Load data:
data = load_data_set(data_path=path_to_data, data_source_name=source)

# Select single feature and compute anomalies, using whole time span as climatology:
data = get_anomalies(feature=feature, data_set=data)

# Create synthetic missing_mask of ONEs, to load FULL validation samples:
missing_mask_1 = (np.ones(data.shape)==1)

# Get scaled validation inputs and targets. Note: Using missing_mask of ONEs, validation inputs and targets are 
# identical. Only difference is found in dimensionality: inputs have channel number (=1) as final dimension, targets don't.
train_input, val_input, train_target, val_target, train_min, train_max, _, _ = split_and_scale_data(
    data, 
    missing_mask_1,
    train_val_split, 
    scale_to
)

########## Set further parameters:

# Set experiment name:
exp_name = 'order_8'

# Create directory to store results:
os.makedirs(path_to_parameters / exp_name, exist_ok=False)

# Set sample number to start from:
start_sample = 0

# Define number of validation samples to consider:
n_samples = 100

# Define list of patch sizes:
patch_sizes = [12,]

## Optionally define stopping criteria:

# Specify maximum number of patches to include (or set -1, to include ALL patches):
max_patch_num = 2

# Specify threshold for maximum accumulated rel. loss reduction (or set 1.0, for NO threshold):
max_acc_rel_loss_reduction = 1.0             

##########

## Loop over list of patch sizes:
for p in range(len(patch_sizes)):
    
    # Get status:
    print("patch size: ",p+1," of ",len(patch_sizes))
    
    # Get current patch size:
    patch_size = patch_sizes[p]
    
    # Get parameters for patches, enumerated line-by-line, from left to right, from top to bottom, starting with ZERO.
    n_lat = int(val_input[0:1].shape[1] / patch_size)
    n_lon = int(val_input[0:1].shape[2] / patch_size)

    # Obtain total number of patches:
    n_patches = int(n_lat * n_lon)
    
    # Check for maximum number of desired patches: If given as -1, set to total number of patches.
    if max_patch_num == -1:
        max_patch_num = n_patches
        
    ## Loop over samples:
    for s in range(n_samples):
    
        # Initialize storage for found orders:
        found_orders = []
        
        # Initialize storage for optimal patch orders:
        patch_orders = []
        
        # Initialize storage for checking identity of found and optimal orders:
        identical_checks = [] 
        
        # Get status:
        print("  sample: ",s+1," of ",n_samples)
    
        # Get current input sample:
        input_sample = train_input[start_sample+s:start_sample+s+1]
        
        # Compute relevance map, patch order and (acc.) rel. and abs. loss reduction for current sample with given patch size:
        (
            rel_loss_reduction_map, 
            patch_order, 
            abs_loss_reduction, 
            rel_loss_reduction, 
            acc_rel_loss_reduction
        ) = compute_single_relevance_map(input_sample=input_sample,
                                         patch_size=patch_size, 
                                         model=model,
                                         max_patch_num=max_patch_num,
                                         max_acc_rel_loss_reduction=max_acc_rel_loss_reduction,
                                        )
        
        ## Loop over possible number of patches to include. 
        ## Max. include max_patch_num patches. And start from 2, since single patch is not relevant. Look for permutations!
        for n in np.arange(2,max_patch_num+1):
            
            # Get status:
            print("    num. of included patches: ",n," of ",max_patch_num)
            
            # Get permutations of specified number of patches:
            permutation_list = list(permutations(range(n_patches),int(n)))
            
            # Create list of patch indices:
            patch_indices = list(np.arange(n_patches))

            # Create empty sample of just ZEROs:
            empty_sample = np.zeros((1, input_sample.shape[1], input_sample.shape[2]))

            ## Create patches:

            # Initialize storage for patches as boolean array. Dimension (# of permutations, latitude, longitude)
            patches = (np.zeros((len(permutation_list), input_sample.shape[1], input_sample.shape[2])) != 0)
            
            # Run over list of permutations:
            for l in range(len(permutation_list)):

                # Get current permutation:
                perm = permutation_list[l]

                # Loop over patch indices in current permutations:
                for patch_index in perm:

                    # Get x and y coordinate from current patch index:
                    y = patch_index // n_lon
                    x = patch_index % n_lon    

                    # Store mask for current patch:
                    patches[l,int(y*patch_size):int((y+1)*patch_size),int(x*patch_size):int((x+1)*patch_size)] = True 
                    
            # Expand dimensions of patches: Have last dimension for channel (=1), to match requirements for CNN inputs.
            patches_extended = np.expand_dims(patches, axis=-1)

            # Create input samples from first validation sample:
            patchy_input = patches_extended * input_sample

            # Get mean state on empty sample as input:
            mean_state = model.predict(empty_sample)

            # Get prediction from complete sample as input:
            full_pred = model.predict(input_sample)

            # Get model predictions on patchy inputs:
            patchy_pred = model.predict(patchy_input)
            
            # Compute mean state loss from prediction on empty sample compared to target (= complete input sample):
            mean_state_loss = np.mean((mean_state[:,:,:,0] - input_sample[0,:,:,0])**2)

            # Compute min loss from prediction on complete sample compared to target (= complete input sample):
            min_loss = np.mean((full_pred[:,:,:,0] - input_sample[0,:,:,0])**2)

            # Compute loss of patchy predictions compared to targets (= complete input sample):
            patchy_loss = np.mean((patchy_pred[:,:,:,0] - input_sample[0,:,:,0])**2,axis=(1,2))
            
            # Get index for patch leading to lowest loss, when adding:
            min_index = np.argsort(patchy_loss)[0]
            
            # Store found and optimal patch order and check, if both include the same initial patches:
            found_orders.append(permutation_list[min_index])
            patch_orders.append(patch_order[:n].astype(int))
            identical_checks.append(all(np.sort(permutation_list[min_index])==np.sort(patch_order[:n].astype(int))))
            
        # Define filenames to store informormation:
        found_orders_filename = 'found_orders_sample_'+str(start_sample+s)+'_patchsize_'+str(patch_size)+'_patches_'+str(max_patch_num)+'.npy'
        patch_orders_filename = 'patch_orders_sample_'+str(start_sample+s)+'_patchsize_'+str(patch_size)+'_patches_'+str(max_patch_num)+'.npy'
        identical_checks_filename = 'identity_checks_sample_'+str(start_sample+s)+'_patchsize_'+str(patch_size)+'_patches_'+str(max_patch_num)+'.npy'

        # Save files:
        np.save(path_to_parameters / exp_name / found_orders_filename, found_orders)
        np.save(path_to_parameters / exp_name / patch_orders_filename, patch_orders)
        np.save(path_to_parameters / exp_name / identical_checks_filename, identical_checks)
        
# Store parameters as json:
parameters = {
    "model_config": model_config,
    "feature": feature,
    "feature_short": feature_short,
    "source": source,
    "seed": seed,
    "mask_type": mask_type,
    "missing_type": missing_type,
    "range_string": range_string,
    "augmentation_factor": augmentation_factor,
    "run": run,
    "train_val_split": train_val_split,
    "missing_values": missing_values,
    "scale_to": scale_to,
    "start_sample": start_sample,
    "n_samples": n_samples,
    "patch_sizes": patch_sizes,
    "n_patches": n_patches,
    "max_patch_num": max_patch_num,
    "max_acc_rel_loss_reduction": max_acc_rel_loss_reduction
}

with open(path_to_parameters / exp_name / "parameters.json", "w") as f:
    dump(parameters, f)