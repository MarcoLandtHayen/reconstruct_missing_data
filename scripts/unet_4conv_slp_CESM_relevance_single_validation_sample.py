# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.
# Opposed to their net, only have 4 instead of 5 convolutional layers.
# Work with sea level pressure (slp) fields from Earth System Models, either FOCI or CESM.
#
# Trained model with variable mask and augmentation factor 3 on samples with rel. amount of missing values in the range of [0, 1].
# Divide training samples into patches and successively add more patches as input.
# Use this single model to find relevance - in terms of biggest loss reduction - in a brute-force manner.

import os
import sys

from json import dump, load
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.initializers as tfi
import tensorflow.keras.regularizers as tfr

sys.path.append(
    "GitHub/MarcoLandtHayen/reconstruct_missing_data/reconstruct_missing_data"
)

from data_loading import (
    clone_data,
    create_missing_mask,
    get_anomalies,
    load_data_set,
    split_and_scale_data,
)
from models import build_unet_4conv
from relevance import compute_single_relevance_map

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

# Suppress Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# # Reload final model, trained on range:

# Specify experiment:
model_config = 'unet_4conv'
feature = 'sea-level-pressure' # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = 'slp' # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = 'CESM' # Choose Earth System Model, either 'FOCI' or 'CESM'.

mask_type = 'variable'
missing_type = 'range_0_100'
augmentation_factor = 3
run = '_final'

# Get path to model:
path_to_model = Path('GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/'+model_config+'_'+feature_short+'_'+source+'_'
                      +mask_type+'_'+missing_type+'_factor_'+str(augmentation_factor)+run)

# Reload parameters for this experiment:
with open(path_to_model / 'parameters.json', 'r') as f:
    parameters=load(f)

train_val_split = parameters['train_val_split']
missing_values = parameters['missing_values']
scale_to = parameters['scale_to']

# Reload final model, trained on range:
model = tf.keras.models.load_model(path_to_model / 'missing_0_100' / 'model')

################ relevance_1
# Specify name of experiment, to store results accordingly in a separate folder:
exp_name = '/relevance_1'

# Set sample number to start from:
start_sample = 0

# Define number of validation samples to consider:
n_samples = 1

# Define patch size:
patch_sizes = [48,24,12,6,3]

# # Optionally define stopping criteria:

# Specify maximum number of patches to include (or set -1, to include ALL patches):
max_patch_num = -1

# Specify threshold for maximum accumulated rel. loss reduction (or set 1.0, for NO threshold):
max_acc_rel_loss_reduction = 1.0   

################



# Get path to store results to:
path_to_store_results = Path('GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/'+model_config+'_'+feature_short+'_'+source+'_'
                      +mask_type+'_'+missing_type+'_factor_'+str(augmentation_factor)+run+exp_name)

# Try to create folder for later saving results, avoid overwriting existing results:
# os.makedirs(path_to_store_results, exist_ok=False)

# Store parameters as json:
parameters = {
    "model_config": model_config,
    "feature": feature,
    "feature_short": feature_short,
    "source": source,
    "mask_type": mask_type,
    "missing_type": missing_type,
    "augmentation_factor": augmentation_factor,
    "run": run,
    "train_val_split": train_val_split,
    "scale_to": scale_to,
    "start_sample": start_sample,
    "n_samples": n_samples,
    "patch_sizes": patch_sizes,
    "max_patch_num": max_patch_num,
    "max_acc_rel_loss_reduction": max_acc_rel_loss_reduction,
}

with open(path_to_store_results / "parameters_.json", "w") as f:
    dump(parameters, f)

# ################


# # Load validation samples:

# Path to full data:
path_to_data = 'climate_index_collection/data/raw/2022-08-22/'

# Load data:
data = load_data_set(data_path=path_to_data, data_source_name=source)

# Select single feature and compute anomalies, using whole time span as climatology:
data = get_anomalies(feature=feature, data_set=data)

# Create synthetic missing_mask of ONEs and convert to boolean mask of TRUE, to load FULL validation samples:
missing_mask_1 = (np.ones(data.shape)==1)

# Get scaled training inputs. Note: Using missing_mask of ONEs, training inputs and targets are 
# identical. Only difference is found in dimensionality: inputs have channel number (=1) as final dimension, targets don't.
_, val_input, _, _, train_min, train_max, _, _ = split_and_scale_data(
    data, 
    missing_mask_1,
    train_val_split, 
    scale_to
)

## Loop over patch sizes:
for p in range(len(patch_sizes)):
    
    # Get current patch size:
    patch_size = patch_sizes[p]

    ## Compute rel. loss reduction maps, serving as relevance maps, for validation inputs and specified patch size:

    ## Need number of possible patches in advance, to initialize storages:

    # Get number of patches in lat and lon directions, respectively:
    n_lat = int(val_input[0:1].shape[1] / patch_size)
    n_lon = int(val_input[0:1].shape[2] / patch_size)

    # Obtain total number of patches:
    n_patches = int(n_lat * n_lon)

    # ## Individual output for EACH sample:
    # ##

    # Loop over samples:
    for n in np.arange(start_sample,start_sample+n_samples):

        # Compute relevance map, patch order and (acc.) rel. and abs. loss reduction for current sample with given patch size:
        (
            rel_loss_reduction_map, 
            patch_order, 
            abs_loss_reduction, 
            rel_loss_reduction, 
            acc_rel_loss_reduction
        ) = compute_single_relevance_map(input_sample=val_input[n:n+1],
                                         patch_size=patch_size, 
                                         model=model,
                                         max_patch_num=max_patch_num,
                                         max_acc_rel_loss_reduction=max_acc_rel_loss_reduction,
                                        )

        # Save output for this experiment, after each sample, to avoid data loss in case of 'out-of-memory event':
        file_name_rel_loss_reduction_map = "rel_loss_reduction_map_sample_"+str(n)+"_patchsize_"+str(patch_size)+".npy"
        file_name_patch_order = "patch_order_sample_"+str(n)+"_patchsize_"+str(patch_size)+".npy"
        file_name_abs_loss_reduction = "abs_loss_reduction_sample_"+str(n)+"_patchsize_"+str(patch_size)+".npy"
        file_name_rel_loss_reduction = "rel_loss_reduction_sample_"+str(n)+"_patchsize_"+str(patch_size)+".npy"
        file_name_acc_rel_loss_reduction = "acc_rel_loss_reduction_sample_"+str(n)+"_patchsize_"+str(patch_size)+".npy"
        np.save(path_to_store_results / file_name_rel_loss_reduction_map, rel_loss_reduction_map)
        np.save(path_to_store_results / file_name_patch_order, patch_order)
        np.save(path_to_store_results / file_name_abs_loss_reduction, abs_loss_reduction)
        np.save(path_to_store_results / file_name_rel_loss_reduction, rel_loss_reduction)
        np.save(path_to_store_results / file_name_acc_rel_loss_reduction, acc_rel_loss_reduction)
