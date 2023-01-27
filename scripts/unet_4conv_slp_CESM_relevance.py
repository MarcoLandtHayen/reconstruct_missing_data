# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.
# Opposed to their net, only have 4 instead of 5 convolutional layers.
# Work with sea level pressure (slp) fields from Earth System Models, either FOCI or CESM.
#
# Trained model with variable mask and augmentation factor 2 on samples with rel. amount of missing values in the range of [0.75, 0.99].
# Divide validation samples into patches and successively add more patches as input.
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


## Reload final model, trained on range:

# Specify experiment:
model_config = 'unet_4conv'
feature = 'sea-level-pressure' # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = 'slp' # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = 'CESM' # Choose Earth System Model, either 'FOCI' or 'CESM'.

mask_type = 'variable'
missing_type = 'range'
augmentation_factor = 2
run = '_final'

# Get path to stored validation loss from batch:
path_to_model = Path('GitGeomar/marco-landt-hayen/reconstruct_missing_data/results/'+model_config+'_'+feature_short+'_'+source+'_'
                      +mask_type+'_'+missing_type+'_factor_'+str(augmentation_factor)+run)

# Reload parameters for this experiment:
with open(path_to_model / 'parameters.json', 'r') as f:
    parameters=load(f)

train_val_split = parameters['train_val_split']
missing_values = parameters['missing_values']
scale_to = parameters['scale_to']

# Reload final model, trained on range:
model = tf.keras.models.load_model(path_to_model / 'missing_75_99' / 'model')

## Load validation samples:

# Path to full data:
path_to_data = 'climate_index_collection/data/raw/2022-08-22/'

# Load data:
data = load_data_set(data_path=path_to_data, data_source_name=source)

# Select single feature and compute anomalies, using whole time span as climatology:
data = get_anomalies(feature=feature, data_set=data)

# Create synthetic missing_mask of ONEs, to load FULL validation samples:
missing_mask_1 = np.ones(data.shape)

# Get scaled validation inputs and targets. Note: Using missing_mask of ONEs, validation inputs and targets are 
# identical. Only difference is found in dimensionality: inputs have channel number (=1) as final dimension, targets don't.
train_input, val_input, train_target, val_target, train_min, train_max, train_mean, train_std = split_and_scale_data(
    data, 
    missing_mask_1,
    train_val_split, 
    scale_to
)

## Compute rel. loss reduction maps, serving as relevance maps, for several validation inputs and various patch sizes:

# Define patch sizes:
patch_sizes = [48, 24, 12, 6]

# Define number of validation samples:
n_samples = 2

# Initialize storage for resulting relevance maps, dimension: (#samples, #patch sizes, latitude, longitude)
rel_loss_reduction_maps = np.zeros((n_samples, len(patch_sizes), data.shape[1], data.shape[2]))

# Loop over samples:
for n in range(n_samples):
    
    # Loop over patch sizes:
    for p in range(len(patch_sizes)):
        
        # Get current patch size:
        patch_size = patch_sizes[p]
        
        # Compute and store relevance map for current sample and chosen patch size:
        rel_loss_reduction_maps[n,p,:,:] = compute_single_relevance_map(input_sample=val_input[n:n+1], patch_size=patch_size, model=model)    

# Save validation loss:
np.save(path_to_model / "rel_loss_reduction_maps.npy", rel_loss_reduction_maps)
