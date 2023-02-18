# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.
# Opposed to their net, only have 4 instead of 5 convolutional layers.
# Work with sea level pressure (slp) fields from Earth System Models, either FOCI or CESM.
#
# Have random mask for missing values, individually for each data sample (mask_type='variable').
# And optionally use each sample multiple times (data augmentation) in this experiment.

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


# ## Set parameters up-front:

# Specify experiment:
model_config = "unet_4conv"
feature = "sea-surface-temperature"  # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = "sst"  # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = "CESM"  # Choose Earth System Model, either 'FOCI' or 'CESM'.

mask_type = "variable"
missing_type = "discrete"
augmentation_factor = 2
run = "_final"

# Get path to model and data:
path = Path(
    "GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/"
    + model_config
    + "_"
    + feature_short
    + "_"
    + source
    + "_"
    + mask_type
    + "_"
    + missing_type
    + "_factor_"
    + str(augmentation_factor)
    + run
)

# Reload parameters for this experiment:
with open(path / "parameters.json", "r") as f:
    parameters = load(f)

train_val_split = parameters["train_val_split"]
missing_values = parameters["missing_values"]
scale_to = parameters["scale_to"]
epochs = parameters["epochs"]

# Path to full data:
path_to_data = "climate_index_collection/data/raw/2022-08-22/"

# Load data, only to infer sample dimensions (lat,lon):
data = load_data_set(data_path=path_to_data, data_source_name=source)

# Get number of train and validation samples: Consider augmentation factor!
n_train = int(len(data[feature]) * augmentation_factor * train_val_split)
n_val = (len(data[feature]) * augmentation_factor) - n_train

# # Sensitivity experiment for final models, each pretrained with fixed rel. amount of missing values.
# # Compute minimum loss (mean squared errof) of predictions compared to targets.

# Initialize storage for validation inputs: Dimensions (#missing values settings, #val.samples, lat, lon)
val_input_all = np.zeros(
    (len(missing_values), n_val, data[feature].shape[1], data[feature].shape[2])
)

# Initialize storage for validation targets: Dimensions (#val.samples, lat, lon)
val_target_all = np.zeros((n_val, data[feature].shape[1], data[feature].shape[2]))

# Initialize storage for validation predictions: Dimensions (#missing values settings, #missing values settings, #samples, lat, lon)
val_pred_all = np.zeros(
    (
        len(missing_values),
        len(missing_values),
        n_val,
        data[feature].shape[1],
        data[feature].shape[2],
    )
)


# Loop over rel. amounts of missing values:
for i in range(len(missing_values)):

    # Get current rel. amount of missing values, as fixed amount:
    missing_fix = missing_values[i]

    # Reload final pre-trained model for current fixed rel. amount of missing values.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing_fix==0.999:
        model = tf.keras.models.load_model(
            path / "missing_" f"{int(missing_fix*1000)}" / "model"
        )
    else:
        model = tf.keras.models.load_model(
            path / "missing_" f"{int(missing_fix*100)}" / "model"
        )

    # Loop over rel. amounts of missing values:
    for j in range(len(missing_values)):

        # Get current rel. amount of missing values:
        missing = missing_values[j]

        ## Load complete data, reconstruct sparse data (as inputs) and complete data (as targets).
        ## Only once, for first setup of fixed rel. amount of missing values:

        if i == 0:

            # Load data:
            data = load_data_set(data_path=path_to_data, data_source_name=source)

            # Select single feature and compute anomalies, using whole time span as climatology:
            data = get_anomalies(feature=feature, data_set=data)

            # Extend data, if desired:
            data = clone_data(data=data, augmentation_factor=augmentation_factor)

            # Reload mask for missing values.
            # Rel. amount of missing values = 0.999 requires special treatment:
            if missing==0.999:
                missing_mask = np.load(
                    path / "missing_" f"{int(missing*1000)}" / "missing_mask.npy"
                )
            else:
                missing_mask = np.load(
                    path / "missing_" f"{int(missing*100)}" / "missing_mask.npy"
                )

            # Use sparse data as inputs and complete data as targets. Split sparse and complete data into training and validation sets.
            # Scale or normlalize data according to statistics obtained from only training data.
            (
                _,
                val_input,
                _,
                val_target,
                train_min,
                train_max,
                train_mean,
                train_std,
            ) = split_and_scale_data(data, missing_mask, train_val_split, scale_to)

            # Store inputs: Omit final dimension of predictions, that only contains the channel (here: 1)
            val_input_all[j, :, :, :] = val_input[:, :, :, 0]

            # Store targets: Only once!
            if j == 0:
                val_target_all[:, :, :] = val_target[:, :, :]

        # Get model predictions on validation data:
        val_pred = model.predict(val_input_all[j])

        # Store predictions: Omit final dimension of predictions, that only contains the channel (here: 1)
        val_pred_all[i, j, :, :, :] = val_pred[:, :, :, 0]

# # Compute validation loss for sensitivity experiment:

# Initialize storage for validation loss for all fixed rel. amounts of missing values.
# Dimensions (#missing values settings, #missing values settings):
val_loss_all = np.zeros((len(missing_values), len(missing_values)))

# Loop over fixed rel. amount of missing values:
for i in range(len(missing_values)):

    # Loop over rel. amounts of missing values:
    for j in range(len(missing_values)):
        # Store validation loss:
        val_loss_all[i, j] = np.mean(
            (val_target_all - val_pred_all[i, j, :, :, :]) ** 2
        )

# Save validation loss:
np.save(path / "val_loss_all.npy", val_loss_all)
