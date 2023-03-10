# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.
# Work with sea surface temperature (sst) fields from Earth System Models, either FOCI or CESM.
#
# Apply optimal mask of missing values, which is identical to all samples (mask_type='fixed').
# Got optimal mask from rel. loss reduction map, obtained on validation samples with range model.
# And only use each sample once, no data augmentation in this experiment.

import os
import sys

from json import dump, load
from pathlib import Path

import xarray as xr
import numpy as np

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
from models import build_unet_5conv



# ## Set parameters up-front:

# Set paths to optimal missing masks as strings:
paths_to_missing_masks_string = [
    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_CESM_variable_range_0_100_factor_3_final/relevance_1',
    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_CESM_variable_range_0_100_factor_3_final/relevance_1',
    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_CESM_variable_range_0_100_factor_3_final/relevance_1',
]

#paths_to_missing_masks_string = [
#    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_FOCI_variable_range_25_999_factor_3_final/relevance_2',
#    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_FOCI_variable_range_25_999_factor_3_final/relevance_2',
#    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_FOCI_variable_range_25_999_factor_3_final/relevance_2',
#]

## Create paths to optimal missing masks as PosixPaths:
paths_to_missing_masks = []
for temp_path in paths_to_missing_masks_string:
    paths_to_missing_masks.append(Path(temp_path))

## Decide to work on test data or full data:
# path_to_data = 'GitHub/MarcoLandtHayen/reconstruct_missing_data/data/test_data/' # Test data
path_to_data = "GitHub/MarcoLandtHayen/reconstruct_missing_data/data/raw/sst.mnmean.nc"  # Full data

# Model configuration, to store results:
model_config = "unet_5conv"

# Data loading and preprocessing:
feature = "sea-surface-temperature"  # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = "sst"  # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = "realworld"  # Choose Earth System Model, either 'FOCI' or 'CESM'.
seed = 4  # Seed for random number generator, for reproducibility of missing value mask.
run = "_seed_4" # Specify run number (or '_final'). Don't need seed, since we use optimal fixed mask.
mask_source = paths_to_missing_masks_string  # Paths to experiments, that produced optimal sampling masks, as strings.
mask_type = "optimal_from_CESM"  # Can have random missing values, individually for each data sample ('variable'),
# or randomly create only a single mask, that is then applied to all samples identically ('fixed'),
# or use fixed mask with optimal grip points, leading to strongest reduction in rel. loss ('optimal').
missing_type = "discrete"  # Either specify discrete amounts of missing values ('discrete') or give a range ('range').
augmentation_factor = (
    1  # Number of times, each sample is to be cloned, keeping the original order.
)
train_val_split = 0.8  # Set rel. amount of samples used for training.
missing_values = [
    0.999,
    0.99,
    0.95,
]  # Set array for desired amounts of missing values: 0.9 means, that 90% of the values are missing.
# Or set a range by only giving minimum and maximum allowed relative amounts of missing values,
# e.g. [0.75, 0.95], according to missing_type 'discrete' or 'range', respectively.
scale_to = "zero_one"  # Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.

# To build, compile and train model:
CNN_filters = [64, 128, 256, 512, 1024]  # [2,4,8,16] # Number of filters.
CNN_kernel_size = 4  # Kernel size
learning_rate = 0.00005
loss_function = "mse"
epochs = 30
batch_size = 10


# Create directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
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
os.makedirs(path, exist_ok=False)

# Store parameters as json:
parameters = {
    "model_config": model_config,
    "feature": feature,
    "feature_short": feature_short,
    "source": source,
    "run": run,
    "mask_source": mask_source,
    "mask_type": mask_type,
    "missing_type": missing_type,
    "augmentation_factor": augmentation_factor,
    "train_val_split": train_val_split,
    "missing_values": missing_values,
    "scale_to": scale_to,
    "CNN_filters": CNN_filters,
    "CNN_kernel_size": CNN_kernel_size,
    "learning_rate": learning_rate,
    "loss_function": loss_function,
    "epochs": epochs,
    "batch_size": batch_size,
}

with open(path / "parameters.json", "w") as f:
    dump(parameters, f)


# # Load data:

# Open data set:
sst_dataset=xr.open_dataset("GitHub/MarcoLandtHayen/reconstruct_missing_data/data/raw/sst.mnmean.nc")

# Start with raw slp fields as lat/lon grids in time, from 1948 to 2022:
sst_fields = (
    sst_dataset.sst
    .sel(time=slice('1880-01-01', '2022-12-01'))
)

# Compute monthly climatology (here 1980 - 2009) for whole world:
sst_climatology_fields = (
    sst_dataset.sst
    .sel(time=slice('1980-01-01','2009-12-01'))
    .groupby("time.month")
    .mean("time")
)

# Get slp anomaly fields by subtracting monthly climatology from raw slp fields:
sst_anomaly_fields = sst_fields.groupby("time.month") - sst_climatology_fields

# Remove last row (latidute) and last 4 columns (longitude), to have even number of steps in latitude (=88)
# and longitude (=176), that can be evenly divided 4 times by two. This serves as 'quick-and-dirty'
# solution to avoid problems with UPSAMPLING in U-Net. There must be a more elegant way, take care of it later!
sst_anomaly_fields = sst_anomaly_fields.values[:,5:-4,:-4]

# # Train models:

# Loop over array of desired amounts of missing values:
for i in range(len(missing_values)):

    # Get current relative amount of missing values:
    missing = missing_values[i]

    # Create sub-directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
    
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        os.makedirs(path / "missing_" f"{int(missing*1000)}", exist_ok=False)
    else:
        os.makedirs(path / "missing_" f"{int(missing*100)}", exist_ok=False)        

    # Extend data, if desired:
    data = clone_data(data=sst_anomaly_fields, augmentation_factor=augmentation_factor)

    # Reload optimal mask for missing values.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        filename_missing_mask = "optimal_sampling_mask_"+str(int(missing*1000))+"_realworld.npy"
        missing_mask = np.load(
            paths_to_missing_masks[i] / filename_missing_mask
        )
    else:
        filename_missing_mask = "optimal_sampling_mask_"+str(int(missing*100))+"_realworld.npy"
        missing_mask = np.load(
            paths_to_missing_masks[i] / filename_missing_mask
        )
    
    # Expand missing mask to have sample dimension as first dimension, then repeat. Dimensions: (#samples, lat, lon).
    missing_mask = np.repeat(np.expand_dims(missing_mask,axis=0),data.shape[0], axis=0)
    
    # Store missing mask.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        np.save(path / "missing_" f"{int(missing*1000)}" / "missing_mask.npy", missing_mask)
    else:
        np.save(path / "missing_" f"{int(missing*100)}" / "missing_mask.npy", missing_mask)

    # Use sparse data as inputs and complete data as targets. Split sparse and complete data into training and validation sets.
    # Scale or normlalize data according to statistics obtained from only training data.
    (
        train_input,
        val_input,
        train_target,
        val_target,
        train_min,
        train_max,
        train_mean,
        train_std,
    ) = split_and_scale_data(data, missing_mask, train_val_split, scale_to)

    # Build and compile U-Net model:
    model = build_unet_5conv(
        input_shape=(train_input.shape[1], train_input.shape[2], 1),
        CNN_filters=CNN_filters,
        CNN_kernel_size=CNN_kernel_size,
        learning_rate=learning_rate,
        loss_function=loss_function,
    )

    # Save untrained model.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        model.save(path / "missing_" f"{int(missing*1000)}" / f"epoch_{0}")
    else:
        model.save(path / "missing_" f"{int(missing*100)}" / f"epoch_{0}")

    # Initialize storage for training and validation loss:
    train_loss = []
    val_loss = []

    # Get model predictions on train and validation data FROM UNTRAINED MODEL!
    train_pred = model.predict(train_input)
    val_pred = model.predict(val_input)

    # Store loss on training and validation data:
    train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
    val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))

    # Loop over number of training epochs:
    for j in range(epochs):

        # Print status:
        print("  Epoch: ", j + 1, " of ", epochs)

        # Train model on sparse inputs with complete 2D fields as targets, for SINGLE epoch:
        history = model.fit(
            train_input,
            train_target,
            epochs=1,
            verbose=0,
            shuffle=True,
            batch_size=batch_size,
            validation_data=(val_input, val_target),
        )

        # Save trained model after current epoch.
        # Rel. amount of missing values = 0.999 requires special treatment:
        if missing==0.999:
            model.save(path / "missing_" f"{int(missing*1000)}" / f"epoch_{j+1}")
        else:
            model.save(path / "missing_" f"{int(missing*100)}" / f"epoch_{j+1}")

        # Get model predictions on train and validation data AFTER current epoch:
        train_pred = model.predict(train_input)
        val_pred = model.predict(val_input)

        # Store loss on training and validation data:
        train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
        val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))

    # Save loss.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        np.save(path / "missing_" f"{int(missing*1000)}" / "train_loss.npy", train_loss)
        np.save(path / "missing_" f"{int(missing*1000)}" / "val_loss.npy", val_loss)
    else:
        np.save(path / "missing_" f"{int(missing*100)}" / "train_loss.npy", train_loss)
        np.save(path / "missing_" f"{int(missing*100)}" / "val_loss.npy", val_loss)

    

# ## Set parameters up-front:

# Set paths to optimal missing masks as strings:
paths_to_missing_masks_string = [
    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_CESM_variable_range_0_100_factor_3_final/relevance_1',
    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_CESM_variable_range_0_100_factor_3_final/relevance_1',
    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_CESM_variable_range_0_100_factor_3_final/relevance_1',
]

#paths_to_missing_masks_string = [
#    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_FOCI_variable_range_25_999_factor_3_final/relevance_2',
#    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_FOCI_variable_range_25_999_factor_3_final/relevance_2',
#    'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_sst_FOCI_variable_range_25_999_factor_3_final/relevance_2',
#]

## Create paths to optimal missing masks as PosixPaths:
paths_to_missing_masks = []
for temp_path in paths_to_missing_masks_string:
    paths_to_missing_masks.append(Path(temp_path))

## Decide to work on test data or full data:
# path_to_data = 'GitHub/MarcoLandtHayen/reconstruct_missing_data/data/test_data/' # Test data
path_to_data = "GitHub/MarcoLandtHayen/reconstruct_missing_data/data/raw/sst.mnmean.nc"  # Full data

# Model configuration, to store results:
model_config = "unet_5conv"

# Data loading and preprocessing:
feature = "sea-surface-temperature"  # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = "sst"  # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = "realworld"  # Choose Earth System Model, either 'FOCI' or 'CESM'.
seed = 5  # Seed for random number generator, for reproducibility of missing value mask.
run = "_seed_5" # Specify run number (or '_final'). Don't need seed, since we use optimal fixed mask.
mask_source = paths_to_missing_masks_string  # Paths to experiments, that produced optimal sampling masks, as strings.
mask_type = "optimal_from_CESM"  # Can have random missing values, individually for each data sample ('variable'),
# or randomly create only a single mask, that is then applied to all samples identically ('fixed'),
# or use fixed mask with optimal grip points, leading to strongest reduction in rel. loss ('optimal').
missing_type = "discrete"  # Either specify discrete amounts of missing values ('discrete') or give a range ('range').
augmentation_factor = (
    1  # Number of times, each sample is to be cloned, keeping the original order.
)
train_val_split = 0.8  # Set rel. amount of samples used for training.
missing_values = [
    0.999,
    0.99,
    0.95,
]  # Set array for desired amounts of missing values: 0.9 means, that 90% of the values are missing.
# Or set a range by only giving minimum and maximum allowed relative amounts of missing values,
# e.g. [0.75, 0.95], according to missing_type 'discrete' or 'range', respectively.
scale_to = "zero_one"  # Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.

# To build, compile and train model:
CNN_filters = [64, 128, 256, 512, 1024]  # [2,4,8,16] # Number of filters.
CNN_kernel_size = 4  # Kernel size
learning_rate = 0.00005
loss_function = "mse"
epochs = 30
batch_size = 10


# Create directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
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
os.makedirs(path, exist_ok=False)

# Store parameters as json:
parameters = {
    "model_config": model_config,
    "feature": feature,
    "feature_short": feature_short,
    "source": source,
    "run": run,
    "mask_source": mask_source,
    "mask_type": mask_type,
    "missing_type": missing_type,
    "augmentation_factor": augmentation_factor,
    "train_val_split": train_val_split,
    "missing_values": missing_values,
    "scale_to": scale_to,
    "CNN_filters": CNN_filters,
    "CNN_kernel_size": CNN_kernel_size,
    "learning_rate": learning_rate,
    "loss_function": loss_function,
    "epochs": epochs,
    "batch_size": batch_size,
}

with open(path / "parameters.json", "w") as f:
    dump(parameters, f)


# # Load data:

# Open data set:
sst_dataset=xr.open_dataset("GitHub/MarcoLandtHayen/reconstruct_missing_data/data/raw/sst.mnmean.nc")

# Start with raw slp fields as lat/lon grids in time, from 1948 to 2022:
sst_fields = (
    sst_dataset.sst
    .sel(time=slice('1880-01-01', '2022-12-01'))
)

# Compute monthly climatology (here 1980 - 2009) for whole world:
sst_climatology_fields = (
    sst_dataset.sst
    .sel(time=slice('1980-01-01','2009-12-01'))
    .groupby("time.month")
    .mean("time")
)

# Get slp anomaly fields by subtracting monthly climatology from raw slp fields:
sst_anomaly_fields = sst_fields.groupby("time.month") - sst_climatology_fields

# Remove last row (latidute) and last 4 columns (longitude), to have even number of steps in latitude (=88)
# and longitude (=176), that can be evenly divided 4 times by two. This serves as 'quick-and-dirty'
# solution to avoid problems with UPSAMPLING in U-Net. There must be a more elegant way, take care of it later!
sst_anomaly_fields = sst_anomaly_fields.values[:,5:-4,:-4]

# # Train models:

# Loop over array of desired amounts of missing values:
for i in range(len(missing_values)):

    # Get current relative amount of missing values:
    missing = missing_values[i]

    # Create sub-directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
    
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        os.makedirs(path / "missing_" f"{int(missing*1000)}", exist_ok=False)
    else:
        os.makedirs(path / "missing_" f"{int(missing*100)}", exist_ok=False)        

    # Extend data, if desired:
    data = clone_data(data=sst_anomaly_fields, augmentation_factor=augmentation_factor)

    # Reload optimal mask for missing values.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        filename_missing_mask = "optimal_sampling_mask_"+str(int(missing*1000))+"_realworld.npy"
        missing_mask = np.load(
            paths_to_missing_masks[i] / filename_missing_mask
        )
    else:
        filename_missing_mask = "optimal_sampling_mask_"+str(int(missing*100))+"_realworld.npy"
        missing_mask = np.load(
            paths_to_missing_masks[i] / filename_missing_mask
        )
    
    # Expand missing mask to have sample dimension as first dimension, then repeat. Dimensions: (#samples, lat, lon).
    missing_mask = np.repeat(np.expand_dims(missing_mask,axis=0),data.shape[0], axis=0)
    
    # Store missing mask.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        np.save(path / "missing_" f"{int(missing*1000)}" / "missing_mask.npy", missing_mask)
    else:
        np.save(path / "missing_" f"{int(missing*100)}" / "missing_mask.npy", missing_mask)

    # Use sparse data as inputs and complete data as targets. Split sparse and complete data into training and validation sets.
    # Scale or normlalize data according to statistics obtained from only training data.
    (
        train_input,
        val_input,
        train_target,
        val_target,
        train_min,
        train_max,
        train_mean,
        train_std,
    ) = split_and_scale_data(data, missing_mask, train_val_split, scale_to)

    # Build and compile U-Net model:
    model = build_unet_5conv(
        input_shape=(train_input.shape[1], train_input.shape[2], 1),
        CNN_filters=CNN_filters,
        CNN_kernel_size=CNN_kernel_size,
        learning_rate=learning_rate,
        loss_function=loss_function,
    )

    # Save untrained model.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        model.save(path / "missing_" f"{int(missing*1000)}" / f"epoch_{0}")
    else:
        model.save(path / "missing_" f"{int(missing*100)}" / f"epoch_{0}")

    # Initialize storage for training and validation loss:
    train_loss = []
    val_loss = []

    # Get model predictions on train and validation data FROM UNTRAINED MODEL!
    train_pred = model.predict(train_input)
    val_pred = model.predict(val_input)

    # Store loss on training and validation data:
    train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
    val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))

    # Loop over number of training epochs:
    for j in range(epochs):

        # Print status:
        print("  Epoch: ", j + 1, " of ", epochs)

        # Train model on sparse inputs with complete 2D fields as targets, for SINGLE epoch:
        history = model.fit(
            train_input,
            train_target,
            epochs=1,
            verbose=0,
            shuffle=True,
            batch_size=batch_size,
            validation_data=(val_input, val_target),
        )

        # Save trained model after current epoch.
        # Rel. amount of missing values = 0.999 requires special treatment:
        if missing==0.999:
            model.save(path / "missing_" f"{int(missing*1000)}" / f"epoch_{j+1}")
        else:
            model.save(path / "missing_" f"{int(missing*100)}" / f"epoch_{j+1}")

        # Get model predictions on train and validation data AFTER current epoch:
        train_pred = model.predict(train_input)
        val_pred = model.predict(val_input)

        # Store loss on training and validation data:
        train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
        val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))

    # Save loss.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        np.save(path / "missing_" f"{int(missing*1000)}" / "train_loss.npy", train_loss)
        np.save(path / "missing_" f"{int(missing*1000)}" / "val_loss.npy", val_loss)
    else:
        np.save(path / "missing_" f"{int(missing*100)}" / "train_loss.npy", train_loss)
        np.save(path / "missing_" f"{int(missing*100)}" / "val_loss.npy", val_loss)

