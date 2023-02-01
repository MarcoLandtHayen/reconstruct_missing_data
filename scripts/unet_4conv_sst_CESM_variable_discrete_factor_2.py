# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.
# Opposed to their net, only have 4 instead of 5 convolutional layers.
# Work with sea surface temperature (sst) fields from Earth System Models, either FOCI or CESM.
#
# Have random mask for missing values, individually for each data sample (mask_type='variable').
# And only use each sample once, no data augmentation in this experiment.

import os
import sys

from json import dump, load
from pathlib import Path

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
from models import build_unet_4conv



# ## Set parameters up-front:

## Decide to work on test data or full data:
# path_to_data = 'GitHub/MarcoLandtHayen/reconstruct_missing_data/data/test_data/' # Test data
path_to_data = "climate_index_collection/data/raw/2022-08-22/"  # Full data

# Model configuration, to store results:
model_config = "unet_4conv"

# Data loading and preprocessing:
feature = "sea-surface-temperature"  # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = "sst"  # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = "CESM"  # Choose Earth System Model, either 'FOCI' or 'CESM'.
seed = 2  # Seed for random number generator, for reproducibility of missing value mask.
mask_type = "variable"  # Can have random missing values, individually for each data sample ('variable'),
# or randomly create only a single mask, that is then applied to all samples identically ('fixed').
missing_type = "discrete"  # Either specify discrete amounts of missing values ('discrete') or give a range ('range').
augmentation_factor = (
    2  # Number of times, each sample is to be cloned, keeping the original order.
)
train_val_split = 0.8  # Set rel. amount of samples used for training.
missing_values = [
    0.99,
    0.95,
    0.9,
    0.75,
    0.5,
    0.25,
]  # Set array for desired amounts of missing values: 0.9 means, that 90% of the values are missing.
# Or set a range by only giving minimum and maximum allowed relative amounts of missing values,
# e.g. [0.75, 0.95], according to missing_type 'discrete' or 'range', respectively.
scale_to = "zero_one"  # Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.

# To build, compile and train model:
CNN_filters = [64, 128, 256, 512]  # [2,4,8,16] # Number of filters.
CNN_kernel_size = 5  # Kernel size
learning_rate = 0.0001
loss_function = "mse"
epochs = 10
batch_size = 10


# Create directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
path = Path(
    "GitGeomar/marco-landt-hayen/reconstruct_missing_data/results/"
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
    + "_seed_"
    + str(seed)
)
os.makedirs(path, exist_ok=False)

# Store parameters as json:
parameters = {
    "model_config": model_config,
    "feature": feature,
    "feature_short": feature_short,
    "source": source,
    "seed": seed,
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


# # Train models:

# Loop over array of desired amounts of missing values:
for i in range(len(missing_values)):

    # Get current relative amount of missing values:
    missing = missing_values[i]

    # Print status:
    print("Missing values: ", i + 1, " of ", len(missing_values))

    # Create sub-directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
    os.makedirs(path / "missing_" f"{int(missing*100)}", exist_ok=False)

    # Load data, including ALL fields and mask for Ocean values:
    data = load_data_set(data_path=path_to_data, data_source_name=source)

    # Select single feature and compute anomalies, using whole time span as climatology:
    data = get_anomalies(feature=feature, data_set=data)

    # Extend data, if desired:
    data = clone_data(data=data, augmentation_factor=augmentation_factor)

    # Create mask for missing values:
    missing_mask = create_missing_mask(
        data=data,
        mask_type=mask_type,
        missing_type=missing_type,
        missing_min=missing,
        missing_max=missing,
        seed=seed,
    )

    # Store missing mask:
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
    model = build_unet_4conv(
        input_shape=(train_input.shape[1], train_input.shape[2], 1),
        CNN_filters=CNN_filters,
        CNN_kernel_size=CNN_kernel_size,
        learning_rate=learning_rate,
        loss_function=loss_function,
    )

    # Save untrained model:
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

        # Save trained model after current epoch:
        model.save(path / "missing_" f"{int(missing*100)}" / f"epoch_{j+1}")

        # Get model predictions on train and validation data AFTER current epoch:
        train_pred = model.predict(train_input)
        val_pred = model.predict(val_input)

        # Store loss on training and validation data:
        train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
        val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))

    # Save loss:
    np.save(path / "missing_" f"{int(missing*100)}" / "train_loss.npy", train_loss)
    np.save(path / "missing_" f"{int(missing*100)}" / "val_loss.npy", val_loss)
    
    
    
    
    
# ## Set parameters up-front:

## Decide to work on test data or full data:
# path_to_data = 'GitHub/MarcoLandtHayen/reconstruct_missing_data/data/test_data/' # Test data
path_to_data = "climate_index_collection/data/raw/2022-08-22/"  # Full data

# Model configuration, to store results:
model_config = "unet_4conv"

# Data loading and preprocessing:
feature = "sea-surface-temperature"  # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = "sst"  # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = "CESM"  # Choose Earth System Model, either 'FOCI' or 'CESM'.
seed = 3  # Seed for random number generator, for reproducibility of missing value mask.
mask_type = "variable"  # Can have random missing values, individually for each data sample ('variable'),
# or randomly create only a single mask, that is then applied to all samples identically ('fixed').
missing_type = "discrete"  # Either specify discrete amounts of missing values ('discrete') or give a range ('range').
augmentation_factor = (
    2  # Number of times, each sample is to be cloned, keeping the original order.
)
train_val_split = 0.8  # Set rel. amount of samples used for training.
missing_values = [
    0.99,
    0.95,
    0.9,
    0.75,
    0.5,
    0.25,
]  # Set array for desired amounts of missing values: 0.9 means, that 90% of the values are missing.
# Or set a range by only giving minimum and maximum allowed relative amounts of missing values,
# e.g. [0.75, 0.95], according to missing_type 'discrete' or 'range', respectively.
scale_to = "zero_one"  # Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.

# To build, compile and train model:
CNN_filters = [64, 128, 256, 512]  # [2,4,8,16] # Number of filters.
CNN_kernel_size = 5  # Kernel size
learning_rate = 0.0001
loss_function = "mse"
epochs = 10
batch_size = 10


# Create directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
path = Path(
    "GitGeomar/marco-landt-hayen/reconstruct_missing_data/results/"
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
    + "_seed_"
    + str(seed)
)
os.makedirs(path, exist_ok=False)

# Store parameters as json:
parameters = {
    "model_config": model_config,
    "feature": feature,
    "feature_short": feature_short,
    "source": source,
    "seed": seed,
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


# # Train models:

# Loop over array of desired amounts of missing values:
for i in range(len(missing_values)):

    # Get current relative amount of missing values:
    missing = missing_values[i]

    # Print status:
    print("Missing values: ", i + 1, " of ", len(missing_values))

    # Create sub-directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
    os.makedirs(path / "missing_" f"{int(missing*100)}", exist_ok=False)

    # Load data, including ALL fields and mask for Ocean values:
    data = load_data_set(data_path=path_to_data, data_source_name=source)

    # Select single feature and compute anomalies, using whole time span as climatology:
    data = get_anomalies(feature=feature, data_set=data)

    # Extend data, if desired:
    data = clone_data(data=data, augmentation_factor=augmentation_factor)

    # Create mask for missing values:
    missing_mask = create_missing_mask(
        data=data,
        mask_type=mask_type,
        missing_type=missing_type,
        missing_min=missing,
        missing_max=missing,
        seed=seed,
    )

    # Store missing mask:
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
    model = build_unet_4conv(
        input_shape=(train_input.shape[1], train_input.shape[2], 1),
        CNN_filters=CNN_filters,
        CNN_kernel_size=CNN_kernel_size,
        learning_rate=learning_rate,
        loss_function=loss_function,
    )

    # Save untrained model:
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

        # Save trained model after current epoch:
        model.save(path / "missing_" f"{int(missing*100)}" / f"epoch_{j+1}")

        # Get model predictions on train and validation data AFTER current epoch:
        train_pred = model.predict(train_input)
        val_pred = model.predict(val_input)

        # Store loss on training and validation data:
        train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
        val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))

    # Save loss:
    np.save(path / "missing_" f"{int(missing*100)}" / "train_loss.npy", train_loss)
    np.save(path / "missing_" f"{int(missing*100)}" / "val_loss.npy", val_loss)