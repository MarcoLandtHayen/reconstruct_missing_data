# Description:
#
# Go beyond [Xiantao et al., 2020] approach: Test U-Net to predict future complete data from sparse inputs.

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
    prepare_univariate_data, 
    prepare_multivariate_data, 
    prepare_timelagged_data,
)
from models import build_unet_4conv



# ## Set parameters up-front:

## Decide to work on test data or full data:
# path_to_data = 'GitHub/MarcoLandtHayen/reconstruct_missing_data/data/test_data/' # Test data
path_to_data = "climate_index_collection/data/raw/2022-08-22/"  # Full data

# Model configuration, to store results:
model_config = "predict_unet_4conv"

# Data loading and preprocessing:
data_source_name=sys.argv[1]
input_features=[
    'sea-level-pressure', 
    'sea-surface-temperature', 
    'geopotential-height', 
    'surface-air-temperature', 
    'sea-surface-salinity', 
    'precipitation'
]
input_features_short=[
    'slp', 
    'sst', 
    'z500', 
    'sat', 
    'sss', 
    'prec'
]
target_feature='precipitation'
feature_short = 'prec'
load_samples_from=0
load_samples_to=-1
mask_type='variable'
missing_type='discrete'
missing_values = [
    float(sys.argv[4]),
]
seed=int(sys.argv[3])
train_val_split=0.8
val_test_split=0.5
scale_to='zero_one'
shift=int(sys.argv[2])

# To build, compile and train model:
CNN_filters = [64, 128, 256, 512]  # [2,4,8,16] # Number of filters.
CNN_kernel_size = 5  # Kernel size
learning_rate = 0.00005
loss_function = "mse"
epochs = 100
batch_size = 10


# Create directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
path = Path(
    "GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/"
    + model_config
    + "_multivariate_"
    + feature_short
    + "_"
    + data_source_name
    + "_"
    + mask_type
    + "_"
    + missing_type
    + "_lead_"
    + str(shift)
    + "_seed_"
    + str(seed)
)
if missing_values[0] == 0.9:
    os.makedirs(path, exist_ok=False)

# Store parameters as json:
parameters = {
    "model_config": model_config,
    "data_source_name": data_source_name,
    "inpute_features": input_features,
    "target_feature": target_feature,
    "feature_short": feature_short,
    "load_sample_from": load_samples_from, 
    "load_samples_to": load_samples_to,
    "mask_type": mask_type,
    "missing_type": missing_type,
    "missing_values": [0.9,0.75,0.5],        
    "seed": seed,
    "train_val_split": train_val_split,
    "val_test_split": val_test_split,
    "scale_to": scale_to,
    "shift": shift,
    "CNN_filters": CNN_filters,
    "CNN_kernel_size": CNN_kernel_size,
    "learning_rate": learning_rate,
    "loss_function": loss_function,
    "epochs": epochs,
    "batch_size": batch_size,
}

if missing_values[0] == 0.9:
    with open(path / "parameters.json", "w") as f:
        dump(parameters, f)

# # Train models:

# Loop over array of desired amounts of missing values:
for i in range(len(missing_values)):

    # Get current relative amount of missing values:
    missing = missing_values[i]

    # Create sub-directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        os.makedirs(path / "missing_" f"{int(missing*1000)}", exist_ok=True)
    else:
        os.makedirs(path / "missing_" f"{int(missing*100)}", exist_ok=True)        

    # Create paths to optimal masks.    
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        
        # Initialize storage:
        path_to_optimal_masks=[]
        
        # Loop over input features:
        for f in range(len(input_features_short)):
        
            # Create path for current feature:
            temp_path=(
                'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_'
                +input_features_short[f]+'_'+data_source_name
                +'_variable_range_0_100_factor_3_final/relevance_1/'
                +'optimal_sampling_mask_'+str(int(missing*1000))+'.npy'
        )
            # Append path for current feature:
            path_to_optimal_masks.append(temp_path)
        
    else:
        
        # Initialize storage:
        path_to_optimal_masks=[]
        
        # Loop over input features:
        for f in range(len(input_features_short)):
        
            # Create path for current feature:
            temp_path=(
                'GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_'
                +input_features_short[f]+'_'+data_source_name
                +'_variable_range_0_100_factor_3_final/relevance_1/'
                +'optimal_sampling_mask_'+str(int(missing*100))+'.npy'
        )
            # Append path for current feature:
            path_to_optimal_masks.append(temp_path)        
        
        
    # Prepare data:
    (
        train_input,
        val_input,
        test_input,
        train_target,
        val_target,
        test_target,
        train_min,
        train_max,
        train_mean,
        train_std,
    ) = prepare_multivariate_data(
            data_path=path_to_data, 
            data_source_name=data_source_name,
            input_features=input_features, 
            target_feature=target_feature,
            load_samples_from=load_samples_from, 
            load_samples_to=load_samples_to,
            mask_type=mask_type, 
            missing_type=missing_type, 
            missing_min=missing, 
            missing_max=missing, 
            seed=seed, 
            path_to_optimal_masks=path_to_optimal_masks,
            train_val_split=train_val_split,
            val_test_split=val_test_split,
            scale_to=scale_to,
            shift=shift,
    )    
        
    # Build and compile U-Net model:
    model = build_unet_4conv(
        input_shape=(train_input.shape[1], train_input.shape[2], train_input.shape[3]),
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

    # Initialize storage for training, validation and test loss:
    train_loss = []
    val_loss = []
    test_loss = []

    # Get model predictions on train, validation and test data FROM UNTRAINED MODEL!
    train_pred = model.predict(train_input)
    val_pred = model.predict(val_input)
    test_pred = model.predict(test_input)

    # Store loss on training, validation and test data:
    train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
    val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))
    test_loss.append(np.mean((test_pred[:, :, :, 0] - test_target) ** 2))

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
#            validation_data=(val_input, val_target),
        )

        # Save trained model after current epoch.
        # Rel. amount of missing values = 0.999 requires special treatment:
        if missing==0.999:
            model.save(path / "missing_" f"{int(missing*1000)}" / f"epoch_{j+1}")
        else:
            model.save(path / "missing_" f"{int(missing*100)}" / f"epoch_{j+1}")

        # Get model predictions on train, validation and test data AFTER current epoch:
        train_pred = model.predict(train_input)
        val_pred = model.predict(val_input)
        test_pred = model.predict(test_input)

        # Store loss on training and validation data:
        train_loss.append(np.mean((train_pred[:, :, :, 0] - train_target) ** 2))
        val_loss.append(np.mean((val_pred[:, :, :, 0] - val_target) ** 2))
        test_loss.append(np.mean((test_pred[:, :, :, 0] - test_target) ** 2))

    # Save loss.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        np.save(path / "missing_" f"{int(missing*1000)}" / "train_loss.npy", train_loss)
        np.save(path / "missing_" f"{int(missing*1000)}" / "val_loss.npy", val_loss)
        np.save(path / "missing_" f"{int(missing*1000)}" / "test_loss.npy", test_loss)
    else:
        np.save(path / "missing_" f"{int(missing*100)}" / "train_loss.npy", train_loss)
        np.save(path / "missing_" f"{int(missing*100)}" / "val_loss.npy", val_loss)
        np.save(path / "missing_" f"{int(missing*100)}" / "test_loss.npy", test_loss)

    

    
