# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.

import os
import sys
sys.path.append(
    "GitHub/MarcoLandtHayen/reconstruct_missing_data/reconstruct_missing_data"
)

from pathlib import Path
from json import dump, load

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

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
from indices import (
    el_nino_southern_oscillation_34,
    atlantic_multidecadal_oscillation,
)

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

# Set path to final model:

# sst FOCI:
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/predict_unet_4conv_multivariate_sst_FOCI_optimal_discrete_lead_0_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/predict_unet_4conv_multivariate_sst_FOCI_optimal_discrete_lead_1_final'
path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/predict_unet_4conv_multivariate_sst_FOCI_optimal_discrete_lead_3_final'


# Reload parameters for this experiment:
with open(Path(path_to_final_model) / 'parameters.json', 'r') as f:
    parameters=load(f)

data_source_name = parameters['data_source_name']
input_features = parameters['inpute_features']
target_feature = parameters['target_feature']
feature_short = parameters['feature_short']
load_samples_from = parameters['load_sample_from']
load_samples_to = parameters['load_samples_to']
mask_type = parameters['mask_type']
missing_type = parameters['missing_type']
missing_values = parameters['missing_values']
seed = parameters['seed']
train_val_split = parameters['train_val_split']
val_test_split = parameters['val_test_split']
scale_to = parameters['scale_to']
shift = parameters['shift']

# Not stored in parameters file, but needed to reload optimal masks:
input_features_short=[
    'slp', 
    'sst', 
    'z500', 
    'sat', 
    'sss', 
    'prec'
]

# Path to full data:
path_to_data = 'climate_index_collection/data/raw/2022-08-22/'

# Load data:
data = load_data_set(data_path=path_to_data, data_source_name=data_source_name)

# Extract time, latitude and longitude dimensions:
time = data['time']
latitude = data['lat']
longitude = data['lon']

# Get number of train and validation samples: Consider augmentation factor!
n_train = int(len(data[target_feature]) * train_val_split) - shift
n_val = int((len(data[target_feature]) - n_train - 2*shift) * val_test_split)
n_test = len(data[target_feature]) - n_train - n_val - 2*shift

# Compute monthly climatology over complete time span for whole world:
sst_climatology_fields = data[target_feature].groupby("time.month").mean("time")

# Get slp anomaly fields by subtracting monthly climatology from raw slp fields:
sst_anomaly_fields = data[target_feature].groupby("time.month") - sst_climatology_fields

# Initialize storage for loss per sample, dimension: (#missing values, #samples)
test_loss_per_sample_all = np.zeros((len(missing_values),n_test))

# Initialize storage for mean loss maps, dimension: (#missing values, latitude, longitude)
test_loss_map_all = np.zeros((len(missing_values),data[target_feature].shape[1],data[target_feature].shape[2]))

# Initialize storage for indices, dimension: (#missing values, #samples)
ENSO_test_pred_all = np.zeros((len(missing_values),n_test))
ENSO_test_target_all = np.zeros((len(missing_values),n_test))

AMO_test_pred_all = np.zeros((len(missing_values),n_test))
AMO_test_target_all = np.zeros((len(missing_values),n_test))


# # Restore input and target data, load final models, compute indices:

# Loop over array of desired amounts of missing values:
for i in range(len(missing_values)):

    # Get current relative amount of missing values:
    missing = missing_values[i]

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
        _,
        _,
        test_input,
        _,
        _,
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
    
    # Reload model: Rel. amount of missing values = 0.999 requires special treatment.
    if missing==0.999:
        model = tf.keras.models.load_model(Path(path_to_final_model) / 'missing_' f'{int(missing*1000)}' / 'model')
    else:
        model = tf.keras.models.load_model(Path(path_to_final_model) / 'missing_' f'{int(missing*100)}' / 'model')

    # Get predictions:
    test_pred = model.predict(test_input)
    
    # Compute loss per sample:
    test_loss_per_sample = np.mean((test_pred[:,:,:,0]-test_target)**2,axis=(1,2))
    
    # Store loss per sample:
    test_loss_per_sample_all[i] = test_loss_per_sample
        
    # Compute mean loss map:
    test_loss_map = np.mean((test_pred[:,:,:,0]-test_target)**2,axis=0)
    
    # Store mean loss map:
    test_loss_map_all[i] = test_loss_map
    
    ## Compute and store indices: SAM, NAO, NP
    
    # Convert predictions and targets to xarray.DataArray:
    test_pred_xr = xr.DataArray(
        test_pred[:,:,:,0],
        dims=('time', 'lat', 'lon'),
        coords={'time': time[-n_test:],
                'lat': latitude, 'lon': longitude}
    )
    
    test_target_xr = xr.DataArray(
        test_target[:,:,:],
        dims=('time', 'lat', 'lon'),
        coords={'time': time[-n_test:],
                'lat': latitude, 'lon': longitude}
    )
    
    
    # Revert scaling: Use min / max for correct feature (sst).
    test_pred_xr_rescaled = test_pred_xr * (train_max[1] - train_min[1]) + train_min[1]
    test_target_xr_rescaled = test_target_xr * (train_max[1] - train_min[1]) + train_min[1]

    # Add climatology, to restore raw fields:
    test_pred_xr_rescaled_fields = test_pred_xr_rescaled.groupby("time.month") + sst_climatology_fields
    test_target_xr_rescaled_fields = test_target_xr_rescaled.groupby("time.month") + sst_climatology_fields
        
    # Compute indices:
    ENSO_test_pred = el_nino_southern_oscillation_34(test_pred_xr_rescaled_fields).values
    ENSO_test_target = el_nino_southern_oscillation_34(test_target_xr_rescaled_fields).values

    AMO_test_pred = atlantic_multidecadal_oscillation(test_pred_xr_rescaled_fields).values
    AMO_test_target = atlantic_multidecadal_oscillation(test_target_xr_rescaled_fields).values
  
    # Store indices:
    ENSO_test_pred_all[i] = ENSO_test_pred
    ENSO_test_target_all[i] = ENSO_test_target

    AMO_test_pred_all[i] = AMO_test_pred
    AMO_test_target_all[i] = AMO_test_target


## Store results:
np.save(Path(path_to_final_model) / 'test_loss_per_sample_all.npy', test_loss_per_sample_all)
np.save(Path(path_to_final_model) / 'test_loss_map_all.npy', test_loss_map_all)
np.save(Path(path_to_final_model) / 'ENSO_test_pred_all.npy', ENSO_test_pred_all)
np.save(Path(path_to_final_model) / 'ENSO_test_target_all.npy', ENSO_test_target_all)
np.save(Path(path_to_final_model) / 'AMO_test_pred_all.npy', AMO_test_pred_all)
np.save(Path(path_to_final_model) / 'AMO_test_target_all.npy', AMO_test_target_all)