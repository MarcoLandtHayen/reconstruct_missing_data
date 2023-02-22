# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct complete data from sparse inputs.
# Opposed to their net, only have 4 instead of 5 convolutional layers.
# Work with sea level pressure (slp) fields from Earth System Models, either FOCI or CESM.

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
    find_data_files, 
    load_data_set, 
    get_anomalies, 
    clone_data, 
    create_missing_mask, 
    split_and_scale_data,
    area_mean_weighted,
    spatial_mask,
)
from models import build_unet_4conv
from indices import (
    southern_annular_mode_zonal_mean,
    north_atlantic_oscillation_station,
    north_pacific,
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

# slp CESM:
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_CESM_fixed_discrete_factor_1_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_CESM_variable_discrete_factor_1_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_CESM_variable_discrete_factor_2_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_CESM_variable_discrete_factor_3_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_CESM_optimal_discrete_factor_1_final'

# slp FOCI:
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_FOCI_fixed_discrete_factor_1_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_FOCI_variable_discrete_factor_1_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_FOCI_variable_discrete_factor_2_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_FOCI_variable_discrete_factor_3_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_FOCI_optimal_discrete_factor_1_final'

# slp realworld:
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_realworld_fixed_discrete_factor_1_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_realworld_variable_discrete_factor_1_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_realworld_variable_discrete_factor_2_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_realworld_variable_discrete_factor_3_final'
#path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_realworld_optimal_from_CESM_discrete_factor_1_final'
path_to_final_model='GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/unet_4conv_slp_realworld_optimal_from_FOCI_discrete_factor_1_final'



# Reload parameters for this experiment:
with open(Path(path_to_final_model) / 'parameters.json', 'r') as f:
    parameters=load(f)

feature = parameters['feature']
feature_short = parameters['feature_short']
source = parameters['source']
mask_type = parameters['mask_type']
missing_type = parameters['missing_type']
augmentation_factor = parameters['augmentation_factor']
train_val_split = parameters['train_val_split']
missing_values = parameters['missing_values'] #or set manually: [0.999, 0.99, 0.95, 0.9, 0.75, 0.5] 
scale_to = parameters['scale_to']

# Real world data needs separate pre-processing:
if source == 'realworld':
    
    # Path to full data:
    path_to_data = "GitHub/MarcoLandtHayen/reconstruct_missing_data/data/raw/pres.sfc.mon.mean.nc"  

    ## Load data:

    # Open data set:
    slp_dataset=xr.open_dataset(path_to_data)

    # Extract time, latitude and longitude dimensions.
    # Already consider, that time dimension and latitude are sliced below: Omit last entry, in either case.
    time = slp_dataset['time'][:-1]
    latitude = slp_dataset['lat'][:-1]
    longitude = slp_dataset['lon']
   
    # Start with raw slp fields as lat/lon grids in time, from 1948 to 2022:
    slp_fields = (
        slp_dataset.pres
        .sel(time=slice('1948-01-01', '2022-12-01'))
    )
    
    # Get number of train and validation samples: Consider augmentation factor!
    n_train = int(len(slp_fields) * augmentation_factor * train_val_split)
    n_val = ((len(slp_fields) * augmentation_factor) - n_train)

    # Compute monthly climatology (here 1980 - 2009) for whole world:
    slp_climatology_fields = (
        slp_dataset.pres
        .sel(time=slice('1980-01-01','2009-12-01'))
        .groupby("time.month")
        .mean("time")
    )

    # Get slp anomaly fields by subtracting monthly climatology from raw slp fields:
    slp_anomaly_fields = slp_fields.groupby("time.month") - slp_climatology_fields

    # Remove last row (latidute), to have equal number of steps in latitude (=72). This serves as 'quick-and-dirty'
    # solution to avoid problems with UPSAMPLING in U-Net. There must be a more elegant way, take care of it later!
    slp_anomaly_fields = slp_anomaly_fields.values[:,:-1,:]

    # Extend data, if desired:
    data = clone_data(data=slp_anomaly_fields, augmentation_factor=augmentation_factor)

else:

    # Path to full data:
    path_to_data = 'climate_index_collection/data/raw/2022-08-22/'

    # Load data:
    data = load_data_set(data_path=path_to_data, data_source_name=source)

    # Extract time, latitude and longitude dimensions:
    time = data['time']
    latitude = data['lat']
    longitude = data['lon']

    # Get number of train and validation samples: Consider augmentation factor!
    n_train = int(len(data[feature]) * augmentation_factor * train_val_split)
    n_val = ((len(data[feature]) * augmentation_factor) - n_train)

    # Select single feature and compute anomalies, using whole time span as climatology:
    data = get_anomalies(feature=feature, data_set=data)

    # Extend data, if desired:
    data = clone_data(data=data, augmentation_factor=augmentation_factor)

    
# Extend time dimension, according to augmentation factor:
for t in range(len(time)):

    # Loop over augmentation_facor:
    for j in range(augmentation_factor):

        # Initialize storage for extended time line:
        if (t==0) & (j==0):
            extended_time=time[t].values
        else:
            extended_time = np.hstack([extended_time,time[t].values])

# Convert extended time line to xarray DataArray:
extended_time_xr = xr.DataArray(
    extended_time,
    name='time',
    dims=('time'),
    coords={'time': extended_time}
)

# Initialize storage for loss per sample, dimension: (#missing values, #samples)
train_loss_per_sample_all = np.zeros((len(missing_values),n_train))
val_loss_per_sample_all = np.zeros((len(missing_values),n_val))

# Initialize storage for mean loss maps, dimension: (#missing values, latitude, longitude)
train_loss_map_all = np.zeros((len(missing_values),data.shape[1],data.shape[2]))
val_loss_map_all = np.zeros((len(missing_values),data.shape[1],data.shape[2]))

# Initialize storage for indices, dimension: (#missing values, #samples)
SAM_train_pred_all = np.zeros((len(missing_values),n_train))
SAM_val_pred_all = np.zeros((len(missing_values),n_val))
SAM_train_target_all = np.zeros((len(missing_values),n_train))
SAM_val_target_all = np.zeros((len(missing_values),n_val))

NAO_train_pred_all = np.zeros((len(missing_values),n_train))
NAO_val_pred_all = np.zeros((len(missing_values),n_val))
NAO_train_target_all = np.zeros((len(missing_values),n_train))
NAO_val_target_all = np.zeros((len(missing_values),n_val))

NP_train_pred_all = np.zeros((len(missing_values),n_train))
NP_val_pred_all = np.zeros((len(missing_values),n_val))
NP_train_target_all = np.zeros((len(missing_values),n_train))
NP_val_target_all = np.zeros((len(missing_values),n_val))

# Loop over rel. amounts of missing values:
for i in range(len(missing_values)):

    # Get status:
    print('missing: ',i+1,' of ',len(missing_values))
    
    # Get current rel. amount of missing values:
    missing = missing_values[i]

    ## Reconstruct sparse data (as inputs) and complete data (as targets).
    
    # Reload mask for missing values.
    # Rel. amount of missing values = 0.999 requires special treatment:
    if missing==0.999:
        missing_mask = np.load(Path(path_to_final_model) / 'missing_' f'{int(missing*1000)}' / 'missing_mask.npy')
    else:
        missing_mask = np.load(Path(path_to_final_model) / 'missing_' f'{int(missing*100)}' / 'missing_mask.npy')

    # Use sparse data as inputs and complete data as targets. Split sparse and complete data into training and validation sets. 
    # Scale or normlalize data according to statistics obtained from only training data.
    train_input, val_input, train_target, val_target, train_min, train_max, train_mean, train_std = split_and_scale_data(
        data, 
        missing_mask, 
        train_val_split, 
        scale_to
    )
    
    # Reload model: Rel. amount of missing values = 0.999 requires special treatment.
    if missing==0.999:
        model = tf.keras.models.load_model(Path(path_to_final_model) / 'missing_' f'{int(missing*1000)}' / 'model')
    else:
        model = tf.keras.models.load_model(Path(path_to_final_model) / 'missing_' f'{int(missing*100)}' / 'model')

    # Get predictions:
    train_pred = model.predict(train_input)
    val_pred = model.predict(val_input)
    
    # Compute loss per sample:
    train_loss_per_sample = np.mean((train_pred[:,:,:,0]-train_target)**2,axis=(1,2))
    val_loss_per_sample = np.mean((val_pred[:,:,:,0]-val_target)**2,axis=(1,2))
    
    # Store loss per sample:
    train_loss_per_sample_all[i] = train_loss_per_sample
    val_loss_per_sample_all[i] = val_loss_per_sample
        
    # Compute mean loss maps:
    train_loss_map = np.mean((train_pred[:,:,:,0]-train_target)**2,axis=0)
    val_loss_map = np.mean((val_pred[:,:,:,0]-val_target)**2,axis=0)
    
    # Store mean loss maps:
    train_loss_map_all[i] = train_loss_map
    val_loss_map_all[i] = val_loss_map
    
    ## Compute and store indices: SAM, NAO, NP
    
    # Convert predictions and targets to xarray.DataArray:
    train_pred_xr = xr.DataArray(
        train_pred[:,:,:,0],
        dims=('time', 'lat', 'lon'),
        coords={'time': extended_time[:n_train], 'lat': latitude, 'lon': longitude}
    )
    val_pred_xr = xr.DataArray(
        val_pred[:,:,:,0],
        dims=('time', 'lat', 'lon'),
        coords={'time': extended_time[n_train:], 'lat': latitude, 'lon': longitude}
    )
    train_target_xr = xr.DataArray(
        train_target[:,:,:],
        dims=('time', 'lat', 'lon'),
        coords={'time': extended_time[:n_train], 'lat': latitude, 'lon': longitude}
    )
    val_target_xr = xr.DataArray(
        val_target[:,:,:],
        dims=('time', 'lat', 'lon'),
        coords={'time': extended_time[n_train:], 'lat': latitude, 'lon': longitude}
    )   
    
    # Revert scaling:
    train_pred_xr_rescaled = train_pred_xr * (train_max - train_min) + train_min
    val_pred_xr_rescaled = val_pred_xr * (train_max - train_min) + train_min
    train_target_xr_rescaled = train_target_xr * (train_max - train_min) + train_min
    val_target_xr_rescaled = val_target_xr * (train_max - train_min) + train_min

    # Add climatology, to restore raw fields:
    train_pred_xr_rescaled_fields = train_pred_xr_rescaled.groupby("time.month") + slp_climatology_fields[:,:-1,:]
    val_pred_xr_rescaled_fields = val_pred_xr_rescaled.groupby("time.month") + slp_climatology_fields[:,:-1,:]
    train_target_xr_rescaled_fields = train_target_xr_rescaled.groupby("time.month") + slp_climatology_fields[:,:-1,:]
    val_target_xr_rescaled_fields = val_target_xr_rescaled.groupby("time.month") + slp_climatology_fields[:,:-1,:]

    # Compute indices:
    SAM_train_pred = southern_annular_mode_zonal_mean(train_pred_xr_rescaled_fields).values
    SAM_val_pred = southern_annular_mode_zonal_mean(val_pred_xr_rescaled_fields).values
    SAM_train_target = southern_annular_mode_zonal_mean(train_target_xr_rescaled_fields).values
    SAM_val_target = southern_annular_mode_zonal_mean(val_target_xr_rescaled_fields).values
    
    NAO_train_pred = north_atlantic_oscillation_station(train_pred_xr_rescaled_fields).values
    NAO_val_pred = north_atlantic_oscillation_station(val_pred_xr_rescaled_fields).values
    NAO_train_target = north_atlantic_oscillation_station(train_target_xr_rescaled_fields).values
    NAO_val_target = north_atlantic_oscillation_station(val_target_xr_rescaled_fields).values

    NP_train_pred = north_pacific(train_pred_xr_rescaled_fields).values
    NP_val_pred = north_pacific(val_pred_xr_rescaled_fields).values
    NP_train_target = north_pacific(train_target_xr_rescaled_fields).values
    NP_val_target = north_pacific(val_target_xr_rescaled_fields).values
  
    # Store indices:
    SAM_train_pred_all[i] = SAM_train_pred
    SAM_val_pred_all[i] = SAM_val_pred
    SAM_train_target_all[i] = SAM_train_target
    SAM_val_target_all[i] = SAM_val_target

    NAO_train_pred_all[i] = NAO_train_pred
    NAO_val_pred_all[i] = NAO_val_pred
    NAO_train_target_all[i] = NAO_train_target
    NAO_val_target_all[i] = NAO_val_target

    NP_train_pred_all[i] = NP_train_pred
    NP_val_pred_all[i] = NP_val_pred
    NP_train_target_all[i] = NP_train_target
    NP_val_target_all[i] = NP_val_target
    
## Store results:
np.save(Path(path_to_final_model) / 'train_loss_per_sample_all.npy', train_loss_per_sample_all)
np.save(Path(path_to_final_model) / 'val_loss_per_sample_all.npy', val_loss_per_sample_all)
np.save(Path(path_to_final_model) / 'train_loss_map_all.npy', train_loss_map_all)
np.save(Path(path_to_final_model) / 'val_loss_map_all.npy', val_loss_map_all)
np.save(Path(path_to_final_model) / 'SAM_train_pred_all.npy', SAM_train_pred_all)
np.save(Path(path_to_final_model) / 'SAM_val_pred_all.npy', SAM_val_pred_all)
np.save(Path(path_to_final_model) / 'SAM_train_target_all.npy', SAM_train_target_all)
np.save(Path(path_to_final_model) / 'SAM_val_target_all.npy', SAM_val_target_all)
np.save(Path(path_to_final_model) / 'NAO_train_pred_all.npy', NAO_train_pred_all)
np.save(Path(path_to_final_model) / 'NAO_val_pred_all.npy', NAO_val_pred_all)
np.save(Path(path_to_final_model) / 'NAO_train_target_all.npy', NAO_train_target_all)
np.save(Path(path_to_final_model) / 'NAO_val_target_all.npy', NAO_val_target_all)
np.save(Path(path_to_final_model) / 'NP_train_pred_all.npy', NP_train_pred_all)
np.save(Path(path_to_final_model) / 'NP_val_pred_all.npy', NP_val_pred_all)
np.save(Path(path_to_final_model) / 'NP_train_target_all.npy', NP_train_target_all)
np.save(Path(path_to_final_model) / 'NP_val_target_all.npy', NP_val_target_all)