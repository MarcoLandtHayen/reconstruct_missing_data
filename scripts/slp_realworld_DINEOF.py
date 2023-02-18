# Description:
#
# DINEOF for slp realworld fields.


import sys
sys.path.append(
    "GitHub/MarcoLandtHayen/reconstruct_missing_data/reconstruct_missing_data"
)

from data_loading import (
    find_data_files, 
    load_data_set, 
    get_anomalies, 
    create_missing_mask, 
    split_and_scale_data, 
    eof_weights, 
    get_land_silhouette,
)

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import xarray as xr
from pathlib import Path
from json import dump, load
import os




# Specify experiment:
feature = 'sea-level-pressure' # Choose either 'sea-level-pressure' or 'sea-surface-temperature' as feature.
feature_short = 'slp' # Free to set short name, to store results, e.g. 'slp' and 'sst'.
source = 'realworld' # Choose Earth System Model, either 'FOCI' or 'CESM'.
mask_type = "fixed"  # Can have random missing values, individually for each data sample ('variable'),
# or randomly create only a single mask, that is then applied to all samples identically ('fixed').
missing_type = "discrete"  # Either specify discrete amounts of missing values ('discrete') or give a range ('range').
augmentation_factor = 1  # Number of times, each sample is to be cloned, keeping the original order.
train_val_split = 0.8  # Set rel. amount of samples used for training.
scale_to = "zero_one"  # Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.

# Set run name:
run = 'run_4' 

# Specify time steps, lat and lon dimensions:
n_time = 120
n_lat = 12 # -1 for full data, when working with real world slp fields
n_lon = 12

# Specify number of EOFs to consider for reconstruction:
n_eof = 10

# Specify max. number of iterations:
n_iter = 1000000

# Set further parameters:
seed = 1  # Seed for random number generator, for reproducibility of missing value mask.
seed_reserved = 4 # Additional seed to create independent mask of reserved grid points.
missing = 0.2 # Set rate of missing values.
reserved = 0.1 # Set rate of valid values reserved for cross-validation.

# Get path to store results to:
path_to_store_results = Path('GitGeomar/marco-landt-hayen/reconstruct_missing_data_results/'+feature_short+'_'+source+'_DINEOF/'+run)

# Try to create folder for later saving results, avoid overwriting existing results:
os.makedirs(path_to_store_results, exist_ok=False)

# Store parameters as json:
parameters = {
    "feature": feature,
    "feature_short": feature_short,
    "source": source,
    "mask_type": mask_type,
    "missing_type": missing_type,
    "augmentation_factor": augmentation_factor,
    "run": run,
    "train_val_split": train_val_split,
    "scale_to": scale_to,
    "n_time": n_time,
    "n_lat": n_lat,
    "n_lon": n_lon,
    "n_eof": n_eof,
    "n_iter": n_iter,
    "seed": seed,
    "seed_reserved": seed_reserved,
    "missing": missing,
    "reserved": reserved,    
}

with open(path_to_store_results / "parameters.json", "w") as f:
    dump(parameters, f)

    
## Load data, including ALL features and mask for Ocean values:

if source=='realworld':
    
    # Specify path to data:
    path_to_data = "GitHub/MarcoLandtHayen/reconstruct_missing_data/data/raw/pres.sfc.mon.mean.nc"
    
    # Open data set:
    slp_dataset=xr.open_dataset(path_to_data)

    # Start with raw slp fields as lat/lon grids in time, from 1948 to 2022:
    slp_fields = (
        slp_dataset.pres
        .sel(time=slice('1948-01-01', '2022-12-01'))
    )

    # Compute monthly climatology (here 1980 - 2009) for whole world:
    slp_climatology_fields = (
        slp_dataset.pres
        .sel(time=slice('1980-01-01','2009-12-01'))
        .groupby("time.month")
        .mean("time")
    )

    # Get slp anomaly fields by subtracting monthly climatology from raw slp fields:
    slp_anomaly_fields = slp_fields.groupby("time.month") - slp_climatology_fields
    
    # When working with complete samples in spatial extend:
    # Remove last row (latidute), to have even number of steps in latitude (=72). This served as 'quick-and-dirty'
    # solution to avoid problems with UPSAMPLING in U-Net. There must be a more elegant way, take care of it later!
    # This step is not essential for EOF, but keep it similar to U-Net approach.
    feature_anomaly = slp_anomaly_fields.values[:n_time,:n_lat,:n_lon]
    
    # Flatten spatial dimensions.
    feature_anomaly_flat = feature_anomaly.reshape((n_time,-1))

    # Remove mean over time, so that every grid point's values have zero mean over time.
    feature_anomaly_flat_zeromean = feature_anomaly_flat - np.mean(feature_anomaly_flat,axis=0)

else:

    # Specify path to data: 
    data_path='climate_index_collection/data/raw/2022-08-22/'

    # Load data:
    data = load_data_set(data_path=data_path, data_source_name=source)

    # Extract feature:
    feature_raw = data[feature]
    
    # Remove seasonal cycle to get anomalies, use whole time span as climatology:
    climatology = feature_raw.groupby("time.month").mean("time")
    feature_anomaly = (feature_raw.groupby("time.month") - climatology).drop("month")[:n_time,:n_lat,:n_lon]
    
    # Flatten spatial dimensions.
    feature_anomaly_flat = feature_anomaly.stack(tmp_space=("lat", "lon")).dropna(dim="tmp_space")

    # Remove mean over time, so that every grid point's values have zero mean over time.
    # Additionally extract values, to have np.array:
    feature_anomaly_flat_zeromean = (feature_anomaly_flat - feature_anomaly_flat.mean(axis=0)).values
    
    

# Get missing mask, fitting to flat feature anomaly with zero mean.
# Missing grid points are masked as 'False'.
missing_mask = create_missing_mask(
    data=feature_anomaly_flat_zeromean, 
    mask_type=mask_type, 
    missing_type=missing_type, 
    missing_min=missing,
    missing_max=missing,
    seed=seed).astype('bool')

# Get mask for grid points that are reserved for cross-validation, fitting to flat feature anomaly with zero mean.
# To reach the specified 'reserved' rate, need to take the rate of missing values into account.
# Reserved grid points are masked as 'False'.
reserved_mask = create_missing_mask(
    data=feature_anomaly_flat_zeromean, 
    mask_type=mask_type, 
    missing_type=missing_type, 
    missing_min=reserved/(1-missing),
    missing_max=reserved/(1-missing),
    seed=seed_reserved).astype('bool')

# Now combine both masks to have reserved_gridpoints as subset of valid grid points (not missing!):
reserved_gridpoints = (reserved_mask==0) * (missing_mask==1)

# And derive missing_gridpoints, for convenience:
missing_gridpoints = (missing_mask==0)

# Set missing values to zero:
feature_anomaly_flat_zeromean_missing = feature_anomaly_flat_zeromean * missing_mask

# Initialize storage for total loss and loss for reserved grid points:
total_loss = []
reserved_loss = []
missing_loss = []
abs_reconstruction = []

# Start iteration:
for iter in range(n_iter):
    
    # Perform SVD on flat feature anomaly with zero mean, missing values set to zero:
    pc, s, eof = sp.linalg.svd(
        feature_anomaly_flat_zeromean_missing, full_matrices=False
    )

    # Reconstruct flat feature anomaly:
    feature_anomaly_flat_zeromean_reconstruct = np.matmul((pc[:,:n_eof] * s[:n_eof]), eof[:n_eof])
    
    # Update former missing values by reconstructed values:
    feature_anomaly_flat_zeromean_missing[missing_gridpoints] = feature_anomaly_flat_zeromean_reconstruct[missing_gridpoints]
  
    # Get and store reconstruction loss for complete samples, reserved grid points and grid points with missing values:
    total_loss.append(np.mean((feature_anomaly_flat_zeromean - feature_anomaly_flat_zeromean_reconstruct)**2))
    reserved_loss.append(np.mean((feature_anomaly_flat_zeromean[reserved_gridpoints] - feature_anomaly_flat_zeromean_reconstruct[reserved_gridpoints])**2))
    missing_loss.append(np.mean((feature_anomaly_flat_zeromean[missing_gridpoints] - feature_anomaly_flat_zeromean_reconstruct[missing_gridpoints])**2))
        
    # Get and store summed absolute reconstructed missing values:
    abs_reconstruction.append(np.sum(np.abs(feature_anomaly_flat_zeromean_missing[missing_gridpoints])))
    
    # Store results every 10,000 iterations:
    if (iter+1)%100000 == 0:
        np.save(path_to_store_results / 'total_loss.npy', np.array(total_loss))
        np.save(path_to_store_results / 'reserved_loss.npy', np.array(reserved_loss))
        np.save(path_to_store_results / 'missing_loss.npy', np.array(missing_loss))
        np.save(path_to_store_results / 'abs_reconstruction.npy', np.array(abs_reconstruction))

        # Set filename to store snapshot of reconstruction after current number of iterations:
        feature_anomaly_flat_zeromean_missing_filename = 'feature_anomaly_flat_zeromean_missing_'+str(iter+1)+'_iterations.npy'
        np.save(path_to_store_results / feature_anomaly_flat_zeromean_missing_filename,feature_anomaly_flat_zeromean_missing)
        