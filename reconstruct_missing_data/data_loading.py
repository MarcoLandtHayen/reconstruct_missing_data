from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


## Working on nesh with Container image py-da-tf-shap.sif:
import sys
sys.path.append('./reconstruct_missing_data')
from timestamp_handling import fix_monthly_time_stamps

## Working on nesh with Container image reconstruct_missing_data_latest.sif:
# from .timestamp_handling import fix_monthly_time_stamps


def find_data_files(data_path="data/test_data/", data_source_name="FOCI"):
    """Find all files for given data source.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    Returns
    -------
    list
        Paths to all data files.

    """
    data_files = list(sorted(Path(data_path).glob(f"{data_source_name}/*.nc")))
    return data_files


def load_and_preprocess_single_data_file(file_name, **kwargs):
    """Load and preprocess individual data file.

    Current pre-processing steps:
    - squeeze (to get rid of, e.g., some of the vertical degenerate dims)
    - fixing monthly timestamps

    Parameters
    ----------
    file_name: str or pathlike
        File name.

    All kwargs will be passed on to xarray.open_dataset().

    Returns
    -------
    xarray.Dataset
        Dataset.

    """
    ds = xr.open_dataset(file_name, **kwargs)

    # get rid of singleton dims
    ds = ds.squeeze()

    # fix time stamps to always be mid month
    ds = fix_monthly_time_stamps(ds)

    return ds


def load_and_preprocess_multiple_data_files(file_names, **kwargs):
    """Loads and preprocesses multiple files.

    Parameters
    ---------
    file_names: iterable
        File names (str or pathlike).

    All kwargs will be passed on to load_and_preprocess_single_data_file().

    Returns
    -------
    xarray.Dataset
        Dataset.

    """
    return xr.merge(
        (load_and_preprocess_single_data_file(fname, **kwargs) for fname in file_names)
    )


def load_data_set(data_path="data/test_data/", data_source_name="FOCI", **kwargs):
    """Load dataset.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    All kwargs will be passed to xarray.open_mfdataset. Use this for, e.g., chunking.

    Returns
    -------
    xarray.Dataset
        Multifile dataset with (pointers to) all data.

    """
    data_files = find_data_files(data_path=data_path, data_source_name=data_source_name)
    raw_data_set = load_and_preprocess_multiple_data_files(data_files, **kwargs)
    data_set = standardize_metadata(raw_data_set, data_source_name=data_source_name)
    data_set = mask_ocean_only_vars(data_set)

    return data_set


# this is dangerous...
VARNAME_MAPPING = {
    "FOCI": {
        "slp": "sea-level-pressure",
        "tsw": "sea-surface-temperature",
        "geopoth": "geopotential-height",
        "temp2": "surface-air-temperature",
        "sosaline": "sea-surface-salinity",
        "precip": "precipitation",
    },
    "CESM": {
        "PSL": "sea-level-pressure",
        "SST": "sea-surface-temperature",
        "Z3": "geopotential-height",
        "TS": "surface-air-temperature",
        "SALT": "sea-surface-salinity",
        "PRECT": "precipitation",
    },
}

OCEAN_ONLY_VARS = (
    "sea-surface-temperature",
    "sea-surface-salinity",
)


def standardize_metadata(raw_data_set=None, data_source_name="FOCI"):
    """Standardize metadata (dims, coords, attributes, varnames).

    Parameters
    ----------
    raw_data_set: xarray.Dataset
        Dataset with potentially non-standard metadata.
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    Returns
    xarray.Dataset
        Dataset with standardised metadata.

    """

    data_set = raw_data_set.rename_vars(VARNAME_MAPPING[data_source_name])
    return data_set


def mask_ocean_only_vars(data_set, get_mask_from="sea-surface-salinity"):
    """Mask those fields which are only defined over water.

    Parameters
    ----------
    data_set: xarray.Dataset
        Contains all variables.
    get_mask_from: str
        Bootstrap the mask from this var. Defaults to "sea-surface-salinity".

    Returns
    -------
    xarray.Dataset
        Same as data_set but with the respective vars masked out over land.

    """
    is_over_ocean = ~data_set[get_mask_from].isel(time=0, drop=True).isnull()
    data_set["is_over_ocean"] = is_over_ocean.rename("is_over_ocean")

    for mask_me in OCEAN_ONLY_VARS:
        data_set[mask_me] = data_set[mask_me].where(is_over_ocean)

    return data_set


def get_anomalies(feature, data_set):
    """Reduce data set by selecting single feature. For this feature, compute anomalies by subtracting seasonal cycle
    Use the whole time span as climatology.
    
    Parameters
    ----------
    feature: string
        Specify single feature to select.
    data_set: xarray.Dataset
        Contains single feature.

    Returns
    -------
    numpy.ndarray
        Obtained anomalies for selected feature.

    """
    
    # Select single feature:
    data = data_set[feature]

    # Compute anomalies, using whole time span as climatology:
    data = data.groupby("time.month") - data.groupby("time.month").mean("time")
    
    return data.values
    

def clone_data(data, augmentation_factor):
    
    """Clone each data sample multiple times, keeping the original order.
    Having e.g. three samples A,B,C, the resulting cloned collection with augmentation_factor=2 is: A,A,B,B,C,C.

    Parameters
    ----------
    data: numpy.ndarray
        Data set containing samples to be cloned.
    augmentation_factor: int
        Number of times, each sample is to be cloned.
   
    Returns
    -------
    numpy.ndarray
        Extended data.
    """

    # Initialize storage for extended data set. Dimension: (#samples * factor, latitude, longitude)
    extended_data = np.zeros((data.shape[0]*augmentation_factor, data.shape[1], data.shape[2]))
    
    # Loop over samples:
    for i in range(len(data)):
        
        # Loop over augmentation_facor:
        for j in range(augmentation_factor):
            
            # Store sample in extended data set:
            extended_data[i*augmentation_factor+j,:,:] = data[i,:,:]
            
    return extended_data


def create_missing_mask(data, mask_type, missing_type, missing_min, missing_max, seed):
    
    """Create mask for missing values fitting complete data's dimensions.
    Missing values are masked as zero (zero-inflated).

    Parameters
    ----------
    data: numpy.ndarray
        Data set containing complete 2D fields.
    mask_type: string
        Can have random mask for missing values, individually for each data sample ('variable').
        Or create only a single random mask, that is then applied to all samples identically ('fixed').
    missing_type: string
        Either specify a discrete amount of missing values ('discrete') or give a range ('range').
        Giving a range only makes sense for mask_type='variable'.
    missing_min, missing_max: float
        Specify the range of allowed relative amounts of missing values.
        For mask_type='fixed', both values are set identically and give the desired amount of missing values.
        For mask_type='variable' and missing_type='discrete', also set both values identically to give the discrete amount of missing values.
        Only for mask_type='variable' with missing_type='range', set the minimum and maximum relative amount of missing values, according to desired range.
    seed: int
        Seed for random number generator, for reproducibility.
   
    Returns
    -------
    boolean
        Mask for missing values.
    """
    
    if mask_type=='fixed':
        
        # Get single mask of missing values and repeat this mask for all samples:
        np.random.seed(seed)
        missing_mask_single = (np.random.uniform(low=0.0, high=1.0, size=(1, data.shape[1], data.shape[2]))>missing_min)
        missing_mask = np.repeat(missing_mask_single,data.shape[0],axis=0)
        
    elif mask_type=='variable':

        # Initialize mask from random uniform distribution in [0,1]:
        missing_mask = np.random.uniform(low=0.0, high=1.0, size=data.shape)
        
        # Initialize another mask from random uniform distribution in the desired range of missing values:
        missing_range = np.random.uniform(low=missing_min, high=missing_max, size=data.shape[0])
        
        # Apply range mask to set amount of missing values for each sample with loop over samples:
        for i in range(data.shape[0]):
            missing_mask[i] = (missing_mask[i] >= missing_range[i])        
   
    return missing_mask


def split_and_scale_data(data, missing_mask, train_val_split, scale_to):
    
    """Optionally scale or normalize values, according to statistics obtained from training data.
    Then apply mask for missing values and split data into training and validation sets.
    Existing NaN values are set to zero.

    Parameters
    ----------
    data: numpy.ndarray
        Data set containing complete 2D fields, used as targets.
    missing_mask: numpy.ndarray
        Mask for missing values fitting complete data's dimensions.
    train_val_split: float
        Relative amount of training data.
    scale_to: string
        Specifies the desired scaling. Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.
   
    Returns
    -------
    train_input, val_input, train_target, val_target: numpy.ndarray
        Data sets containing training and validation inputs and targets, respectively.
    train_min, train_max, train_mean, train_std: float
        Statistics obtained from training data: Minimum, maximum, mean and standard deviation, respectively.    
    """
   
    # Get number of train samples:
    n_train = int(len(data) * train_val_split)

    # Optionally scale inputs to [-1,1] or [0,1], according to min/max obtained from only train inputs. 
    # Or normalize inputs to have zero mean and unit variance. 

    # Look for NaN values:
    invalid_gridpoints = np.isnan(data)
    
    # Set NaN values to zero:
    data[invalid_gridpoints] = 0
    
    # Remenber min/max used for scaling.
    train_min = np.min(data[:n_train])
    train_max = np.max(data[:n_train])

    # Remenber mean and std dev used for scaling.
    train_mean = np.mean(data[:n_train])
    train_std = np.std(data[:n_train])

    # Scale or normalize inputs depending on desired scaling parameter:
    if scale_to == 'one_one':
        # Scale inputs to [-1,1]:
        data_scaled = 2 * (data - train_min) / (train_max - train_min) - 1

    elif scale_to == 'zero_one':
        # Alternatively scale inputs to [0,1]
        data_scaled = (data - train_min) / (train_max - train_min)

    elif scale_to == 'norm':
        # Alternatively scale inputs to [0,1]
        data_scaled = (data - train_mean) / train_std

    # Get sparse data by applying given mask for missing values to scaled/normalized data:
    data_sparse_scaled = data_scaled * missing_mask

    # Again set former NaN values to zero, after scaling / normalizing:
    data_scaled[invalid_gridpoints] = 0
    data_sparse_scaled[invalid_gridpoints] = 0
        
    ## Split inputs and targets:
    train_input = data_sparse_scaled[:n_train]
    val_input = data_sparse_scaled[n_train:]
    train_target = data_scaled[:n_train]
    val_target = data_scaled[n_train:]

    # Add dimension for number of channels, required for Conv2D:
    train_input = np.expand_dims(train_input, axis=-1)
    val_input = np.expand_dims(val_input, axis=-1)
    
    return train_input, val_input, train_target, val_target, train_min, train_max, train_mean, train_std