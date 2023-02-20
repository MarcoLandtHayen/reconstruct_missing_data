## Working on nesh with Container image py-da-tf-shap.sif:
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("./reconstruct_missing_data")
from timestamp_handling import fix_monthly_time_stamps

from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split, unary_union

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
    extended_data = np.zeros(
        (data.shape[0] * augmentation_factor, data.shape[1], data.shape[2])
    )

    # Loop over samples:
    for i in range(len(data)):

        # Loop over augmentation_facor:
        for j in range(augmentation_factor):

            # Store sample in extended data set:
            extended_data[i * augmentation_factor + j, :, :] = data[i, :, :]

    return extended_data


def create_missing_mask(data, mask_type, missing_type, missing_min, missing_max, seed):

    """Create mask for missing values fitting complete data's dimensions.
    Missing values are masked as zero (zero-inflated).

    Parameters
    ----------
    data: numpy.ndarray
        Data set containing complete 2D fields or flattened 1D fields.
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

    if mask_type == "fixed":

        # Get single mask of missing values and repeat this mask for all samples:
        np.random.seed(seed)

        # Check dimensions: Do we have flat or 2D field as data?
        if len(data.shape)==3:

            missing_mask_single = (
                np.random.uniform(low=0.0, high=1.0, size=(1, data.shape[1], data.shape[2]))
                > missing_min
            )
            
        else:
            
            missing_mask_single = (
                np.random.uniform(low=0.0, high=1.0, size=(1, data.shape[1]))
                > missing_min
            )

        missing_mask = np.repeat(missing_mask_single, data.shape[0], axis=0)

    elif mask_type == "variable":

        # Initialize mask from random uniform distribution in [0,1]:
        missing_mask = np.random.uniform(low=0.0, high=1.0, size=data.shape)

        # Initialize another mask from random uniform distribution in the desired range of missing values:
        missing_range = np.random.uniform(
            low=missing_min, high=missing_max, size=data.shape[0]
        )

        # Apply range mask to set amount of missing values for each sample with loop over samples:
        for i in range(data.shape[0]):
            missing_mask[i] = missing_mask[i] >= missing_range[i]

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
    if scale_to == "one_one":
        # Scale inputs to [-1,1]:
        data_scaled = 2 * (data - train_min) / (train_max - train_min) - 1

    elif scale_to == "zero_one":
        # Alternatively scale inputs to [0,1]
        data_scaled = (data - train_min) / (train_max - train_min)

    elif scale_to == "norm":
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

    return (
        train_input,
        val_input,
        train_target,
        val_target,
        train_min,
        train_max,
        train_mean,
        train_std,
    )


def eof_weights(dobj):
    """Empirical orthogonal functions (EOFs) can be thought of as the eigen-vectors of the spatial covariance matrix.
    Grid cells can be thought of as representing spatial averages which are low-pass filtering the raw signals and dampen the variance by a factor proportional to the square root of the cell size.
    For equidistant latitude/longitude grids the area weights are proportional to cos(latitude).
    Before applying Singular Value Decomposition (SVD), input data needs to be multiplied with the square root of the weights.


    Parameters
    ----------
    dobj: xarray.DataArray
        Contains the original input data.

    Returns
    -------
    xarray.DataArray
        Square root of weights needed to pre-process input data for SVD.
    """
    return np.sqrt(np.cos(np.deg2rad(dobj.coords["lat"])))


def mean_unweighted(dobj, dim=None):
    """Calculate an unweighted mean for one or more dimensions.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    dim: str or list
        Dimension name or list of dimension names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Averaged data. Has the same variable name(s) as dobj.

    """
    return dobj.mean(dim)


def area_mean_weighted(
    dobj,
    lat_south=None,
    lat_north=None,
    lon_west=None,
    lon_east=None,
    lon_name="lon",
    lat_name="lat",
    lat_extent_name=None,
    lon_extent_name=None,
):
    """Area average over lat and lon range.

    Parameters
    ----------
    dobj: xarray.DataArray
        Data for which the area average will be computed.
    lat_south: float
        Southern latitude bound. If "None", only the other lat bound will be applied.
    lat_north: float
        Northern latitude bound. If "None", only the other lat bound will be applied.
    lon_west: float
        Western longitude bound. If "None", only the other lon bound will be applied.
    lon_east: float
        Eastern longitude bound. If "None", only the other lon bound will be applied.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".
    lat_extent_name: str
        Name of the lat extent. Defaults to None.
    lon_extent_name: str
        Name of the lon extent. Defaults to None.

    Returns
    -------
    dobj:
        Area averaged input data.

    """
    # extract coords
    lat = dobj.coords[lat_name]
    lon = dobj.coords[lon_name]

    # coords
    mask = spatial_mask(
        dobj=dobj,
        lat_south=lat_south,
        lat_north=lat_north,
        lon_west=lon_west,
        lon_east=lon_east,
        lon_name=lon_name,
        lat_name=lat_name,
    )

    # do we have dimensional coords?
    have_dimensional_coords = (lat.dims[0] == lat_name) & (lon.dims[0] == lon_name)

    # prepare spatial coords for final sum
    if have_dimensional_coords:
        spatial_dims = [lat_name, lon_name]
    else:
        spatial_dims = list(set(lat.dims) + set(lon.dims))

    # prepare weights
    if have_dimensional_coords:
        # will only handle equidistant lons and lats here...
        dlon = abs(lon.diff(lon_name).mean())
        dlat = abs(lat.diff(lat_name).mean())
        dx = dlon * np.cos(np.deg2rad(lat))
        dy = dlat
        weights = dx * dy
    else:
        lat_extent = dobj.coords[lat_extent_name]
        lon_extent = dobj.coords[lon_extent_name]
        weights = lat_extent * lon_extent

    # weighted averaging
    averaged = (dobj * weights).where(mask).sum(spatial_dims) / weights.where(mask).sum(
        spatial_dims
    )

    return averaged


def spatial_mask(
    dobj,
    lat_south=None,
    lat_north=None,
    lon_west=None,
    lon_east=None,
    lon_name="lon",
    lat_name="lat",
):
    """Mask over lat and lon range.

    Parameters
    ----------
    dobj: xarray.DataArray
        Data for which the mask will be created.
    lat_south: float
        Southern latitude bound. If "None", only the other lat bound will be applied.
    lat_north: float
        Northern latitude bound. If "None", only the other lat bound will be applied.
    lon_west: float
        Western longitude bound. If "None", only the other lon bound will be applied.
    lon_east: float
        Eastern longitude bound. If "None", only the other lon bound will be applied.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".

    Returns
    -------
    dobj:
        Mask for given lat and lon range.

    """
    # extract coords
    lat = dobj.coords[lat_name]
    lon = dobj.coords[lon_name]

    # maybe standardize lon
    lon = lon % 360
    if lon_west is not None:
        lon_west = lon_west % 360
    if lon_east is not None:
        lon_east = lon_east % 360

    # catch unset lat and lon bounds
    if lon_west is None:
        lon_west = -np.inf
    if lon_east is None:
        lon_east = np.inf
    if lat_south is None:
        lat_south = -np.inf
    if lat_north is None:
        lat_north = np.inf

    # coords
    # check if lon range does not cross the greenwhich meridian at 0°W.
    if lon_west <= lon_east:
        return (
            (lat >= lat_south)
            & (lat <= lat_north)
            & (lon >= lon_west)
            & (lon <= lon_east)
        )
    # otherwise us 'or' instead of 'and' for the lon logical operation.
    else:
        return (
            (lat >= lat_south)
            & (lat <= lat_north)
            & ((lon >= lon_west) | (lon <= lon_east))
        )


def get_land_silhouette(data_path="data/test_data/", data_source_name="FOCI"):
    """Create silhouett surrounding land masses, to highlight continents.

    Parameters
    ----------
    data_path: str | path
        Location of the data files. Defaults to "data/test_data/".
    data_source_name: str
        Name of the model dataset. Defaults to "FOCI".

    Returns
    -------
    numpy.ndarray
        Two-dimensional boolean mask, matching lat/lon dimensions of raw data.

    """
    
    # Load data, including mask for ocean values:
    data = load_data_set(data_path=data_path, data_source_name=data_source_name)
    is_over_ocean = data['is_over_ocean'].values
    
    ## Derive mask for showing only the continents' silhouette as NaN values, for better orientation in slp fields.

    # Initialize storage for silhouette as boolean mask:
    land_silhouette = (np.zeros(is_over_ocean.shape)!=0)

    # Loop over latitude in ocean mask, to scan mask line-by-line:
    for i in range(is_over_ocean.shape[0]):

        # Loop over longitude, to scan current line:
        for j in range(is_over_ocean.shape[1]):

            # Check, if current grid point is over land, while previous grid point was over ocean.
            # Take care of initial border:
            if j>0:
                if (is_over_ocean[i,j]==False) & (is_over_ocean[i,j-1]==True):

                    # Set land silhouette:
                    land_silhouette[i,j] = True

            # Check, if current grid point is over ocean, while previous grid point was over land.
            # Take care of initial border:
            if j>0:
                if (is_over_ocean[i,j]==True) & (is_over_ocean[i,j-1]==False):

                    # Set land silhouette:
                    land_silhouette[i,j] = True

    # Loop over longitude in ocean mask, to scan mask row-by-row:
    for j in range(is_over_ocean.shape[1]):

        # Loop over latitude, to scan current row:
        for i in range(is_over_ocean.shape[0]):

            # Check, if current grid point is over land, while previous grid point was over ocean.
            # Take care of initial border:
            if i>0:
                if (is_over_ocean[i,j]==False) & (is_over_ocean[i-1,j]==True):

                    # Set land silhouette:
                    land_silhouette[i,j] = True

            # Check, if current grid point is over ocean, while previous grid point was over land.
            # Take care of initial border:
            if i>0:
                if (is_over_ocean[i,j]==True) & (is_over_ocean[i-1,j]==False):

                    # Set land silhouette:
                    land_silhouette[i,j] = True
    
    return land_silhouette 


def area_mean_weighted_polygon_selection(
    dobj,
    polygon_lon_lat=None,
    lon_name="lon",
    lat_name="lat",
    lat_extent_name=None,
    lon_extent_name=None,
):
    """Area average over polygon-selected region.

    Parameters
    ----------
    dobj: xarray.DataArray
        Data for which the area average will be computed.
    polygon_lon_lat: shapely Polygon
        (Multi)Polygon containing the region over which we want to average.
        If "None", all data will be used.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".
    lat_extent_name: str
        Name of the lat extent. Defaults to None.
    lon_extent_name: str
        Name of the lon extent. Defaults to None.

    Returns
    -------
    dobj:
        Area averaged input data.

    """
    # extract coords
    lat = dobj.coords[lat_name]
    lon = dobj.coords[lon_name]

    # mask
    # TODO: Pass polygon_lon_lat once through polygon_prime_meridian()?
    if polygon_lon_lat is not None:
        mask = polygon2mask(dobj=dobj, pg=polygon_lon_lat)
    else:
        mask = xr.ones_like(dobj).astype(bool)

    # do we have dimensional coords?
    have_dimensional_coords = (lat.dims[0] == lat_name) & (lon.dims[0] == lon_name)

    # prepare spatial coords for final sum
    if have_dimensional_coords:
        spatial_dims = [lat_name, lon_name]
    else:
        spatial_dims = list(set(lat.dims) + set(lon.dims))

    # prepare weights
    if have_dimensional_coords:
        # will only handle equidistant lons and lats here...
        dlon = abs(lon.diff(lon_name).mean())
        dlat = abs(lat.diff(lat_name).mean())
        dx = dlon * np.cos(np.deg2rad(lat))
        dy = dlat
        weights = dx * dy
    else:
        lat_extent = dobj.coords[lat_extent_name]
        lon_extent = dobj.coords[lon_extent_name]
        weights = lat_extent * lon_extent

    # weighted averaging
    averaged = (dobj * weights).where(mask).sum(spatial_dims) / weights.where(mask).sum(
        spatial_dims
    )

    return averaged


def polygon_prime_meridian(pg):
    """
    Transforms shapely Polygons or MultiPolygons defined in [180W, 180E) coords into [0E,360E) coords.
    Takes care of Polygons crossing the prime meridan.
    Polygon points are expected be (lon, lat) tuples.

    Parameters
    ----------
    pg: shaply Polygon or shapely MultiPolygon
        Polygon including the area wanted.

    Returns
    -------
    shapely MultPolygon
        shaply MultiPolygon containing at least one Polygon.
    """

    # handle empty Polygons and MultiPolygons
    if pg.is_empty:
        return MultiPolygon([pg])

    # create prime_meridian to eventually split the polygon
    prime_meridian = LineString([(0, 90), (0, 0), (0, -90)])
    pg_split = split(pg, prime_meridian)

    # create a list containing all Polygons given by the split operation
    # polygons on the negative (western) side of the prime meridian are translated into new coords, by adding 360 to the lon values.
    pg_list = []
    for temp_pg in pg_split.geoms:
        # check if the polygons minx is negative and add 360 to it.
        if temp_pg.bounds[0] < 0:
            temp_pg = translate(temp_pg, xoff=360)
        pg_list += [temp_pg]

    # create the multipolygon existing in [0E, 360E) coords from the list of polygons
    result = unary_union(pg_list)

    # for consistency always return MultiPolygon
    if type(result) is not MultiPolygon:
        # convert Polygon to MultiPolygon
        return MultiPolygon([result])
    else:
        return result

    
def monthly_anomalies_unweighted(dobj):
    """Calculates the monthly anomalies from the monthly climatology of a dataset.
    The monthly climatology is calculated using "monthly_mean"

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Monthly anomalies from the monthly climatology of the original data.
        Has the same variable name(s) as dobj.
    """
    # Note: the dobj needs to be grouped by the same group_dim as the DataArray returned by monthly_mean_unweighted!
    return (dobj.groupby("time.month") - monthly_mean_unweighted(dobj)).drop_vars(
        "month"
    )


def polygon2mask(dobj, pg, lat_name="lat", lon_name="lon"):
    """
    This funciton creates a mask for a given DataArray or DataSet based on a shapely Polygon or MultiPolygon.
    Polygon points are expected be (lon, lat) tuples.
    To fit the polygon to the dobj coords, "polygon_split_arbitrary" function is used.
    The dobj is expected to have lon values in [0E, 360E) coords and lat values in [90S, 90N] coords.


    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    pg: shapely Polygon or shapely MultiPolygon
        Polygon including the area wanted.
    lat_name: str
        Name of the latitude coordinate. Defaults to "lat".
    lon_name: str
        Name of the longitude coordinate. Defaults to "lon".

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Mask for given Polygon on dobj grid.
    """

    # handle prime meridian crossing Polygons and transform [180W, 180E) coords into [0E,360E) coords
    pg = polygon_split_arbitrary(
        pg=pg,
        lat_max=90,
        lat_min=-90,
        lon_max=360,
        lon_min=0,
    )

    # create the mask
    lon_2d, lat_2d = xr.broadcast(dobj.coords[lon_name], dobj.coords[lat_name])

    mask = xr.DataArray(
        np.reshape(
            [
                pg.contains(Point(_lon, _lat)) | pg.boundary.contains(Point(_lon, _lat))
                for _lon, _lat in zip(
                    np.ravel(lon_2d, order="C"), np.ravel(lat_2d, order="C")
                )
            ],
            lon_2d.shape,
            order="C",
        ),
        dims=lon_2d.dims,
        coords=lon_2d.coords,
    )
    # transpose to ensure the same order of horizontal dims as the input object
    mask = mask.transpose(*[d for d in dobj.dims if d in [lon_name, lat_name]])

    return mask



def polygon_split_arbitrary(
    pg,
    lon_min=0,
    lon_max=360,
    lat_min=-90,
    lat_max=90,
    max_iteration=1e3,
    max_extend=1e6,
):
    """
    Transforms shapely Polygons or MultiPolygons defined in arbitrary coords into
    other arbitrary coords bound by latitude and longitude boundaries.
    If necessary, the polygon is split and the splitted part will be transformed along the corresponding dimension to fit into the boundaries.
    Polygon points are expected be (lon, lat) tuples.



    Parameters
    ----------
    pg: shaply Polygon or shapely MultiPolygon
        Polygon including the area wanted.
    lon_min : float
        Minimum boundary of the longitude bounds
        Default to 0.
    lon_max :
        Maximum boundary of the longitude bounds.
        Default to 360.
    lat_min : float
        Minimum boundary of the longitude bounds
        Default to -90.
    lat_max :
        Maximum boundary of the longitude bounds.
        Default to 90.
    max_iteration : int
        Sets upper limit of splitting iterations across all longitude bounds.
    max_extend : int or float
        Maximum extend allowed for lat and lon.

    Returns
    -------
    shapely MultPolygon
        shaply MultiPolygon containing at least one Polygon.
    """
    check_bounds = lambda pg: pg.bounds[0] >= lon_min and pg.bounds[2] <= lon_max

    diff_lon = lon_max - lon_min
    diff_lat = lat_max - lat_min
    # handle empty Polygons and MultiPolygons
    if pg.is_empty:
        return MultiPolygon([pg])

    # create cutting lines at lon_min and lon_max to eventually split the polygon there.
    left_splitter = LineString(
        [(lon_min, max_extend), (lon_min, 0), (lon_min, -max_extend)]
    )
    right_splitter = LineString(
        [(lon_max, max_extend), (lon_max, 0), (lon_max, -max_extend)]
    )

    # create cutting lines at lat_min and lat_max to eventually split the polygon there.
    lower_splitter = LineString(
        [(-max_extend, lat_min), (0, lat_min), (max_extend, lat_min)]
    )
    upper_splitter = LineString(
        [(-max_extend, lat_max), (0, lat_max), (max_extend, lat_max)]
    )

    i = 0

    # LONGITUDE HANDLING
    # handle all polygon parts with longitude smaller than lon_min
    while not pg.bounds[0] >= lon_min and i < max_iteration:
        i += 1
        pg_list = []
        # while not split_done :
        pg_split = split(pg, left_splitter)
        for temp_pg in pg_split.geoms:
            # check if the polygons minx is negative and add 360 to it.
            if temp_pg.bounds[0] < lon_min:
                temp_pg = translate(temp_pg, xoff=diff_lon)
            pg_list += [temp_pg]
        pg = unary_union(pg_list)

    # handle all polygon parts with longitude greater than lon_max
    while not pg.bounds[2] <= lon_max and i < max_iteration:
        i += 1
        pg_list = []
        pg_split = split(pg, right_splitter)
        for temp_pg in pg_split.geoms:
            # check if the polygons minx is negative and add 360 to it.
            if temp_pg.bounds[2] > lon_max:
                temp_pg = translate(temp_pg, xoff=-diff_lon)
            pg_list += [temp_pg]
        pg = unary_union(pg_list)

    # LATUTUDE HANDLING
    # handle all polygon parts with latitude smaller than lon_min
    while not pg.bounds[1] >= lat_min and i < max_iteration:
        i += 1
        pg_list = []
        # while not split_done :
        pg_split = split(pg, lower_splitter)
        for temp_pg in pg_split.geoms:
            # check if the polygons minx is negative and add 360 to it.
            if temp_pg.bounds[1] < lat_min:
                temp_pg = translate(temp_pg, yoff=diff_lat)
            pg_list += [temp_pg]
        pg = unary_union(pg_list)

    # handle all polygon parts with latitude greater than lon_max
    while not pg.bounds[3] <= lat_max and i < max_iteration:
        i += 1
        pg_list = []
        pg_split = split(pg, upper_splitter)
        for temp_pg in pg_split.geoms:
            # check if the polygons minx is negative and add 360 to it.
            if temp_pg.bounds[3] > lat_max:
                temp_pg = translate(temp_pg, yoff=-diff_lat)
            pg_list += [temp_pg]
        pg = unary_union(pg_list)

    # create the multipolygon existing in the given boudaries from the list of polygons
    result = pg
    # for consistency always return MultiPolygon
    if type(result) is not MultiPolygon:
        # convert Polygon to MultiPolygon
        return MultiPolygon([result])
    else:
        return result
    
    
def monthly_mean_unweighted(dobj):
    """Calculates the monthly mean values of a dataset.

    Parameters
    ----------
    dobj: xarray.Dataset or xarray.DataArray
        Contains the original data.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Monthly mean data. Has the same variable name(s) as dobj.
        Dimension 'time' will be removed.
        Dimension 'month' is gained. Int values, starting with 1 for January.
    """
    return mean_unweighted(dobj=dobj.groupby("time.month"), dim="time")

