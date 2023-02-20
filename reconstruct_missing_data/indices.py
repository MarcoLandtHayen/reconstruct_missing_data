import numpy as np
import scipy as sp
import xarray as xr

from data_loading import (
    mean_unweighted,
    area_mean_weighted,
    area_mean_weighted_polygon_selection,
    monthly_anomalies_unweighted,
    polygon_prime_meridian,
    spatial_mask,
)

from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split, unary_union

    
def southern_annular_mode_zonal_mean(data_set):
    """Calculate the southern annular mode (SAM) index.

    This follows https://doi.org/10.1029/1999GL900003 in defining
    the southern annular mode index using zonally averaged sea-level pressure at 65°S and
    40°S.

    It differs from the definition in that it uses the raw time
    series of zonally averaged sea-level pressure and then only normalizes (zero mean,
    unit standard deviation) the difference of zonally avearged sea-level pressure at
    65°S and 40°S.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SLP field.

    Returns
    -------
    xarray.DataArray
        Time series containing the SAM index.

    """
    slp = data_set

    slp40S = mean_unweighted(slp.sel(lat=-40, method="nearest"), dim="lon")
    slp65S = mean_unweighted(slp.sel(lat=-65, method="nearest"), dim="lon")

    slp_diff = slp40S - slp65S

    SAM_index = (slp_diff - slp_diff.mean("time")) / slp_diff.std("time")
    SAM_index = SAM_index.rename("SAM_ZM")
    SAM_index.attrs["long_name"] = "southern_annular_mode_zonal_mean"

    return SAM_index


def north_atlantic_oscillation_station(data_set):
    """Calculate the station based North Atlantic Oscillation (NAO) index

    This uses station-based sea-level pressure closest to Reykjavik (64°9'N, 21°56'W) and
    Ponta Delgada (37°45'N, 25°40'W) and, largely following
    https://doi.org/10.1126/science.269.5224.676, defines the north-atlantic oscillation index
    as the difference of normalized in Reykjavik and Ponta Delgada without normalizing the
    resulting time series again. (This means that the north atlantic oscillation presented here
    has vanishing mean because both minuend and subtrahend have zero mean, but no unit
    standard deviation.)

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SLP field.

    Returns
    -------
    xarray.DataArray
        Time series containing the NAO index.

    """
    slp = data_set

    slp_northern_station = slp.sel(lat=64, lon=338, method="nearest")
    slp_southern_station = slp.sel(lat=38, lon=334, method="nearest")

    slp_northern_station_norm = (
        slp_northern_station - slp_northern_station.mean("time")
    ) / slp_northern_station.std("time")
    slp_southern_station_norm = (
        slp_southern_station - slp_southern_station.mean("time")
    ) / slp_southern_station.std("time")

    NAO_index = slp_northern_station_norm - slp_southern_station_norm
    NAO_index = NAO_index.rename("NAO_ST")
    NAO_index.attrs["long_name"] = "north_atlantic_oscillation_station"

    return NAO_index


def north_pacific(data_set):
    """Calculate the North Pacific index (NP)

    Following https://climatedataguide.ucar.edu/climate-data/north-pacific-np-index-trenberth-and-hurrell-monthly-and-winter
    the index is derived from area-weighted sea level pressure (SLP) anomalies in a box
    bordered by 30°N to 65°N and 160°E to 140°W. This translates to 30°N to 65°N and 160°E to 220°E.

    Computation is done as follows:
    1. Compute area averaged total SLP from region of interest.
    2. Compute monthly climatology for area averaged total SLP from that region.
    3. Subtract climatology from area averaged total SLP time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Note: Usually the index focusses on anomalies during November and March. Here we keep full information and
    compute monthly anomalies for all months of a year.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SLP field.

    Returns
    -------
    xarray.DataArray
        Time series containing the NP index.

    """
    slp = area_mean_weighted(
        dobj=data_set,
        lat_south=30,
        lat_north=65,
        lon_west=160,
        lon_east=220,
    )

    climatology = slp.groupby("time.month").mean("time")

    std_dev = slp.std("time")

    NP_index = (slp.groupby("time.month") - climatology) / std_dev
    NP_index = NP_index.rename("NP")
    NP_index.attrs["long_name"] = "north_pacific"

    return NP_index


def el_nino_southern_oscillation_34(data_set):
    """Calculate the El Nino Southern Oscillation 3.4 index (ENSO 3.4)

    Following https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    the index is derived from equatorial pacific sea-surface temperature (SST) anomalies in a box
    bordered by 5°S - 5°N and 170°W - 120°W. This translates to -5°N - 5°N and 190°E - 240°E.

    Computation is done as follows:
    1. Compute area averaged total SST from Niño 3.4 region.
    2. Compute monthly climatology for area averaged total SST from Niño 3.4 region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.
    4. Normalize anomalies by its standard deviation over the climatological period.

    Note: Usually the index is smoothed by taking some rolling mean over 5 months before
    normalizing. We omit the rolling mean here and directly take sst anomaly index instead,
    to preserve the information in full detail. And as climatology we use the complete time span,
    since we deal with model data.

    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing an SST field.

    Returns
    -------
    xarray.DataArray
        Time series containing the ENSO 3.4 index.

    """
    sst_nino34 = area_mean_weighted(
        dobj=data_set,
        lat_south=-5,
        lat_north=5,
        lon_west=190,
        lon_east=240,
    )

    climatology = sst_nino34.groupby("time.month").mean("time")

    std_dev = sst_nino34.std("time")

    ENSO34_index = (sst_nino34.groupby("time.month") - climatology) / std_dev
    ENSO34_index = ENSO34_index.rename("ENSO_34")
    ENSO34_index.attrs["long_name"] = "el_nino_southern_oscillation_34"

    return ENSO34_index



def atlantic_multidecadal_oscillation(data_set):
    """Calculate the Atlantic Multi-decadal Oscillation (AMO) index.

    This follows the NOAA method <https://psl.noaa.gov/data/timeseries/AMO/> in defining the Atlantic Multi-decadal Oscillation
    index using area weighted averaged sea-surface temperature anomalies of the north Atlantic between 0°N and 70°N,
    The anomalies are relative to a monthly climatology calculated from the whole time covered by the data set.
    It differs from the definition of the NOAA in that it does not detrend the time series and the smomothing is not performed.

    Computation is done as follows:
    1. Compute area averaged total SST from north Atlantic region.
    2. Compute monthly climatology for area averaged total SST from north Atlantic  region.
    3. Subtract climatology from area averaged total SST time series to obtain anomalies.

    Further informations can be found in :
    - [Trenberth and Shea, 2006] <https://doi.org/10.1029/2006GL026894>.
    - NCAR climate data guide <https://climatedataguide.ucar.edu/climate-data/atlantic-multi-decadal-oscillation-amo>


    Parameters
    ----------
    data_set: xarray.DataSet
        Dataset containing a SST field.

    Returns
    -------
    xarray.DataArray
        Time series containing the AMO index.

    """
    sst = data_set

    # create Atlantic polygon and calculate horizontal average
    atlanctic_polygon_lon_lat = polygon_prime_meridian(
        Polygon([(15, 0), (-65, 0), (-105, 25), (-45, 70), (15, 70), (-7, 35)])
    )
    sst_box_ave = area_mean_weighted_polygon_selection(
        dobj=sst, polygon_lon_lat=atlanctic_polygon_lon_lat
    )

    AMO = monthly_anomalies_unweighted(sst_box_ave)

    AMO = AMO.rename("AMO")
    AMO.attrs["long_name"] = "atlantic_multidecadal_oscillation"

    return AMO