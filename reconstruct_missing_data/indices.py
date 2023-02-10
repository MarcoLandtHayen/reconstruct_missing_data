import numpy as np
import scipy as sp
import xarray as xr

from data_loading import (
    mean_unweighted,
    area_mean_weighted,
    spatial_mask,
)

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