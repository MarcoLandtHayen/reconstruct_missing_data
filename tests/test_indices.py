from pathlib import Path

#import cftime
import numpy as np
import pytest
import scipy as sp
import xarray as xr

from numpy.testing import assert_allclose, assert_almost_equal

from reconstruct_missing_data.data_loading import VARNAME_MAPPING, load_data_set
from reconstruct_missing_data.indices import (
    atlantic_multidecadal_oscillation,
    el_nino_southern_oscillation_34,
    north_atlantic_oscillation_station,
    north_pacific,
    sahel_precipitation,
    southern_annular_mode_zonal_mean,
)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_ZM_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_ZM = southern_annular_mode_zonal_mean(data_set["sea-level-pressure"])

    # Check, if calculated index only has one dimension: 'time'
    assert SAM_ZM.dims[0] == "time"
    assert len(SAM_ZM.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_ZM_standardisation(source_name):
    """Ensure that standardisation works correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_ZM = southern_annular_mode_zonal_mean(data_set["sea-level-pressure"])

    # Check, if calculated index has zero mean and unit std dev:
    assert_almost_equal(actual=SAM_ZM.mean("time").values[()], desired=0, decimal=3)
    assert_almost_equal(actual=SAM_ZM.std("time").values[()], desired=1, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_SAM_ZM_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    SAM_ZM = southern_annular_mode_zonal_mean(data_set["sea-level-pressure"])

    assert SAM_ZM.name == "SAM_ZM"
    assert SAM_ZM.long_name == "southern_annular_mode_zonal_mean"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_ST_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_ST = north_atlantic_oscillation_station(data_set["sea-level-pressure"])

    # Check, if calculated index only has one dimension: 'time'
    assert NAO_ST.dims[0] == "time"
    assert len(NAO_ST.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_ST_zeromean(source_name):
    """Ensure that index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_ST = north_atlantic_oscillation_station(data_set["sea-level-pressure"])

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=NAO_ST.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NAO_ST_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NAO_ST = north_atlantic_oscillation_station(data_set["sea-level-pressure"])

    assert NAO_ST.name == "NAO_ST"
    assert NAO_ST.long_name == "north_atlantic_oscillation_station"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_enso34_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3.4 index
    ENSO34 = el_nino_southern_oscillation_34(data_set["sea-surface-temperature"])

    # Check, if calculated index only has one dimension: 'time'
    assert ENSO34.dims[0] == "time"
    assert len(ENSO34.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO34_zeromean(source_name):
    """Ensure that ENSO 3.4 has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3.4 index
    ENSO34 = el_nino_southern_oscillation_34(data_set["sea-surface-temperature"])

    # Check, if calculated ENSO 3.4 index has zero mean:
    assert_almost_equal(actual=ENSO34.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_ENSO34_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate ENSO 3.4 index
    result = el_nino_southern_oscillation_34(data_set["sea-surface-temperature"])

    assert result.name == "ENSO_34"
    assert result.long_name == "el_nino_southern_oscillation_34"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    AMO = atlantic_multidecadal_oscillation(data_set["sea-surface-temperature"])

    # Check, if calculated index only has one dimension: 'time'
    assert AMO.dims[0] == "time"
    assert len(AMO.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_zeromean(source_name):
    """Ensure that AMO has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    AMO = atlantic_multidecadal_oscillation(data_set["sea-surface-temperature"])

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=AMO.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_AMO_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    AMO = atlantic_multidecadal_oscillation(data_set["sea-surface-temperature"])

    assert AMO.name == "AMO"
    assert AMO.long_name == "atlantic_multidecadal_oscillation"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PREC_SAHEL_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    PREC_SAHEL = sahel_precipitation(data_set["precipitation"])

    # Check, if calculated index only has one dimension: 'time'
    assert PREC_SAHEL.dims[0] == "time"
    assert len(PREC_SAHEL.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PREC_SAHEL_zeromean(source_name):
    """Ensure that index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate Sahel precipitation anomaly index
    PREC_SAHEL = sahel_precipitation(data_set["precipitation"])

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=PREC_SAHEL.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_PREC_SAHEL_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index index
    result = sahel_precipitation(data_set["precipitation"])

    assert result.name == "PREC_SAHEL"
    assert result.long_name == "sahel_precipitation"


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_metadata(source_name):
    """Ensure that index only contains time dimension."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NP = north_pacific(data_set["sea-level-pressure"])

    # Check, if calculated index only has one dimension: 'time'
    assert NP.dims[0] == "time"
    assert len(NP.dims) == 1


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_zeromean(source_name):
    """Ensure that index has zero mean."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    NP = north_pacific(data_set["sea-level-pressure"])

    # Check, if calculated index has zero mean:
    assert_almost_equal(actual=NP.mean("time").values[()], desired=0, decimal=3)


@pytest.mark.parametrize("source_name", list(VARNAME_MAPPING.keys()))
def test_NP_naming(source_name):
    """Ensure that the index is named correctly."""
    # Load test data
    TEST_DATA_PATH = Path(__file__).parent / "../data/test_data/"
    data_set = load_data_set(data_path=TEST_DATA_PATH, data_source_name=source_name)

    # Calculate index
    result = north_pacific(data_set["sea-level-pressure"])

    assert result.name == "NP"
    assert result.long_name == "north_pacific"

