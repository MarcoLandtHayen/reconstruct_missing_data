import os

from json import dump, load
from pathlib import Path

import numpy as np
import xarray as xr

from reconstruct_missing_data.dummy_module import dummy_foo


print(os.getcwd())

# Set relative path on local machine (or remote machine):
slp_path_FOCI = "climate_index_collection/data/raw/2022-08-22/FOCI/FOCI1.3-SW038_echam6_BOT_mm_2350-3349_slp_monthly_1_midmonth.nc"
slp_path_CESM = "climate_index_collection/data/raw/2022-08-22/CESM/B1850WCN_f19g16_1000y_v3.2_mod-S15-G16.cam2.h0.0001-0999.PSL.midmonth.nc"

# Load data:
slp_FOCI = xr.open_dataset(slp_path_FOCI)
slp_CESM = xr.open_dataset(slp_path_CESM)

print(slp_FOCI.dims["lon"])
print(slp_CESM.dims["lon"])

dummy_path = Path("GitGeomar/marco-landt-hayen/reconstruct_missing_data/results")

# Create directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
os.makedirs(dummy_path, exist_ok=True)

model_config = "gpu_config"
source = "test_source"

# Store parameters as json:
parameters = {
    "model_config": model_config,
    "source": source,
}

with open(dummy_path / "parameters.json", "w") as f:
    dump(parameters, f)
