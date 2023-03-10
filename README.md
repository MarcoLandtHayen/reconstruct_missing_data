reconstruct_missing_data
==============================
[![Build Status](https://github.com/MarcoLandtHayen/reconstruct_missing_data/workflows/Tests/badge.svg)](https://github.com/MarcoLandtHayen/reconstruct_missing_data/actions)
[![codecov](https://codecov.io/gh/MarcoLandtHayen/reconstruct_missing_data/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoLandtHayen/reconstruct_missing_data)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![Docker Image Version (latest by date)](https://img.shields.io/docker/v/mlandthayen/reconstruct_missing_data?label=DockerHub)](https://hub.docker.com/r/mlandthayen/reconstruct_missing_data/tags)


Reconstruct complete two-dimensional geospatial data from sparse inputs.
As data, we use the output of control simulations from two Earth System Models (ESMs):
The Flexible Ocean and Climate Infrastructure (FOCI) and the Whole Atmosphere Community Climate Model as extension of the Community Earth System Model (CESM) are both coupled, global climate models that provide state-of-the-art computer simulations of the past, present and future states of the Earth system. Here, we use the output of FOCI and CESM control simulations over 1,000 and 999 years, respectively. In particular, we work with the following two-dimensional fields on latitude-longitude grids:

- sea surface temperature (SST)
- surface air temperature (SAT)
- sea level pressure (SLP)
- geopotential height at pressure level 500 mbar (Z500)
- sea surface salinity (SSS)
- precipitation (PREC)

## Development

For now, we're developing in a Docker container with JupyterLab environment, Tensorflow and several extensions, based on martinclaus/py-da-stack.

To start a JupyterLab within this container, run
```shell
$ docker pull mlandthayen/py-da-tf:shap
$ docker run -p 8888:8888 --rm -it -v $PWD:/work -w /work mlandthayen/py-da-tf:shap jupyter lab --ip=0.0.0.0
```
and open the URL starting on `http://127.0.0.1...`.

Then, open a Terminal within JupyterLab and run
```shell
$ python -m pip install -e .
```
to have a local editable installation of the package.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
