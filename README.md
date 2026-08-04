# madis-data

**madis-data** is a Python toolkit to download, process, extract, interpolate, and plot Meteorological Assimilation Data Ingest System ([MADIS](https://madis-data.ncep.noaa.gov)) data. Currently, **madis-data** operates on the following publicly available MADIS [Mesonet](https://madis.ncep.noaa.gov/madis_mesonet.shtml) data:

- 2 m temperature
- 2 m dewpoint temperature
- 10 m west-east and south-north wind speed
- Solar radiation
<!--
## Overview

## Key features
-->
## Workflow (using API)

The workflow is shown in the [HowTo](https://github.com/jankazil/madis-data/blob/main/notebooks/HowTo.ipynb) Jupyter notebook. The notebook

1. downloads MESONET surface observations data files for January 2026 from the MADIS archive,
2. extracts data for stations in the state of Colorado of the following variables:
    - 2 m temperature,
    - 2 m dewpoint temperature,
    - 10 m west-east and south-north wind speed,
    - solar radiation,
3. interpolates them to full-hourly time series,
4. plots the original and the interpolated time series at an individual station over select time intervals,
5. counts the number of stations that have valid data for at least a given fraction of the time,
6. plots the stations in the state of Colorado on a US map.

### Requirements
This notebook requires the Jupyter kernel madis-data, which is automaticaly installed as part of the madis-data conda development environment. The conda development environment can be installed from the root directory of the madis-data distribution with

```bash
make setup-dev-env
```
## Installation

### pip

```bash
pip install madis-data
```

### conda / mamba

```bash
mamba install -c jan.kazil -c conda-forge madis-data
```

<!--
## Executables

## Public API

### Modules

## Notes
-->
## Development

### Code Quality and Testing Commands

- `make fmt` - Runs `ruff format`, which reformats Python files according to the style rules in `pyproject.toml`.
- `make lint` - Runs `ruff check --fix`, which lints the code and auto-fixes what it can.
- `make check` - Runs formatting and linting.
- `make type` - Currently disabled. Intended to run `mypy` using the settings in `pyproject.toml`.
- `make test` - Runs `pytest` with the test settings configured in `pyproject.toml`.

## Author

Jan Kazil - jan.kazil.dev@gmail.com - [github.com/jankazil/madis-data](https://github.com/jankazil/madis-data)

## License

BSD-3-Clause
