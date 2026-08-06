# madis-data

**madis-data** produces analysis-ready full-hour MADIS Mesonet datasets for arbitrary date ranges and operationally meaningful U.S. regions or individual stations as netCDF files.

It downloads, processes, extracts, and interpolates to the full hour Meteorological Assimilation Data Ingest System ([MADIS](https://madis-data.ncep.noaa.gov)) data. Currently, **madis-data** operates with the following publicly available MADIS [Mesonet](https://madis.ncep.noaa.gov/madis_mesonet.shtml) data:

- 2 m temperature
- 2 m dewpoint temperature
- 10 m west-east and south-north wind speed
- solar radiation

MADIS data are provided by the [National Centers for Environmental Prediction](https://www.weather.gov/ncep/).

## Installation

### pip

```bash
pip install madis-data
```

### Conda/Mamba

```bash
mamba install -c jan.kazil -c conda-forge madis-data
```
### Jupyter

If the Conda/Mamba environment in which **madis-data** was installed is not yet available as a Jupyter kernel, install the Jupyter dependencies and register the environment as a kernel:

```bash
mamba install -c conda-forge jupyter_client jupyter_core notebook ipykernel

python -m ipykernel install --user --name "$CONDA_DEFAULT_ENV" --display-name "Python ($CONDA_DEFAULT_ENV)"
```

The kernel will then appear in Jupyter as `Python (<environment-name>)`.

## Overview

The package provides a command-line tool that:

- selects stations by geography (an individual MADIS station, a U.S. state, a Regional Transmission Organization/Independent System Operator regions, and the special region CONUS representing the contiguous United States),
- downloads Mesonet observation files for a specified date range,
- extracts, preprocesses, and saves in netCDF files the observables,
    - 2 m temperature
    - 2 m dewpoint temperature
    - 10 m west-east and south-north wind speed
    - solar radiation
- optionally removes the original Mesonet observation files after successful preprocessing to reduce storage requirements,
- constructs full-hourly time series for the supported observables,
- and saves the spatially filtered observations and hourly time series in netCDF files.

It also generates a map showing the selected station locations.

Once preprocessed files with the above variables have been generated for a given date range, constructing full-hourly time series for that date range for individual stations, U.S. states, or the CONUS region no loner requires downloading the original Mesonet observations again.

Geospatial region selection is based on U.S. Energy Information Administration definitions of Regional Transmission Organization/Independent System Operator footprints and U.S. Census Bureau state and territory boundaries included with the package.

## Command-line interface (CLI)

The CLI is exposed as `"build-mesonet-dataset"` when installed.

**Usage:**

```bash
build-mesonet-dataset START_YEAR START_MONTH START_DAY END_YEAR END_MONTH END_DAY REGION DATA_DIR \
    [-n N_JOBS] [--remove-original] [-r] [-v]
```

**Positional arguments**  

- `START_YEAR`, `START_MONTH`, and `START_DAY`: Components of the first requested timestamp at 00:00 UTC  
- `END_YEAR`, `END_MONTH`, and `END_DAY`: Components of the last requested timestamp at 00:00 UTC  
- `REGION`: Region or station selector. Use a two-letter U.S. state or territory code, `CONUS`, an RTO/ISO code, or a MADIS Mesonet station identifier  
- `DATA_DIR`: Directory where downloaded data and generated outputs will be stored  

**Options**  

- `-n, --n N_JOBS`: Number of parallel workers used for downloading, spatial extraction, and interpolation. Values greater than 1 can accelerate processing but may increase memory use or the risk of network errors.  
- `--remove-original`: Remove downloaded original MADIS files after successful preprocessing.
- `-r, --refresh`: Download files even when corresponding local files already exist.  
- `-v, --verbose`: Print detailed progress information.  

The command downloads one additional day before and after the requested interval so observations can bracket its boundary hours during interpolation. It creates a NetCDF file containing the selected record-oriented observations, a map of the selected station locations, and a NetCDF file containing the merged full-hourly UTC time series.

**Examples:**

```bash
# Show usage information:
build-mesonet-dataset --help

# Download and process MADIS Mesonet data for station SMPC2 from 00:00 UTC on January 1, 2026,
# through 00:00 UTC on January 31, 2026, and save the results in /path/to/data:
build-mesonet-dataset -v 2026 1 1 2026 1 31 SMPC2 /path/to/data

# Build datasets for Colorado using 16 parallel workers and remove source files after preprocessing:
build-mesonet-dataset -v 2026 1 1 2026 2 1 CO /path/to/data -n 16 --remove-original

```

## Workflow (using API)

The detailed workflow is documented in the [HowTo](https://github.com/jankazil/madis-data/blob/main/notebooks/HowTo.ipynb) Jupyter notebook. The notebook performs the following tasks:

1. download MESONET surface observations data files for January 2026 from the MADIS archive,
2. extract January 2026 data for stations in the state of Colorado of the following observables:
    - 2 m temperature,
    - 2 m dewpoint temperature,
    - 10 m west-east and south-north wind speed,
    - solar radiation
3. save the extracted data in a s netCDF file,
4. interpolate the data to full-hourly time series,
5. save the full-hourly time series in a netCDF file,
6. plot the original and the full-hourly interpolated time series at an individual station over select time intervals,
7. plot the stations in the state of Colorado on a US map.

### Requirements
The notebook requires a Jupyter kernel being available in the Python environment in which madis-data has been installed (Section "Jupyter").
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

## Disclaimers

The data accessed by this software are publicly available from NOAA's National Centers for Environmental Prediction (NCEP) and are subject to their terms of use. This project is not affiliated with or endorsed by NOAA.

This software uses U.S. Census Bureau and U.S. Energy Information Administration data, but is neither endorsed nor certified by the U.S. Census Bureau or the U.S. Energy Information Administration.

## Author

Jan Kazil - jan.kazil.dev@gmail.com - [github.com/jankazil/madis-data](https://github.com/jankazil/madis-data)

## License

BSD-3-Clause
