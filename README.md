# madis-data

**madis-data** produces analysis-ready full-hour [MADIS Mesonet](https://madis.ncep.noaa.gov/madis_mesonet.shtml) and [MADIS Metar](https://madis.ncep.noaa.gov/madis_metar.shtml) datasets for arbitrary date ranges and operationally meaningful U.S. regions or individual stations as netCDF files.

Currently, **madis-data** operates with the following publicly available MADIS [Mesonet](https://madis.ncep.noaa.gov/madis_mesonet.shtml) and [Metar](https://madis.ncep.noaa.gov/madis_metar.shtml) observables:

- 2 m temperature,
- 2 m dewpoint temperature,
- 10 m west-east and south-north wind speed,
- solar radiation (Mesonet only).

The Jupyter notebook [HowTo](https://github.com/jankazil/madis-data/blob/main/notebooks/HowTo.ipynb) gives an overview of the functionality **madis-data** provides.

MADIS Mesonet and Metar data are provided by the [National Centers for Environmental Prediction](https://www.weather.gov/ncep/).

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

The package provides command-line tools that:

- select stations by geography (an individual Mesonet or Metar station, a U.S. state, a Regional Transmission Organization/Independent System Operator regions, and the special region CONUS representing the contiguous United States),
- download Mesonet or Metar observation files for a specified date range,
- extract, preprocess, and save in netCDF files the observables
    - 2 m temperature,
    - 2 m dewpoint temperature,
    - 10 m west-east and south-north wind speed,
    - solar radiation (Mesonet only).
- optionally removes the original MADIS files after successful preprocessing to reduce storage use,
- constructs full-hourly time series for the supported observables,
- and saves the spatially filtered observations and hourly time series in netCDF files.

It also generates a map showing the selected station locations.

Once preprocessed files with the above variables have been generated for a given date range, constructing full-hourly time series for that date range for individual stations, U.S. states, or the CONUS region no loner requires downloading the original Mesonet or Metar observations again.

Geospatial region selection is based on U.S. Energy Information Administration definitions of Regional Transmission Organization/Independent System Operator footprints and U.S. Census Bureau state and territory boundaries included with the package.

## Workflow using command-line interface (CLI)

The CLI is exposed as `"build-mesonet-dataset"` and `"build-metar-dataset"` when installed. These executables

1. download Mesonet or Metar surface observations data files, respectively, for a given time range from the MADIS archive,
2. extract data of the following observables for stations in a given region:
    - 2 m temperature,
    - 2 m dewpoint temperature,
    - 10 m west-east and south-north wind speed,
    - solar radiation (Mesonet only).
3. save the regional data in a netCDF file,
4. interpolate the regional data to full-hourly time series,
5. save the full-hourly time series for the stations in the region in a netCDF file,
6. plot the stations in the region on a US map and save it as a PNG file.

**Usage**

```bash
build-mesonet-dataset START_YEAR START_MONTH START_DAY END_YEAR END_MONTH END_DAY REGION DATA_DIR \
    [-n N_JOBS] [--remove-original] [-r] [-v]
build-metar-dataset START_YEAR START_MONTH START_DAY END_YEAR END_MONTH END_DAY REGION DATA_DIR \
    [-n N_JOBS] [--remove-original] [-r] [-v]
```

**Positional arguments**  

- `START_YEAR`, `START_MONTH`, and `START_DAY`: Components of the first requested timestamp at 00:00 UTC  
- `END_YEAR`, `END_MONTH`, and `END_DAY`: Components of the last requested timestamp at 00:00 UTC  
- `REGION`: Region or station selector. Use a two-letter U.S. state or territory code, `CONUS`, an RTO/ISO code, or a MADIS Mesonet or Metar station identifier
- `DATA_DIR`: Directory where downloaded data and generated outputs will be stored  

**Options**  

- `-n, --n N_JOBS`: Number of parallel workers used for downloading, spatial extraction, and interpolation. Values greater than 1 can accelerate processing but may increase memory use or the risk of network errors.  
- `--remove-original`: Remove downloaded original MADIS files after successful preprocessing.
- `-r, --refresh`: Download files even when corresponding local files already exist.  
- `-v, --verbose`: Print detailed progress information.  

The command downloads one additional day before and after the requested interval so observations can bracket its boundary hours during interpolation. It creates a NetCDF file containing the selected record-oriented observations, a map of the selected station locations, and a NetCDF file containing the merged full-hourly UTC time series.

**Examples**

```bash
# Show usage information:
build-mesonet-dataset --help

# Download and process MADIS Mesonet data for station SMPC2 from 00:00 UTC on January 1, 2026,
# through 00:00 UTC on January 31, 2026, and save the results in /path/to/data:
build-mesonet-dataset -v 2026 1 1 2026 1 31 SMPC2 /path/to/data

# Build MADIS Metar datasets for Colorado using 16 parallel workers and remove source files after preprocessing:
build-metar-dataset -v 2026 1 1 2026 2 1 CO /path/to/data -n 16 --remove-original

```

**Example scripts**

The Bash scripts

```bash
./scripts/Download_MADIS_Mesonet.CO.2026-01.sh
./scripts/Download_MADIS_Metar.CO.2026-01.sh
```

can be executed from within the directory ./scripts. They download MADIS Mesonet or Metar surface observation data files from the MADIS archive, respectively, and build full-hourly datasets for Colorado for January 2026, in the directory ./data.

**Requirements**  

The **madis-data** package must be installed in the Python environment in which the scripts are called.

## Workflow using API

The **madis-data** workflow is documented in the [HowTo](https://github.com/jankazil/madis-data/blob/main/notebooks/HowTo.ipynb) Jupyter notebook. The notebook performs the following tasks:

1. download Metar surface observations data files for January 2026 from the MADIS archive,
2. extract January 2026 data for stations in the state of Colorado of the following observables:
    - 2 m temperature,
    - 2 m dewpoint temperature,
    - 10 m west-east and south-north wind speed,
3. save the extracted data in a s netCDF file,
4. interpolate the data to full-hourly time series,
5. save the full-hourly time series in a netCDF file,
6. plot the original and the full-hourly interpolated time series at an individual station over select time intervals,
7. plot the stations in the state of Colorado on a US map.

The analogous workflow applies to MADIS Metar data.

**Requirements**  

The notebook requires a Jupyter kernel being available in the Python environment in which madis-data has been installed (Section "Jupyter").
<!--
## Executables

## Public API

### Modules

## Notes
-->
## Notes

MADIS Mesonet data files represent > 30000 stations. Downloading the hourly files for extended periods of time can take hours to days. Creating full-hourly time series requires sufficient computing power and RAM. E.g., creating the full-hourly time series for stations in Texas, for the period 2021-2025, takes on a machine with 128 GB RAM and 8 cores approximately 4 h. Downloading and processing MADIS Metar data files takes significantly less time owing to the lower number of Metar stations.


## Development

### Code Quality and Testing Commands

- `make fmt` - Runs `ruff format`, which reformats Python files according to the style rules in `pyproject.toml`.
- `make lint` - Runs `ruff check --fix`, which lints the code and auto-fixes what it can.
- `make check` - Runs formatting and linting.
- `make type` - Currently disabled. Intended to run `mypy` using the settings in `pyproject.toml`.
- `make test` - Runs `pytest` with the test settings configured in `pyproject.toml`.

## Disclaimers

This software uses publicly available NOAA National Centers for Environmental Prediction (NCEP) data that are subject to their terms of use, but is neither affiliated, endorsed, or certified by NOAA or NCEP.

This software uses publicly available U.S. Census Bureau data that are subject to their terms of use, but is neither affiliated, endorsed, or certified by the U.S. Census Bureau.

This software uses publicly available U.S. Energy Information Administration data that are subject to their terms of use, but is neither affiliated, endorsed, or certified by the U.S. Energy Information Administration.

This software uses publicly available University Corporation for Atmospheric Research (UCAR) data that are subject to their terms of use, but is neither affiliated, endorsed, or certified by UCAR.

## Author

Jan Kazil - jan.kazil.dev@gmail.com - [github.com/jankazil/madis-data](https://github.com/jankazil/madis-data)

## License

BSD-3-Clause
