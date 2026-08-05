'''
Download, process, extract, interpolate, and plot Meteorological Assimilation
Data Ingest System (MADIS) Mesonet data.
'''

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import (
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeFormatter,
    LongitudeLocator,
)
from dateutil import rrule

from madis_data.web import download_threaded
from madis_data.xarray_tools import get_missing_value, mask_with_missing_value


def mesonet_variables() -> tuple[list[str], list[str], list[str], list[str]]:
    '''
    Return the MADIS Mesonet variables to operate on.

    Returns
    -------
    tuple[list[str], list[str], list[str], list[str]]
        Variables without quality-control flags, variables with flags,
        quality-control flag variables, and preprocessed output variables.
    '''

    # List variables that do not have a quality-control flag.
    mm_vars_no_qc = ['stationId', 'latitude', 'longitude', 'elevation', 'solarRadiation']

    # List variables that have a quality-control flag.
    mm_vars_with_qc = ['dewpoint', 'temperature', 'windSpeed', 'windDir']

    # Construct the corresponding quality-control variable names.
    mm_vars_qc = [mm_var + 'DD' for mm_var in mm_vars_with_qc]

    # List the variables retained in preprocessed files.
    mm_vars_pp = [
        'time',
        'stationId',
        'latitude',
        'longitude',
        'elevation',
        'temperature',
        'dewpoint',
        'U10',
        'V10',
        'solarRadiation',
    ]

    return mm_vars_no_qc, mm_vars_with_qc, mm_vars_qc, mm_vars_pp


def mesonet_urls(start_date: datetime, end_date: datetime) -> list[str]:
    '''
    Construct hourly MADIS Mesonet archive URLs for an inclusive date range.

    Parameters
    ----------
    start_date : datetime
        First date/time to include.
    end_date : datetime
        Last date/time to include.

    Returns
    -------
    list[str]
        URLs of the hourly compressed NetCDF files.
    '''

    # Generate every hourly timestamp within the requested date range.
    dt = rrule.HOURLY
    dates = list(rrule.rrule(dt, dtstart=start_date, until=end_date))

    # Convert each timestamp to its MADIS archive URL.
    urls = [
        (
            'https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/'
            f'{date.year}/{date.month:02d}/{date.day:02d}/LDAD/mesonet/'
            f'netCDF/{date.year}{date.month:02d}{date.day:02d}_'
            f'{date.hour:02d}00.gz'
        )
        for date in dates
    ]

    return urls


def mesonet_local_file_paths(data_dir: Path, urls: list[str]) -> list[Path]:
    '''
    Construct local paths for MADIS Mesonet URLs.

    Parameters
    ----------
    data_dir : Path
        Directory in which the files are or will be stored.
    urls : list[str]
        MADIS Mesonet file URLs.

    Returns
    -------
    list[Path]
        Local paths formed from the URL filenames.
    '''

    # Attach each URL filename to the local data directory.
    file_paths = [data_dir / PurePosixPath(urlparse(file_url).path).name for file_url in urls]

    return file_paths


def mesonet_local_file_paths_preprocessed(file_paths: list[Path]) -> list[Path]:
    '''
    Construct paths for the preprocessed versions of MADIS files.

    Parameters
    ----------
    file_paths : list[Path]
        Paths of original MADIS Mesonet files.

    Returns
    -------
    list[Path]
        Corresponding paths with the suffix changed to ``.nc``.
    '''

    # Replace the compressed-file suffix with the NetCDF suffix.
    file_paths = [file_path.with_suffix('.nc') for file_path in file_paths]

    return file_paths


def mesonet_station_ids2strings(station_id: xr.DataArray) -> np.ndarray:
    '''
    Decode MADIS station identifiers into a one-dimensional string array.

    Parameters
    ----------
    station_id : xr.DataArray
        One- or two-dimensional MADIS station-identifier array.

    Returns
    -------
    np.ndarray
        Decoded identifiers with null characters and padding removed.

    Raises
    ------
    ValueError
        If the input has neither one nor two dimensions.
    '''

    # Use the fast fixed-width byte-string path when possible.
    encoded_ids = _pack_mesonet_station_ids(station_id)

    if encoded_ids is not None:
        # Cleaning before Unicode conversion is substantially faster for the
        # normal ASCII station IDs. Preserve the former replacement behavior
        # for unexpected non-ASCII or invalid UTF-8 bytes.
        try:
            return encoded_ids.astype(str)
        except UnicodeDecodeError:
            # Replace malformed UTF-8 instead of stopping extraction.
            decoded_ids = np.char.decode(encoded_ids, encoding='utf-8', errors='replace')
            return np.char.strip(np.char.replace(decoded_ids, '\x00', ''))

    # Handle Unicode, object, or numeric representations.
    values = np.asarray(station_id.values)

    # Return one empty identifier per record for an empty character dimension.
    if values.ndim == 2 and values.shape[1] == 0:
        return np.full(values.shape[0], '', dtype='U1')

    if values.dtype.kind == 'U':
        if values.ndim == 2:
            # Join each fixed-width Unicode row without a Python loop.
            values = np.ascontiguousarray(values)
            character_width = values.shape[1] * values.dtype.itemsize // np.dtype('U1').itemsize
            byte_order = values.dtype.str[0]
            decoded_ids = values.view(f'{byte_order}U{character_width}')
            decoded_ids = decoded_ids.reshape(values.shape[0])
        else:
            decoded_ids = values

        return np.char.strip(np.char.replace(decoded_ids, '\x00', ''))

    # Retain a general fallback for uncommon object or numeric representations.
    if values.ndim == 1:
        decoded_ids = np.asarray([_value2string(value) for value in values])
    else:
        decoded_ids = np.asarray([''.join(_value2string(value) for value in row) for row in values])

    return np.char.strip(np.char.replace(decoded_ids, '\x00', ''))


def download_mesonet(
    start_date: datetime,
    end_date: datetime,
    data_dir: Path,
    preprocess: bool = True,
    refresh: bool = False,
    remove_original: bool = False,
    verbose: bool = False,
    n_jobs: int = 8,
) -> list[Path]:
    '''
    Download and optionally preprocess hourly MADIS Mesonet files.

    Existing valid output files are reused unless ``refresh`` is True. When
    preprocessing is enabled, selected observations are quality controlled,
    converted to the output variables, and saved as NetCDF files.

    Parameters
    ----------
    start_date : datetime
        First date to download.
    end_date : datetime
        Last date to download.
    data_dir : Path
        Directory in which files are stored.
    preprocess : bool, optional
        If True, preprocess downloaded files. The default is True.
    refresh : bool, optional
        If True, download files even when local copies exist. The default is
        False.
    remove_original : bool, optional
        If True, remove downloaded source files after preprocessing. The
        default is False.
    verbose : bool, optional
        Passed to the downloader to control messages. The default is False.
    n_jobs : int, optional
        Maximum number of simultaneous downloads or preprocessing workers.
        The default is 8.

    Returns
    -------
    list[Path]
        Sorted paths of downloaded or preprocessed files, including reused
        files.

    Raises
    ------
    ValueError
        If ``n_jobs`` is invalid or processing returns the wrong number of
        datasets.
    '''

    # Require at least one worker.
    if n_jobs < 1 or not np.isfinite(n_jobs):
        raise ValueError('n_jobs must be a positive finite integer.')

    # Create the data directory if needed.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Construct the archive URLs for the requested dates.
    file_urls = mesonet_urls(start_date, end_date)

    # Collect files that can be reused instead of downloaded.
    skip_urls = []
    skip_files = []
    skip_files_preprocessed = []

    if not refresh:
        # Construct the expected local source paths.
        file_paths = mesonet_local_file_paths(data_dir, file_urls)

        if preprocess:
            # Construct the expected preprocessed paths.
            file_paths_preprocessed = mesonet_local_file_paths_preprocessed(file_paths)

            # Obtain the variables required in a complete preprocessed file.
            _, _, _, mm_vars_pp = mesonet_variables()

            for file_url, file_path_preprocessed in zip(file_urls, file_paths_preprocessed, strict=True):
                if file_path_preprocessed.is_file():
                    ds = xr.open_dataset(file_path_preprocessed)
                    try:
                        # Reuse the file only when all required variables exist.
                        all_present = all(var in ds.data_vars for var in mm_vars_pp)
                        if all_present:
                            skip_urls.append(file_url)
                    finally:
                        # Release the file after validation succeeds or fails.
                        ds.close()
        else:
            # Reuse existing source files when preprocessing is disabled.
            for file_url, file_path in zip(file_urls, file_paths, strict=True):
                if file_path.is_file():
                    skip_urls.append(file_url)

        # Remove reusable URLs from the download list.
        skip_urls = list(set(skip_urls))
        skip_files = mesonet_local_file_paths(data_dir, skip_urls)
        skip_files_preprocessed = mesonet_local_file_paths_preprocessed(skip_files)

        file_urls = [url for url in file_urls if url not in skip_urls]

    # Collect the paths produced during this call.
    downloaded_files = []

    # Print initial progress information.
    print(f'\rProgress: {0:.2f} %', end='', flush=True)

    for ii in range(0, len(file_urls), n_jobs):
        # Process one worker-sized batch of URLs.
        file_urls_short_list = file_urls[ii : ii + n_jobs]

        # Download one batch of Mesonet files.
        file_paths_short_list = download_threaded(
            file_urls_short_list, data_dir, refresh=refresh, verbose=verbose, n_jobs=n_jobs
        )

        # Preprocess the downloaded files when requested.
        if preprocess:
            file_paths_preprocessed_short_list = mesonet_local_file_paths_preprocessed(file_paths_short_list)

            dss = extract_mesonet_data(file_paths_short_list, n_jobs=n_jobs)
            try:
                if len(dss) < len(file_paths_short_list):
                    raise ValueError(
                        'The number of downloaded MADIS Mesonet files is greater than the '
                        'number of resulting processed datasets.'
                    )
                if len(dss) > len(file_paths_short_list):
                    raise ValueError(
                        'The number of downloaded MADIS Mesonet files is smaller than the '
                        'number of resulting processed datasets.'
                    )

                for ds, file_path_preprocessed, file_path in zip(
                    dss, file_paths_preprocessed_short_list, file_paths_short_list, strict=True
                ):
                    # Save the preprocessed dataset.
                    ds.to_netcdf(file_path_preprocessed)

                    # Record the preprocessed output path.
                    downloaded_files.append(file_path_preprocessed)
                    if remove_original:
                        # Remove the source file and any accompanying ETAG.
                        file_path.unlink(missing_ok=True)
                        Path(f'{file_path}.etag').unlink(missing_ok=True)
            finally:
                # Release every processed dataset if validation or writing fails.
                for ds in dss:
                    ds.close()

        else:
            # Record the downloaded source paths.
            downloaded_files += file_paths_short_list

        # Print progress after completing the batch.
        progress = 100 * (ii + n_jobs) / len(file_urls)
        print(f'\rProgress: {progress:.2f} %', end='', flush=True)

    # End the progress line at 100 percent.
    print(f'\rProgress: {100:.2f} %')

    # Combine reused files with files produced during this call.
    if preprocess:
        result_files = skip_files_preprocessed + downloaded_files
    else:
        result_files = skip_files + downloaded_files

    return sorted(result_files)


def extract_mesonet_data(file_paths: list[Path], n_jobs: int = 1) -> list[xr.Dataset]:
    '''
    Preprocess multiple MADIS Mesonet files in parallel.

    Parameters
    ----------
    file_paths : list[Path]
        Paths of MADIS Mesonet source files.
    n_jobs : int, optional
        Number of worker threads. The default is 1.

    Returns
    -------
    list[xr.Dataset]
        One preprocessed dataset for each input path, in input order.
    '''

    # Collect datasets as workers complete in input order.
    dss = []
    completed = False
    try:
        # Process files concurrently while preserving their order.
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for ds in executor.map(extract_mesonet_data_single, file_paths):
                dss.append(ds)
        completed = True
    finally:
        if not completed:
            # Release datasets returned before another worker failed.
            for ds in dss:
                ds.close()

    return dss


def extract_mesonet_data_single(file_path: Path) -> xr.Dataset:
    '''
    Preprocess and quality-control one MADIS Mesonet file.

    Parameters
    ----------
    file_path : Path
        Path of a MADIS Mesonet source file.

    Returns
    -------
    xr.Dataset
        Dataset containing retained observations, Cartesian wind components,
        processing metadata, and the source-file path.
    '''

    # Obtain the variable groups used for extraction and quality control.
    mm_vars_no_qc, mm_vars_with_qc, mm_vars_qc, _ = mesonet_variables()

    # Read only the required source variables and standardize missing values.
    selected_variables = mm_vars_no_qc + mm_vars_with_qc + mm_vars_qc
    ds = preprocess_mesonet_file(file_path, selected_variables)

    # Retain values flagged as screened or verified.
    for varname, varname_qc in zip(mm_vars_with_qc, mm_vars_qc, strict=True):
        da_var = ds[varname]
        da_qc = ds[varname_qc]

        # Obtain the missing value used by this variable.
        missing_value = get_missing_value(da_var)

        # Identify observations that passed quality control.
        qc_mask = (da_qc == np.bytes_('S')) | (da_qc == np.bytes_('V'))

        # Replace rejected observations with the declared missing value.
        ds[varname] = mask_with_missing_value(da_var, qc_mask, missing_value)

    # Reject wind speeds outside the permitted range of 0 to 50 m s-1.
    varname_ws = 'windSpeed'
    varname_wd = 'windDir'

    if varname_ws in ds.data_vars:
        da_var = ds[varname_ws]
        missing_value = get_missing_value(da_var)
        qc_mask = (da_var >= 0.0) & (da_var <= 50.0)
        ds[varname_ws] = mask_with_missing_value(da_var, qc_mask, missing_value)

        if varname_wd in ds.data_vars:
            # Reject wind directions associated with invalid wind speeds.
            da_var = ds[varname_wd]
            missing_value = get_missing_value(da_var)
            ds[varname_wd] = mask_with_missing_value(da_var, qc_mask, missing_value)

    # Convert wind speed and direction to Cartesian wind components.
    if varname_ws in ds.data_vars and varname_wd in ds.data_vars:
        ws = ds[varname_ws]
        wd = ds[varname_wd]

        # Mark records for which both wind values are finite.
        mask = np.isfinite(ws) & np.isfinite(wd)

        # Initialize the wind components with missing values.
        ds['U10'] = ('recNum', np.full(ds.sizes['recNum'], np.nan, dtype=np.float32))
        ds['V10'] = ('recNum', np.full(ds.sizes['recNum'], np.nan, dtype=np.float32))

        # Convert meteorological direction from degrees to radians.
        theta = np.deg2rad(wd)

        # Calculate west-east and south-north wind components.
        ds['U10'] = xr.where(mask, -ws * np.sin(theta), ds['U10'], keep_attrs=False).astype(np.float32)
        ds['V10'] = xr.where(mask, -ws * np.cos(theta), ds['V10'], keep_attrs=False).astype(np.float32)

        # Describe the calculated wind components.
        ds['U10'].attrs['long_name'] = 'West-east wind speed'
        ds['U10'].attrs['units'] = ws.attrs['units']
        ds['U10'].attrs['missing_value'] = np.nan
        ds['U10'].encoding['_FillValue'] = np.nan

        ds['V10'].attrs['long_name'] = 'South-north wind speed'
        ds['V10'].attrs['units'] = ws.attrs['units']
        ds['V10'].attrs['missing_value'] = np.nan
        ds['V10'].encoding['_FillValue'] = np.nan

        # Remove the source wind speed and direction variables.
        ds = ds.drop_vars([varname_ws, varname_wd])

    # Remove quality-control flag variables from the output.
    ds = ds.drop_vars(mm_vars_qc)

    # Record the processing applied to the observations.
    ds.attrs['postprocessing'] = (
        'For variables that have a quality control information, only values '
        'with '
        'the flags S (screened) or V (verified) were retained. '
        'If wind speed magnitude and wind direction variables are present in '
        'the original file, their values were removed where the wind speed '
        'magnitude was < 0 or > 50 m s-1. Then the U and V wind speeds were '
        'calculated, and the wind speed magnitude and direction variables were removed.'
    )

    # Record the source-file path.
    ds.attrs['file_orig'] = str(file_path)

    return ds


def preprocess_mesonet_file(file_path: Path, varnames: str | list[str]) -> xr.Dataset:
    '''
    Read selected variables from a MADIS Mesonet file.

    Declared missing values are replaced with NaN. The ``reportTime`` variable
    is renamed ``time`` and configured for NetCDF output as seconds since the
    Unix epoch.

    Parameters
    ----------
    file_path : Path
        Path of a compressed or uncompressed MADIS NetCDF file.
    varnames : str or list[str]
        Data variables to retain. ``reportTime`` is included automatically.

    Returns
    -------
    xr.Dataset
        Dataset containing the selected variables and standardized missing
        values and time encoding.
    '''
    # Convert a single variable name to a list.
    if isinstance(varnames, str):
        varnames = [varnames]

    # Always retain report time and remove duplicate variable names.
    varnames = ['reportTime'] + varnames
    varnames = list(dict.fromkeys(varnames))

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        '''
        Select the required variables from one opened dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset opened by xarray.

        Returns
        -------
        xr.Dataset
            Dataset containing only the required variables.

        Raises
        ------
        ValueError
            If any required variable is absent.
        '''
        # Identify required variables that are absent from the file.
        missing = [name for name in varnames if name not in ds.data_vars]

        if missing:
            source = ds.encoding.get('source')
            raise ValueError(f'Missing data variables: {missing} in {source}')

        # Return only the requested variables.
        return ds[varnames]

    # Suppress warnings raised while opening the MADIS file collection.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        ds = xr.open_mfdataset(file_path, preprocess=preprocess)

        # Rename report time to the standard output name.
        ds = ds.rename({'reportTime': 'time'})

        for name in ds.data_vars:
            # The time variable receives its own explicit NetCDF encoding below.
            if name == 'time':
                continue

            variable = ds[name]
            fill_values = []

            # Find fill-value declarations in both encoding and attributes.
            for metadata in (variable.encoding, variable.attrs):
                for key in ('_FillValue', 'missing_value'):
                    value = metadata.get(key)

                    if value is not None:
                        fill_values.extend(np.asarray(value).reshape(-1).tolist())

            # Existing NaNs do not need to be replaced.
            valid_fill_values = []

            for value in fill_values:
                try:
                    if np.isnan(value):
                        continue
                except TypeError:
                    pass

                if value not in valid_fill_values:
                    valid_fill_values.append(value)

            # Replace declared missing values with NaN.
            if valid_fill_values:
                attrs = variable.attrs.copy()
                encoding = variable.encoding.copy()

                variable = variable.where(~variable.isin(valid_fill_values))

                variable.attrs = attrs
                variable.encoding = encoding
                ds[name] = variable

            # Remove conflicting missing-data metadata.
            ds[name].attrs.pop('_FillValue', None)
            ds[name].attrs.pop('missing_value', None)
            ds[name].encoding.pop('_FillValue', None)
            ds[name].encoding.pop('missing_value', None)

            # Write floating-point missing values as NaN.
            if np.issubdtype(ds[name].dtype, np.floating):
                ds[name].encoding['_FillValue'] = np.nan
                ds[name].attrs['missing_value'] = np.nan

        # Remove the original time metadata and specify exactly how the
        # datetime values should be encoded when written to NetCDF.
        ds['time'].attrs = {}
        ds['time'].encoding = {
            'dtype': 'float64',
            '_FillValue': np.nan,
            'units': 'seconds since 1970-01-01T00:00:00+00:00',
            'calendar': 'proleptic_gregorian',
        }

    return ds


def extract_station_data(
    ds: xr.Dataset,
    station_ids: list[str] | None = None,
    show_progress: bool = True,
) -> dict[str, xr.Dataset]:
    '''
    Efficiently extract time-dependent datasets for requested MADIS stations.

    Fixed-width station IDs remain encoded during selection and are replaced
    by compact integer station codes. pandas then groups the selected table
    once by ``(stationCode, time)`` and retains the first nonmissing value of
    every variable independently.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset produced by ``filter_by_region``
    station_ids : list[str], optional
        List of stations for which data to extract. These must be a
        subset of the stations in ``ds``. If not provided, data for
        all stations in ``ds`` will be extracted.
    show_progress : bool, optional
        Whether to display phase-based extraction progress. The default is
        True.

    Returns
    -------
    dict[str, xr.Dataset]
        Dataset for each requested station, keyed by station identifier. Time
        and station are the dimensions. Station identifiers, latitudes,
        longitudes, and elevations are functions of the singleton station
        dimension.

    Raises
    ------
    TypeError
        If ``station_ids`` is not a list of strings.
    KeyError
        If a requested station is absent.
    ValueError
        If required variables, dimensions, or valid times are absent.

    Notes
    -----
    The input dataset remains open and owned by the caller.
    '''

    # Validate the station-list type and its contents.
    if station_ids is not None:
        if not isinstance(station_ids, list):
            raise TypeError('station_ids must be a list.')

        if not all(isinstance(station_id, str) for station_id in station_ids):
            raise TypeError('Every station ID must be a string.')

    def report_progress(task: str) -> None:
        '''Display progress update.'''
        if show_progress:
            ending = '\n'
            print(f'Completed {task}', end=ending, flush=True)

    if show_progress:
        print('Extracting data by station ...')

    # Confirm that the required record and station-table variables exist.
    required_variables = {'stationId', 'stationID', 'time'}
    missing_variables = sorted(required_variables.difference(ds.variables))
    if missing_variables:
        raise ValueError(f'ds is missing required variables: {missing_variables}')

    if ds['time'].ndim != 1:
        raise ValueError('The time variable must have one dimension.')

    # Require station identifiers and times to describe the same records.
    if ds['stationId'].dims != ds['time'].dims:
        raise ValueError('stationId and time must use the same record dimension.')

    # Require one station-table dimension for station metadata lookup.
    if ds['stationID'].ndim != 1:
        raise ValueError('The stationID variable must have one dimension.')

    if station_ids is None:
        # If no station IDs were provided, extract them from the station table.
        station_ids = [str(value) for value in ds['stationID'].values]

    # Load the complete source dataset into memory without closing the caller's dataset.
    ds.load()

    # Report completion of source loading.
    report_progress('loading data to memory - moving on to preparation of station identifiers ...')

    # Obtain the record dimension shared by time-dependent variables.
    record_dimension = ds['time'].dims[0]

    # Preserve request order while avoiding repeated work for duplicate IDs.
    requested_station_ids = list(dict.fromkeys(station_ids))

    # Prepare compact station identifiers for fast matching.
    station_id_keys = _prepare_mesonet_station_id_keys(ds['stationId'], requested_station_ids)

    if station_id_keys.shape != (ds.sizes[record_dimension],):
        raise ValueError(f'Station ID keys do not correspond one-to-one with the {record_dimension} dimension.')

    # Report completion of station-identifier preparation.
    report_progress('preparation of station identifiers - moving on to selection of data at requested stations ...')

    if requested_station_ids:
        # Find every requested record and assign its compact station code.
        record_positions, station_codes = _select_mesonet_station_records(station_id_keys, requested_station_ids)
        del station_id_keys

        # Report completion of requested-record selection.
        report_progress(
            'selection of data at requested stations - moving on to assembling valid data at requested stations ...'
        )

        # Assemble one table containing all requested, valid-time records.
        station_frame, variable_names = _build_bulk_station_frame(
            ds, station_codes, requested_station_ids, record_positions, record_dimension
        )

        # Combine duplicate times, retaining the first non-missing values.
        grouped_frame = station_frame.groupby(['stationCode', 'time'], as_index=False, sort=True, observed=True).first()

        # Report completion of table construction and time grouping.
        report_progress(
            'assembling valid data at requested stations - moving on to building datasets for requested stations ...'
        )

        # Convert each grouped station table back to an xarray dataset.
        ds_dictionary = _bulk_frame_to_station_datasets(grouped_frame, ds, requested_station_ids, variable_names)
    else:
        ds_dictionary = {}

    # Finish the interactive progress line.
    report_progress('building datasets for requested stations')

    return ds_dictionary


def interpolate_to_full_hour(
    ds_dict: dict[str, xr.Dataset],
    start_date: datetime | pd.Timestamp | None = None,
    end_date: datetime | pd.Timestamp | None = None,
    max_interpolation_interval_h: float = 2,
    verbose: bool = False,
    max_workers: int | None = None,
) -> dict[str, xr.Dataset]:
    '''
    Interpolate station datasets to full hours while preserving station order.

    Stations without an available output hour are retained as all-missing
    datasets on the first nonempty output time coordinate, when one exists.
    '''

    # Preserve the input station order for deterministic output.
    stations = list(ds_dict.items())
    n_stations = len(stations)

    print('\rProgress: 0.00 %', end='', flush=True)

    # Return an empty result without constructing an invalid process pool.
    if not stations:
        print(f'\rProgress: {100:.2f} %')
        return {}

    # Determine the first and last full hours within the time spans of
    # all nonempty station datasets.
    if start_date is None and end_date is None:
        earliest_time = None
        latest_time = None

        for _, ds_station in stations:
            if ds_station.sizes.get('time', 0) == 0:
                continue

            time = ds_station.indexes['time']
            first_time = pd.Timestamp(time[0])
            last_time = pd.Timestamp(time[-1])

            if first_time.tz is None:
                first_time = first_time.tz_localize('UTC')
                last_time = last_time.tz_localize('UTC')
            else:
                first_time = first_time.tz_convert('UTC')
                last_time = last_time.tz_convert('UTC')

            if earliest_time is None or first_time < earliest_time:
                earliest_time = first_time

            if latest_time is None or last_time > latest_time:
                latest_time = last_time

        if earliest_time is not None:
            first_hour = earliest_time.ceil('h')
            last_hour = latest_time.floor('h')

            if first_hour <= last_hour:
                start_date = first_hour
                end_date = last_hour

    if max_workers is None:
        max_workers = min(n_stations, os.cpu_count() or 1)

    # Load station names once before distributing interpolation across processes.
    station_names = _mesonet_station_names()

    # Store completed results independently of process completion order.
    completed_datasets: dict[str, xr.Dataset] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                interpolate_to_full_hour_single,
                ds_station,
                start_date=start_date,
                end_date=end_date,
                max_interpolation_interval_h=max_interpolation_interval_h,
                verbose=verbose,
                station_name=station_names.get(station_id, station_id),
            ): station_id
            for station_id, ds_station in stations
        }

        for ii, future in enumerate(as_completed(futures), start=1):
            station_id = futures[future]
            completed_datasets[station_id] = future.result()

            progress = 100 * ii / n_stations
            print(f'\rProgress: {progress:.2f} %', end='', flush=True)

    print()

    # Restore the input order after completion-order collection.
    ds_hourly_dict = {station_id: completed_datasets[station_id] for station_id, _ in stations}

    # Find a nonempty time coordinate for stations skipped without explicit bounds.
    common_time = next(
        (ds_hourly['time'] for ds_hourly in ds_hourly_dict.values() if ds_hourly.sizes.get('time', 0) > 0),
        None,
    )

    if common_time is not None:
        for station_id, ds_hourly in ds_hourly_dict.items():
            if ds_hourly.sizes.get('time', 0) == 0:
                # Represent the skipped station on the common time coordinate.
                ds_hourly = ds_hourly.reindex(time=common_time.values)
                ds_hourly['time'].encoding = common_time.encoding.copy()
                ds_hourly_dict[station_id] = ds_hourly

    return ds_hourly_dict


def interpolate_to_full_hour_single(
    ds: xr.Dataset,
    start_date: datetime | pd.Timestamp | None = None,
    end_date: datetime | pd.Timestamp | None = None,
    max_interpolation_interval_h: float = 2,
    verbose: bool = False,
    station_name: str | None = None,
) -> xr.Dataset:
    '''
    Interpolate numeric station observables to a regular full-hourly
    Coordinated Universal Time (UTC) grid using nearest finite neighbors and
    linear interpolation.

    The procedure:

      1. Validate that the input time coordinate is nonempty, unique, and
         strictly increasing.
      2. Interpret timezone-naive input times as UTC and convert timezone-aware
         input times to UTC.
      3. Build an hourly time coordinate from the first full hour on or after
         the first observation to the last full hour on or before the final
         observation. If ``start_date`` and ``end_date`` are provided, build
         the hourly time coordinate from the first full hour on or before
         ``start_date`` to the last full hour on or before ``end_date``.
      4. For each eligible numeric variable and target hour, use an exact
         finite full-hour observation when available. Otherwise, search for
         the nearest finite observations on the left and right whose total
         temporal separation does not exceed
         ``max_interpolation_interval_h``, and linearly interpolate between
         them.
      5. Package the hourly series and station metadata into an xarray
         Dataset.

    This function is suitable for instantaneous observables, such as
    temperature, dew point, and wind speed. It must not be used without
    additional treatment for accumulated or interval-total quantities, such
    as precipitation totals.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for one station. It must contain a ``time`` coordinate and the
        station metadata variables ``stationID``, ``latitude``,
        ``longitude``, and ``elevation``. Numeric data variables whose
        dimensions are exactly ``('time',)`` are interpolated; the station
        metadata variables are excluded.
    start_date : datetime or pd.Timestamp, optional
        Start of the requested output period. It is used only when
        ``end_date`` is also supplied. The default is None.
    end_date : datetime or pd.Timestamp, optional
        End of the requested output period. It is used only when
        ``start_date`` is also supplied. The default is None.
    max_interpolation_interval_h : float, optional
        Maximum permitted temporal separation, in hours, between the finite
        observations selected on the left and right of a target hour. The
        default is ``2``.
    verbose : bool, optional
        If True, print progress and skip information. The default is False.
    station_name : str, optional
        Station name. If omitted, it is retrieved from the MADIS station table.

    Returns
    -------
    xr.Dataset
        Dataset containing:

          - A regular full-hourly ``time`` coordinate with UTC implicit.
          - A singleton ``station`` coordinate.
          - Each interpolated observable with dimensions
            ``('time', 'station')`` and values stored as ``float32``.
          - Station name, latitude, longitude, and elevation variables.
          - Global attributes describing the source and processing.

        If the input contains no usable output hour, returns the same station
        metadata and variables on an empty or requested all-missing time grid.

    Raises
    ------
    ValueError
        If the time coordinate contains duplicate timestamps.
    ValueError
        If the time coordinate is not monotonically increasing.
    KeyError
        If a required station attribute or metadata variable is absent.

    Notes
    -----
    - Positive and negative infinity are treated as invalid values, like
      ``NaN``.
    - If a finite observation falls exactly on a target hour, it is copied
      without interpolation.
    - A target is interpolated only when finite observations are available on
      both sides and their total temporal separation is no greater than
      ``max_interpolation_interval_h``.
    - The output ``time`` coordinate is timezone-naive ``datetime64[ns]``;
      UTC is implicit and is specified in its encoding.

    Examples
    --------
    Interpolate all eligible observables in a single-station dataset:

        ds_hourly = interpolate_to_full_hour_single(
            ds,max_interpolation_interval_h=2.0,verbose=False
        )
    '''

    if (start_date is not None) and (end_date is None):
        raise ValueError('When providing start_date, end_date must be provided as well.')

    if (start_date is None) and (end_date is not None):
        raise ValueError('When providing end_date, start_date must be provided as well.')

    # Read the singleton station identifier stored by the extraction routines.
    station_id_values = np.asarray(ds['stationID'].values).reshape(-1)
    if station_id_values.size != 1:
        raise ValueError('The stationID variable must contain one value.')
    station_id = str(station_id_values[0])

    # Retrieve the station name from the MADIS station table.
    if station_name is None:
        station_name = _mesonet_station_names().get(station_id, station_id)

    # Exclude station metadata from interpolation.
    skip_var_names = ['stationID', 'latitude', 'longitude', 'elevation']

    # Convert the time coordinate to a pandas time index.
    time = pd.DatetimeIndex(ds.indexes['time'])

    # Reject duplicate times because their interpolation order is ambiguous.
    if time.has_duplicates:
        print('The dataset contains duplicate times:')

        # Show all records at duplicated timestamps.
        dup_all_mask = time.duplicated(keep=False)
        print(ds.isel(time=dup_all_mask).sortby('time'))

        raise ValueError('Cannot safely interpolate because the time coordinate contains duplicates.')

    # Duplicate times were rejected above, so this enforces strictly
    # increasing time.

    if not time.is_monotonic_increasing:
        raise ValueError('Cannot safely interpolate because the time coordinate does not increase monotonically.')

    # Require a finite and positive maximum interpolation interval.
    if not np.isfinite(max_interpolation_interval_h):
        raise ValueError('Maximum interpolation interval is not a finite number.')
    if max_interpolation_interval_h <= 0:
        raise ValueError('Maximum interpolation interval must be a positive number.')

    # Normalize the complete time index to UTC.
    if time.tz is None:
        time = time.tz_localize('UTC')
    else:
        time = time.tz_convert('UTC')

    # Determine the requested full-hour output period.
    if start_date is not None and end_date is not None:
        # Determine first and last full hours from the times provided as arguments.
        # Round both the start and end dates to the next-lower full hour.
        start_date_ = pd.Timestamp(start_date)
        if start_date_.tz is None:
            start_date_ = start_date_.tz_localize('UTC')
        else:
            start_date_ = start_date_.tz_convert('UTC')
        first_hour = start_date_.floor('h')
        end_date_ = pd.Timestamp(end_date)
        if end_date_.tz is None:
            end_date_ = end_date_.tz_localize('UTC')
        else:
            end_date_ = end_date_.tz_convert('UTC')

        # Reject reversed periods before rounding can conceal the reversal.
        if start_date_ > end_date_:
            raise ValueError('start_date must not be later than end_date.')

        last_hour = end_date_.floor('h')
        hours = pd.date_range(start=first_hour, end=last_hour, freq='h')
    elif time.size:
        # Restrict default output hours to the period covered by observations.
        first_hour = time[0].ceil('h')
        last_hour = time[-1].floor('h')
        if first_hour <= last_hour:
            hours = pd.date_range(start=first_hour, end=last_hour, freq='h')
        else:
            hours = pd.DatetimeIndex([], tz='UTC')
    else:
        # No implicit output period can be constructed without observations.
        hours = pd.DatetimeIndex([], tz='UTC')

    # Determine whether any observations can contribute to the output period.
    observation_overlap = bool(time.size and hours.size and hours[0] <= time[-1] and hours[-1] >= time[0])

    if not observation_overlap and verbose:
        print(f'No full-hour observations are available for station {station_id}. Using missing values.')

    # Retain printable bounds only for nonempty output periods.
    start_time = hours[0].to_pydatetime() if hours.size else None
    end_time = hours[-1].to_pydatetime() if hours.size else None

    # Store time without timezone information; UTC remains implicit.
    hours_datetime64 = hours.tz_localize(None).to_numpy(dtype='datetime64[ns]')

    # Initialize the output dataset with time and station coordinates.
    ds_out = xr.Dataset(coords={'time': hours_datetime64, 'station': [station_id]})

    # Specify the NetCDF time encoding.
    ds_out['time'].encoding = {
        'dtype': 'float64',
        'units': 'seconds since 1970-01-01T00:00:00+00:00',
        'calendar': 'proleptic_gregorian',
    }

    # Find times in the MADIS time series that bound each full hour.
    # Pandas indexes are used here to obtain positional comparisons
    # without xarray coordinate alignment.

    # Initialize every target hour without bounding observations.
    index_l = [None] * len(hours)
    index_r = [None] * len(hours)

    if observation_overlap:
        times_l = time[:-1]
        times_r = time[1:]

        for hour_i, hour in enumerate(hours):
            mask = (hour >= times_l) & (hour < times_r)

            if mask.any():
                index = int(np.flatnonzero(mask)[0])
                index_l[hour_i] = index
                index_r[hour_i] = index + 1

    # Interpolate the MADIS time series to full hours for each numeric
    # data variable that is a function of time only.

    hour_n = len(hours)
    max_interpolation_interval_s = 3600 * max_interpolation_interval_h

    if verbose:
        print()

    for var_name, data_array in ds.data_vars.items():
        if var_name in skip_var_names:
            continue

        if data_array.dims != ('time',):
            continue

        if not np.issubdtype(data_array.dtype, np.number):
            continue

        values = data_array.to_numpy()

        if verbose and hours.size:
            # Report the variable and output period being processed.
            print(
                'Constructing full-hourly time series for the station '
                + station_id
                + ' for the time range '
                + str(start_time)
                + '-'
                + str(end_time)
                + ' for '
                + var_name
            )

        # Initialize the hourly output values as missing.
        timeseries_hourly = np.full(hour_n, np.nan)

        # Loop over full hours.

        for hour_i in range(hour_n):
            hour = hours[hour_i]
            left_index = index_l[hour_i]
            right_index = index_r[hour_i]

            # Use an observation at the exact full hour without
            # interpolation.

            exact_value = left_index is not None and hour == time[left_index] and np.isfinite(values[left_index])
            if exact_value:
                timeseries_hourly[hour_i] = values[left_index]
                continue

            # Only proceed if the full hour has adjacent time indices.

            if left_index is not None and right_index is not None:
                # Search leftward for the nearest finite observation
                # within the maximum interpolation interval.

                ii = left_index
                value_l = np.nan
                time_l = None
                time_delta_seconds_l = np.nan

                while ii >= 0:
                    time_delta_seconds = abs((hour - time[ii]).total_seconds())

                    if time_delta_seconds > max_interpolation_interval_s:
                        break

                    if np.isfinite(values[ii]):
                        time_l = time[ii]
                        time_delta_seconds_l = time_delta_seconds
                        value_l = values[ii]
                        break

                    ii -= 1

                # Search rightward for the nearest finite observation
                # within the remaining interpolation interval.

                ii = right_index
                value_r = np.nan
                time_r = None

                if np.isfinite(time_delta_seconds_l):
                    while ii < len(time):
                        time_delta_seconds = abs((hour - time[ii]).total_seconds())

                        remaining_interval_s = max_interpolation_interval_s - time_delta_seconds_l
                        if time_delta_seconds > remaining_interval_s:
                            break

                        if np.isfinite(values[ii]):
                            time_r = time[ii]
                            value_r = values[ii]
                            break

                        ii += 1

                # Perform linear interpolation when both bounding
                # observations are finite.

                if np.isfinite(value_l) and np.isfinite(value_r):
                    timeseries_hourly[hour_i] = (
                        value_l + (hour - time_l).total_seconds() * (value_r - value_l) / (time_r - time_l).total_seconds()
                    )

        # Copy an exact final observation because it has no right-hand interval.
        if hours.size and time.size and np.isfinite(values[-1]):
            last_hour_position = hours.get_indexer([time[-1]])[0]
            if last_hour_position >= 0:
                timeseries_hourly[last_hour_position] = values[-1]

        # Add the interpolated observable as a function of time and station.

        ds_out[var_name] = (('time', 'station'), timeseries_hourly[:, np.newaxis].astype(np.float32))

        if 'long_name' in ds[var_name].attrs:
            ds_out[var_name].attrs['long_name'] = ds[var_name].attrs['long_name']

        if 'units' in ds[var_name].attrs:
            units = ds[var_name].attrs['units']
            if units == 'kelvin':
                ds_out[var_name].attrs['units'] = 'K'
            elif units == 'meter':
                ds_out[var_name].attrs['units'] = 'm'
            elif units == 'watt/meter2':
                ds_out[var_name].attrs['units'] = 'W m-2'
            elif units == 'meter/sec':
                ds_out[var_name].attrs['units'] = 'm s-1'
            else:
                ds_out[var_name].attrs['units'] = units

        ds_out[var_name].attrs['max_interpolation_interval_h'] = max_interpolation_interval_h

        ds_out[var_name].attrs['processing'] = (
            'Hourly values interpolated from values at reported '
            'station times, with a maximum interpolation interval of ' + str(np.round(max_interpolation_interval_h, 2)) + ' h'
        )

    # Add station name.
    ds_out['STATION_NAME'] = ('station', [station_name])

    # Add station latitude and its metadata.
    ds_out['LAT'] = ('station', [np.float32(ds['latitude'].values[0])])
    ds_out['LAT'].attrs['_FillValue'] = np.nan
    ds_out['LAT'].attrs['long_name'] = 'latitude'
    ds_out['LAT'].attrs['units'] = '° N'

    # Add station longitude and its metadata.
    ds_out['LON'] = ('station', [np.float32(ds['longitude'].values[0])])
    ds_out['LON'].attrs['_FillValue'] = np.nan
    ds_out['LON'].attrs['long_name'] = 'longitude'
    ds_out['LON'].attrs['units'] = '° E'

    # Add station elevation and its metadata.
    ds_out['ELEV'] = ('station', [np.float32(ds['elevation'].values[0])])
    ds_out['ELEV'].attrs['_FillValue'] = np.nan
    ds_out['ELEV'].attrs['long_name'] = 'elevation'
    ds_out['ELEV'].attrs['units'] = 'm'

    # Add global attributes describing the output dataset.
    ds_out.attrs['name'] = 'MADIS Mesonet surface data'
    ds_out.attrs['long_name'] = 'Meteorological Assimilation Data Ingest System Mesonet data'
    ds_out.attrs['description'] = (
        'MADIS Mesonet is a National Oceanic and Atmospheric '
        'Administration (NOAA) dataset that combines observations '
        'from thousands of local, state, federal, academic, and '
        'private stations.'
    )
    ds_out.attrs['source'] = 'National Centers for Environmental Prediction (NCEP)'
    ds_out.attrs['URL'] = 'https://madis-data.ncep.noaa.gov'
    ds_out.attrs['processed_with'] = 'https://github.com/jankazil/madis-data'

    # Add a global attribute with the list of files from which the dataset was constructed
    if 'files_orig' in ds.attrs:
        files_orig = ds.attrs['files_orig']
        if isinstance(files_orig, str):
            files_orig = [files_orig]
        ds_out.attrs['files_orig'] = list(files_orig)
    ds_out.attrs.pop('file_orig', None)

    return ds_out


def merge_hourly(station_datasets: dict[str, xr.Dataset]) -> xr.Dataset:
    '''
    Merge single-station hourly timeseries datasets along their existing
    'station' dimension. All datasets must have identical time coordinates.

    Parameters
    ----------
    station_datasets : dictionary whose values are xarray.Datasets with
        hourly time series for one station. This dictionary would typically
        be the result of a call of ``interpolate_to_full_hour_single``.

    Returns
    -------

    xarray.Dataset : xarray.Dataset with the station time series as variables
        with the dimensions (time,station).

    '''
    if not station_datasets:
        raise ValueError('station_datasets is empty')

    datasets = []

    for station_key, ds in station_datasets.items():
        if ds.sizes.get('station') != 1:
            raise ValueError(f'{station_key!r} has station dimension size {ds.sizes.get("station")}; expected 1')

        # Make station metadata coordinates rather than data variables.
        station_coords = [name for name in ('STATION_NAME', 'LAT', 'LON', 'ELEV') if name in ds]
        datasets.append(ds.set_coords(station_coords))

    combined = xr.concat(
        datasets,
        dim='station',
        data_vars='minimal',
        coords='minimal',
        compat='equals',
        join='exact',  # Fail rather than silently align different times
        combine_attrs='no_conflicts',
    )

    return combined


def plot_locations_conus(
    station_ids: list[str],
    station_lats: list[float],
    station_lons: list[float],
    plot_file: Path,
    markersize: float = 1,
    verbose: bool = False,
) -> None:
    '''
    Generate and save a map of station locations within the contiguous
    United States (CONUS).

    The map uses a Lambert Conformal projection consistent with the HRRR
    model domain. All stations are plotted using identical markers;
    elevations are not represented graphically.

    Parameters
    ----------
    station_ids : list[str]
        Station identifiers to plot.
    station_lats : list[float]
        Station latitudes in degrees north.
    station_lons : list[float]
        Station longitudes in degrees east.
    plot_file : Path
        Path at which to save the plot.
    markersize : float, optional
        Station-marker diameter in points. The default is 1.
    verbose : bool, optional
        If True, print the output path. The default is False.

    Returns
    -------
    None
        The function saves the plot and does not return a value.

    Raises
    ------
    ValueError
        If ``station_ids`` is empty.
    '''
    # Reject an empty station list because no map could be produced.
    if not station_ids:
        raise ValueError('station_ids must not be empty.')

    # Use the Lambert Conformal projection of the HRRR domain.
    hrrr_proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5, standard_parallels=(38.5, 38.5))

    # Use one geographic axes without a separate color-bar axes.
    fig, ax = plt.subplots(figsize=(13.8, 8), subplot_kw={'projection': hrrr_proj})

    # Limit the map to the contiguous United States.
    ax.set_extent([-123, -71, 25, 50], crs=ccrs.PlateCarree())

    # Draw coastlines and political boundaries.
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25)
    ax.add_feature(cfeature.STATES, linewidth=0.25)

    # Plot each station at its longitude and latitude.
    ax.plot(station_lons, station_lats, 'o', color='blue', markersize=markersize, transform=ccrs.PlateCarree())

    # Add labeled latitude and longitude grid lines.
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=1.0, linestyle='--')

    gl.x_inline = False
    gl.y_inline = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.top_labels = False
    gl.right_labels = False

    gl.xlocator = LongitudeLocator(nbins=6)
    gl.ylocator = LatitudeLocator(nbins=6)

    gl.xformatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
    gl.yformatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')

    gl.xlabel_style = {'rotation': 0, 'ha': 'center', 'va': 'top', 'size': 14}
    gl.ylabel_style = {'rotation': 0, 'ha': 'right', 'va': 'center', 'size': 14}

    # Create the output directory when it does not yet exist.
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure as a high-resolution plot.
    fig.savefig(plot_file, bbox_inches='tight', pad_inches=0, dpi=600)

    # Release the figure and its memory.
    plt.close(fig)

    if verbose:
        print('Created plot', plot_file)

    return


'''
Internal helper functions
'''


def _build_bulk_station_frame(
    ds: xr.Dataset,
    station_codes: np.ndarray,
    station_ids: list[str],
    record_positions: np.ndarray,
    record_dimension: str,
) -> tuple[pd.DataFrame, list[str]]:
    '''
    Build one pandas table for all requested station records.

    Every retained data variable must be a one-dimensional function of the MADIS
    record dimension, matching the constraint enforced by the original routines.

    Parameters
    ----------
    ds : xr.Dataset
        Loaded MADIS dataset containing all records.
    station_codes : np.ndarray
        Integer station code for each selected record.
    station_ids : list[str]
        Requested station identifiers in station-code order.
    record_positions : np.ndarray
        Positions of selected records in ``ds``.
    record_dimension : str
        Name of the dimension indexing MADIS records.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Table of selected records and the retained data-variable names.

    Raises
    ------
    ValueError
        If a station has no valid times or a retained variable does not use
        only the record dimension.
    '''

    # Identify selected records with finite times.
    selected_times = np.asarray(ds['time'].values)[record_positions]
    valid_time = np.isfinite(selected_times)

    valid_station_codes = station_codes[valid_time]
    valid_record_positions = record_positions[valid_time]
    stations_with_valid_times = np.zeros(len(station_ids), dtype=bool)
    stations_with_valid_times[np.unique(valid_station_codes)] = True

    # Confirm that every requested station has at least one valid time.
    for station_code, station_id in enumerate(station_ids):
        if not stations_with_valid_times[station_code]:
            raise ValueError(f'Station {station_id!r} has no records with valid times.')

    # Retain observational variables that can become functions of time.
    variable_names = [
        name
        for name in ds.data_vars
        if name
        not in {'stationId', 'time', 'stationID', 'latitudes', 'longitudes', 'elevations', 'latitude', 'longitude', 'elevation'}
    ]
    unexpected_dimensions = {name: ds[name].dims for name in variable_names if ds[name].dims != (record_dimension,)}

    # Reject variables that cannot be represented in a station time series.
    if unexpected_dimensions:
        raise ValueError(f'Some variables cannot be represented using only the time dimension: {unexpected_dimensions}')

    # Assemble station codes, times, and data values in one table.
    frame_data = {
        'stationCode': valid_station_codes,
        'time': np.asarray(ds['time'].values)[valid_record_positions],
    }

    for variable_name in variable_names:
        variable_values = np.asarray(ds[variable_name].values)
        selected_values = variable_values[valid_record_positions]

        # Treat nonfinite numeric values as missing before duplicate consolidation.
        if np.issubdtype(selected_values.dtype, np.number):
            finite_values = np.isfinite(selected_values)
            if not finite_values.all():
                selected_values = selected_values.astype(np.result_type(selected_values.dtype, np.float64), copy=True)
                selected_values[~finite_values] = np.nan

        frame_data[variable_name] = selected_values

    station_frame = pd.DataFrame(frame_data, copy=False)

    return station_frame, variable_names


def _bulk_frame_to_station_datasets(
    grouped_frame: pd.DataFrame,
    ds: xr.Dataset,
    station_ids: list[str],
    variable_names: list[str],
) -> dict[str, xr.Dataset]:
    '''
    Convert a grouped station table to individual xarray datasets.

    Parameters
    ----------
    grouped_frame : pd.DataFrame
        Table grouped by station code and time with duplicates consolidated.
    ds : xr.Dataset
        Source dataset providing metadata.
    station_ids : list[str]
        Requested station identifiers in station-code order.
    variable_names : list[str]
        Data variables to include in every station dataset.

    Returns
    -------
    dict[str, xr.Dataset]
        Dataset for each station, keyed by station identifier.

    Raises
    ------
    ValueError
        If an output variable is not solely a function of time.
    '''

    # Match requested stations to the source station table.
    station_table_ids = np.asarray(mesonet_station_ids2strings(ds['stationID']))
    station_table_positions: dict[str, int] = {}

    for station_id in station_ids:
        positions = np.flatnonzero(station_table_ids == station_id)
        if positions.size == 0:
            raise KeyError(f'Station ID {station_id!r} was not found in the station table.')
        station_table_positions[station_id] = int(positions[0])

    # Split the consolidated table into one table per station code.
    station_frames = {
        int(station_code): frame.drop(columns='stationCode').set_index('time')
        for station_code, frame in grouped_frame.groupby('stationCode', sort=False, observed=True)
    }

    # Build the output dictionary in requested-station order.
    ds_dictionary = {}

    for station_code, station_id in enumerate(station_ids):
        # Convert the station table to an xarray dataset indexed by time.
        station_frame = station_frames[station_code]
        station_ds = station_frame[variable_names].to_xarray()

        unexpected_dimensions = {
            name: variable.dims for name, variable in station_ds.data_vars.items() if variable.dims != ('time',)
        }

        # Confirm that every variable is solely a function of time.
        if unexpected_dimensions:
            raise ValueError(f'Some variables cannot be represented using only the time dimension: {unexpected_dimensions}')

        # Restore global, coordinate, and time-dependent variable metadata.
        station_ds.attrs = ds.attrs.copy()
        station_ds.attrs.pop('stationID', None)
        station_ds['time'].attrs = ds['time'].attrs.copy()

        for variable_name in station_ds.data_vars:
            station_ds[variable_name].attrs = ds[variable_name].attrs.copy()

        # Add the identifier and location metadata on a singleton station
        # dimension. Values come from the source station table, while metadata
        # from the corresponding legacy record variables is retained.
        station_position = station_table_positions[station_id]
        station_metadata_names = {
            'stationID': 'stationId',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'elevation': 'elevation',
        }

        station_ds['stationID'] = ('station', np.asarray([station_id], dtype=str))

        for output_name, legacy_name in station_metadata_names.items():
            if output_name != 'stationID':
                value = np.asarray(ds[output_name].values)[station_position]
                station_ds[output_name] = ('station', np.asarray([value], dtype=np.float32))

            station_ds[output_name].attrs = ds[output_name].attrs.copy()
            station_ds[output_name].attrs.update(ds[legacy_name].attrs)

            # Preserve missing-data encoding used by the legacy float
            # metadata variables.
            if output_name != 'stationID':
                for encoding_name in ('_FillValue', 'missing_value'):
                    if encoding_name in ds[legacy_name].encoding:
                        station_ds[output_name].encoding[encoding_name] = ds[legacy_name].encoding[encoding_name]

        # Define the NetCDF time encoding.
        station_ds['time'].encoding = {
            'dtype': 'float64',
            'units': 'seconds since 1970-01-01T00:00:00+00:00',
            'calendar': 'proleptic_gregorian',
        }

        # Store the finished dataset under its station identifier.
        ds_dictionary[station_id] = station_ds

    return ds_dictionary


@cache
def _mesonet_station_names() -> dict[str, str]:
    '''Return cleaned station names from the packaged NOAA station table.'''

    # Access the station table through the installed madis_data package.
    resource = files('madis_data') / 'data' / 'NOAA' / 'public_stntbl.csv'
    with as_file(resource) as station_table_file:
        # Read only the station identifier and station name columns.
        madis_stations = pd.read_csv(
            station_table_file,
            header=None,
            usecols=[0, 7],
            names=['stationId', 'stationName'],
            dtype=str,
            keep_default_na=False,
        )

    # Remove fixed-width padding from station identifiers.
    madis_stations['stationId'] = madis_stations['stationId'].str.strip()

    # Discard the subprovider stored in characters 40-50 of the station-location field.
    station_names = madis_stations['stationName'].str.slice(stop=39).str.rstrip()
    # Remove padded state, province, and country designations used internationally.
    station_names = station_names.str.replace(r'\s{2,}(?:[A-Z]{2}\s+)?[A-Z]{2}$', '', regex=True)
    # Remove compact United States and Canadian state/province-country designations.
    station_names = station_names.str.replace(r'\s+[A-Z]{2}\s+(?:US|CA)$', '', regex=True)
    madis_stations['stationName'] = station_names.str.strip()

    # Ignore empty identifiers or names.
    usable = madis_stations['stationId'].ne('') & madis_stations['stationName'].ne('')
    madis_stations = madis_stations.loc[usable].drop_duplicates('stationId', keep='first')

    # Build the cached station identifier-to-name lookup.
    return dict(zip(madis_stations['stationId'], madis_stations['stationName'], strict=True))


def _pack_mesonet_station_ids(station_id: xr.DataArray) -> np.ndarray | None:
    '''
    Pack a MADIS station-identifier array into fixed-width byte strings.

    Parameters
    ----------
    station_id : xr.DataArray
        One- or two-dimensional MADIS station-identifier array.

    Returns
    -------
    np.ndarray or None
        Cleaned one-dimensional byte strings, or None when the input does not
        use a byte-string dtype.

    Raises
    ------
    ValueError
        If the input has neither one nor two dimensions.
    '''

    # Access the underlying station-identifier values.
    values = np.asarray(station_id.values)

    if values.ndim not in {1, 2}:
        raise ValueError(f'Unexpected stationId shape: {values.shape}')

    if values.dtype.kind != 'S':
        return None

    if values.ndim == 2:
        # Return empty identifiers when the character dimension is empty.
        if values.shape[1] == 0:
            return np.full(values.shape[0], b'', dtype='S1')

        # MADIS commonly stores stationId as an (n_records, n_characters)
        # array of one-byte strings. Reinterpret each contiguous row as one
        # fixed-width byte string rather than joining every row in Python.
        values = np.ascontiguousarray(values)
        row_width = values.shape[1] * values.dtype.itemsize
        encoded_ids = values.view(f'S{row_width}').reshape(values.shape[0])
    else:
        encoded_ids = values

    # Remove null characters and surrounding padding.
    return np.char.strip(np.char.replace(encoded_ids, b'\x00', b''))


def _prepare_mesonet_station_id_keys(
    station_id: xr.DataArray,
    requested_station_ids: list[str],
) -> np.ndarray:
    '''
    Prepare station identifiers for fast record selection.

    Parameters
    ----------
    station_id : xr.DataArray
        MADIS station-identifier array.
    requested_station_ids : list[str]
        Station identifiers that will be selected.

    Returns
    -------
    np.ndarray
        Packed byte identifiers for ordinary ASCII requests, or decoded
        Unicode identifiers when byte matching is unsafe.
    '''

    # Pack byte identifiers without converting every record to Unicode.
    encoded_ids = _pack_mesonet_station_ids(station_id)

    # ASCII requests can be matched exactly against the cleaned MADIS bytes.
    # A non-ASCII request requires the UTF-8 behavior of the public decoder.
    if encoded_ids is not None and all(station_id.isascii() for station_id in requested_station_ids):
        return encoded_ids

    return np.asarray(mesonet_station_ids2strings(station_id))


def _select_mesonet_station_records(
    station_id_keys: np.ndarray,
    station_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Locate records belonging to requested stations.

    Parameters
    ----------
    station_id_keys : np.ndarray
        One station-selection key per MADIS record.
    station_ids : list[str]
        Unique station identifiers to select, in output order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Record positions and corresponding integer station codes.

    Raises
    ------
    KeyError
        If any requested station is absent.
    '''

    # Assign one compact integer code to each requested station.
    station_count = len(station_ids)
    requested_keys = []
    matchable_codes = []

    if station_id_keys.dtype.kind == 'S':
        key_width = station_id_keys.dtype.itemsize

        # Encode only identifiers that fit without truncation.
        for station_code, station_id in enumerate(station_ids):
            encoded_id = station_id.encode('ascii')

            # Converting an overlong key to a fixed-width NumPy byte string
            # would truncate it and could create a false match.
            if len(encoded_id) <= key_width:
                requested_keys.append(encoded_id)
                matchable_codes.append(station_code)
    else:
        requested_keys = station_ids
        matchable_codes = list(range(station_count))

    # Track which requested stations were found.
    found = np.zeros(station_count, dtype=bool)

    # Direct comparisons avoid allocating one full-size integer search array
    # when only a few stations are requested.
    if len(requested_keys) <= 8:
        position_groups = []
        station_code_groups = []

        for requested_key, station_code in zip(requested_keys, matchable_codes, strict=True):
            positions = np.flatnonzero(station_id_keys == requested_key)

            if positions.size:
                # Store matching positions and their station code.
                found[station_code] = True
                position_groups.append(positions)
                station_code_groups.append(np.full(positions.size, station_code, dtype=np.intp))

        if position_groups:
            record_positions = np.concatenate(position_groups)
            station_codes = np.concatenate(station_code_groups)
        else:
            record_positions = np.empty(0, dtype=np.intp)
            station_codes = np.empty(0, dtype=np.intp)
    else:
        # Sort many requested keys for one efficient bulk search.
        if station_id_keys.dtype.kind == 'S':
            requested_keys = np.asarray(requested_keys, dtype=station_id_keys.dtype)
        else:
            requested_keys = np.asarray(requested_keys)

        matchable_codes = np.asarray(matchable_codes, dtype=np.intp)
        key_order = np.argsort(requested_keys, kind='stable')
        sorted_keys = requested_keys[key_order]
        sorted_codes = matchable_codes[key_order]

        insertion_points = np.searchsorted(sorted_keys, station_id_keys)
        possible_match = insertion_points < sorted_keys.size
        possible_positions = np.flatnonzero(possible_match)
        possible_insertions = insertion_points[possible_positions]
        exact_match = sorted_keys[possible_insertions] == station_id_keys[possible_positions]

        record_positions = possible_positions[exact_match]
        station_codes = sorted_codes[possible_insertions[exact_match]]
        found[np.unique(station_codes)] = True

    # Report the first requested station that was not found.
    missing_codes = np.flatnonzero(~found)
    if missing_codes.size:
        missing_station = station_ids[int(missing_codes[0])]
        raise KeyError(f'Station ID {missing_station!r} was not found.')

    return record_positions, station_codes


def _value2string(value: object) -> str:
    '''
    Convert one identifier value to text while preserving missingness.
    '''
    # Normalize null object values to empty identifier components.
    if value is None:
        return ''

    # Normalize floating-point NaN values to empty identifier components.
    try:
        if bool(pd.isna(value)):
            return ''
    except (TypeError, ValueError):
        pass

    # Decode byte values without retaining their Python representation.
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode('utf-8', errors='replace')

    # Convert uncommon numeric or object values to their text representation.
    return str(value)
