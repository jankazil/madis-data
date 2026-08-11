'''
Download, process, extract, interpolate, and plot Meteorological Assimilation
Data Ingest System (MADIS) METAR data.
'''

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path

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

from madis_data.stations import (
    InvalidStationDataFileError,
    StationDataSource,
    download_station_data,
    extract_station_files,
    local_file_paths,
    local_file_paths_preprocessed,
    prepare_station_id_keys,
    select_station_records,
    station_ids2strings,
)
from madis_data.xarray_tools import get_missing_value, mask_with_missing_value


def metar_variables() -> tuple[list[str], list[str], list[str], list[str]]:
    '''
    Return the MADIS METAR variables to operate on.

    Returns
    -------
    tuple[list[str], list[str], list[str], list[str]]
        Variables without quality-control flags, variables with flags,
        quality-control flag variables, and preprocessed output variables.
    '''

    # List variables that do not have a quality-control flag.
    mm_vars_no_qc = ['stationName', 'latitude', 'longitude', 'elevation']

    # List source variables whose values are filtered with quality-control flags.
    mm_vars_with_qc = ['tempFromTenths', 'dpFromTenths', 'windDir', 'windSpeed']

    # The tenths-degree variables use the flags attached to the whole-degree fields.
    mm_vars_qc = ['temperatureDD', 'dewpointDD', 'windDirDD', 'windSpeedDD']

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
    ]

    return mm_vars_no_qc, mm_vars_with_qc, mm_vars_qc, mm_vars_pp


def metar_urls(start_date: datetime, end_date: datetime) -> list[str]:
    '''
    Construct hourly MADIS METAR archive URLs for an inclusive date range.

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
            f'{date.year}/{date.month:02d}/{date.day:02d}/point/metar/'
            f'netcdf/{date.year}{date.month:02d}{date.day:02d}_'
            f'{date.hour:02d}00.gz'
        )
        for date in dates
    ]

    return urls


def metar_local_file_paths(data_dir: Path, urls: list[str]) -> list[Path]:
    '''
    Construct local paths for MADIS METAR URLs.

    Parameters
    ----------
    data_dir : Path
        Directory in which the files are or will be stored.
    urls : list[str]
        MADIS METAR file URLs.

    Returns
    -------
    list[Path]
        Local paths formed from the URL filenames.
    '''

    return local_file_paths(data_dir, urls, 'metar')


def metar_local_file_paths_preprocessed(file_paths: list[Path]) -> list[Path]:
    '''
    Construct paths for the preprocessed versions of MADIS METAR files.

    Parameters
    ----------
    file_paths : list[Path]
        Paths of original MADIS METAR files.

    Returns
    -------
    list[Path]
        Corresponding paths with the suffix changed to ``.nc``.
    '''

    return local_file_paths_preprocessed(file_paths)


def metar_station_ids2strings(station_id: xr.DataArray) -> np.ndarray:
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

    return station_ids2strings(station_id)


def download_metar(
    start_date: datetime,
    end_date: datetime,
    data_dir: Path,
    refresh: bool = False,
    remove_original: bool = False,
    verbose: bool = False,
    n_jobs: int = 8,
) -> list[Path]:
    '''
    Download and preprocess MADIS METAR files.

    Existing valid output files are reused unless ``refresh`` is True.
    Preprocessing results in the  selected observations being quality
    controlled, converted to the output variables, and saved as NetCDF
    files.

    Parameters
    ----------
    start_date : datetime
        First date to download.
    end_date : datetime
        Last date to download.
    data_dir : Path
        Directory in which files are stored.
    refresh : bool, optional
        If True, download files even when local copies exist. The default is
        False.
    remove_original : bool, optional
        If True, remove downloaded original MADIS METAR files after successful preprocessing.
        The default is False.
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

    return download_station_data(
        start_date,
        end_date,
        data_dir,
        source=METAR_DATA_SOURCE,
        refresh=refresh,
        remove_original=remove_original,
        verbose=verbose,
        n_jobs=n_jobs,
    )


def extract_metar_data(file_paths: list[Path], n_jobs: int = 1) -> list[xr.Dataset]:
    '''
    Preprocess multiple MADIS METAR files in parallel.

    Parameters
    ----------
    file_paths : list[Path]
        Paths of original MADIS METAR files.
    n_jobs : int, optional
        Number of worker threads. The default is 1.

    Returns
    -------
    list[xr.Dataset]
        One preprocessed dataset for each input path, in input order.
    '''

    return extract_station_files(
        file_paths,
        extract_metar_data_single,
        n_jobs=n_jobs,
    )


def extract_metar_data_single(file_path: Path) -> xr.Dataset:
    '''
    Preprocess and quality-control a MADIS METAR file.

    Parameters
    ----------
    file_path : Path
        Path of an original MADIS METAR file.

    Returns
    -------
    xr.Dataset
        Dataset containing retained observations, Cartesian wind components,
        processing metadata, and the original data file path.
    '''

    # Obtain the variable groups used for extraction and quality control.
    mm_vars_no_qc, mm_vars_with_qc, mm_vars_qc, _ = metar_variables()

    # Read only the required source variables and standardize missing values.
    selected_variables = mm_vars_no_qc + mm_vars_with_qc + mm_vars_qc
    ds = preprocess_metar_file(file_path, selected_variables)

    # Retain values flagged as screened or verified.
    for varname, varname_qc in zip(mm_vars_with_qc, mm_vars_qc, strict=True):
        da_var = ds[varname]
        da_qc = ds[varname_qc]

        # Obtain the missing value used by this variable.
        missing_value = np.asarray(get_missing_value(da_var), dtype=np.float32)

        # Identify observations that passed quality control.
        qc_mask = (da_qc == np.bytes_('S')) | (da_qc == np.bytes_('V'))

        # Replace rejected observations with the declared missing value.
        ds[varname] = mask_with_missing_value(da_var, qc_mask, missing_value).astype(np.float32)

    # Give the tenths-degree source fields the standard processed names.
    ds = ds.rename(
        {
            'tempFromTenths': 'temperature',
            'dpFromTenths': 'dewpoint',
        }
    )

    # Reject wind speeds outside the permitted range of 0 to 50 m s-1.
    varname_ws = 'windSpeed'
    varname_wd = 'windDir'

    if varname_ws in ds.data_vars:
        da_var = ds[varname_ws]
        missing_value = np.asarray(get_missing_value(da_var), dtype=np.float32)
        qc_mask = (da_var >= 0.0) & (da_var <= 50.0)
        ds[varname_ws] = mask_with_missing_value(da_var, qc_mask, missing_value).astype(np.float32)

        if varname_wd in ds.data_vars:
            # Reject wind directions associated with invalid wind speeds.
            da_var = ds[varname_wd]
            missing_value = np.asarray(get_missing_value(da_var), dtype=np.float32)
            ds[varname_wd] = mask_with_missing_value(da_var, qc_mask, missing_value).astype(np.float32)

    # Convert wind speed and direction to Cartesian wind components.
    if varname_ws in ds.data_vars and varname_wd in ds.data_vars:
        ws = ds[varname_ws]
        wd = ds[varname_wd]

        # Mark records for which both wind values are finite.
        mask = np.isfinite(ws) & np.isfinite(wd)

        # Convert meteorological direction from degrees to radians.
        theta = np.deg2rad(wd)

        # Calculate west-east and south-north wind components.
        ds['U10'] = xr.where(mask, -ws * np.sin(theta), np.float32(np.nan), keep_attrs=False).astype(np.float32)
        ds['V10'] = xr.where(mask, -ws * np.cos(theta), np.float32(np.nan), keep_attrs=False).astype(np.float32)

        # Describe the calculated wind components.
        ds['U10'].attrs['long_name'] = 'West-east wind speed'
        ds['U10'].attrs['units'] = ws.attrs['units']
        ds['U10'].attrs['missing_value'] = np.float32(np.nan)
        ds['U10'].encoding['_FillValue'] = np.float32(np.nan)

        ds['V10'].attrs['long_name'] = 'South-north wind speed'
        ds['V10'].attrs['units'] = ws.attrs['units']
        ds['V10'].attrs['missing_value'] = np.float32(np.nan)
        ds['V10'].encoding['_FillValue'] = np.float32(np.nan)

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

    # Record the original data file path.
    ds.attrs['file_orig'] = str(file_path)

    return ds


def preprocess_metar_file(file_path: Path, varnames: str | list[str]) -> xr.Dataset:
    '''
    Read selected variables from a MADIS METAR file.

    Declared missing values are replaced with NaN. ``timeObs`` and
    ``stationName`` are renamed ``time`` and ``stationId``, respectively, and
    time is configured for NetCDF output as seconds since the Unix epoch.

    Parameters
    ----------
    file_path : Path
        Path of a compressed or uncompressed MADIS NetCDF file.
    varnames : str or list[str]
        Data variables to retain. ``timeObs`` is included automatically.

    Returns
    -------
    xr.Dataset
        Dataset containing the selected variables and standardized missing
        values and time encoding.
    '''
    # Convert a single variable name to a list.
    if isinstance(varnames, str):
        varnames = [varnames]

    # Always retain observation time and remove duplicate variable names.
    varnames = ['timeObs'] + varnames
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
        InvalidStationDataFileError
            If any required variable is absent.
        '''
        # Identify required variables that are absent from the file.
        missing = [name for name in varnames if name not in ds.data_vars]

        if missing:
            source = ds.encoding.get('source')
            raise InvalidStationDataFileError(f'Missing data variables: {missing} in {source}')

        # Return only the requested variables.
        return ds[varnames]

    # Suppress warnings raised while opening the MADIS file collection.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        ds = xr.open_mfdataset(file_path, preprocess=preprocess, decode_times=False)

        # Retain only observation times in the supported half-open interval.
        time_obs = ds['timeObs']
        time_min = np.datetime64('1970-01-01T00:00:00', 's').astype(np.int64)
        time_max = np.datetime64('3000-01-01T00:00:00', 's').astype(np.int64)
        valid_time = (time_obs >= time_min) & (time_obs < time_max)
        valid_time_values = np.asarray(valid_time.compute().values, dtype=bool)

        if not np.any(valid_time_values):
            source = ds.encoding.get('source', file_path)
            ds.close()
            raise InvalidStationDataFileError(
                f'No valid timeObs values in {source}; expected times in [1970-01-01, 3000-01-01).'
            )

        if not np.all(valid_time_values):
            record_dimension = time_obs.dims[0]
            ds = ds.isel({record_dimension: np.flatnonzero(valid_time_values)})

        # Decode the validated numeric observation times.
        ds = xr.decode_cf(ds)

        # Rename METAR identifier and time fields to the shared output schema.
        ds = ds.rename({'stationName': 'stationId', 'timeObs': 'time'})

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

                if np.issubdtype(variable.dtype, np.floating):
                    variable = variable.where(
                        ~variable.isin(valid_fill_values),
                        other=np.float32(np.nan),
                    ).astype(np.float32)
                else:
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
                ds[name].encoding['_FillValue'] = np.float32(np.nan)
                ds[name].attrs['missing_value'] = np.float32(np.nan)

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


METAR_DATA_SOURCE = StationDataSource(
    name='METAR',
    file_prefix='metar',
    urls=metar_urls,
    variables=metar_variables,
    extract_file=extract_metar_data_single,
)


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
        Whether to display extraction progress. The default is True.

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
    station_id_keys = prepare_station_id_keys(ds['stationId'], requested_station_ids)

    if station_id_keys.shape != (ds.sizes[record_dimension],):
        raise ValueError(f'Station ID keys do not correspond one-to-one with the {record_dimension} dimension.')

    # Report completion of station-identifier preparation.
    report_progress('preparation of station identifiers - moving on to selection of data at requested stations ...')

    if requested_station_ids:
        # Find every requested record and assign its compact station code.
        record_positions, station_codes = select_station_records(station_id_keys, requested_station_ids)
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
    show_progress: bool = True,
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

    if show_progress:
        print('Interpolating time series to full hour ...')
        print('\rProgress: 0.00 %', end='', flush=True)

    # Return an empty result without constructing an invalid process pool.
    if not stations:
        if show_progress:
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
    station_names = _metar_station_names()

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
            if show_progress:
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
        station_name = _metar_station_names().get(station_id, station_id)

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
        timeseries_hourly = np.full(hour_n, np.nan, dtype=np.float32)

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

        ds_out[var_name] = (('time', 'station'), timeseries_hourly[:, np.newaxis])

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
    ds_out['LAT'].attrs['_FillValue'] = np.float32(np.nan)
    ds_out['LAT'].attrs['long_name'] = 'latitude'
    ds_out['LAT'].attrs['units'] = '° N'

    # Add station longitude and its metadata.
    ds_out['LON'] = ('station', [np.float32(ds['longitude'].values[0])])
    ds_out['LON'].attrs['_FillValue'] = np.float32(np.nan)
    ds_out['LON'].attrs['long_name'] = 'longitude'
    ds_out['LON'].attrs['units'] = '° E'

    # Add station elevation and its metadata.
    ds_out['ELEV'] = ('station', [np.float32(ds['elevation'].values[0])])
    ds_out['ELEV'].attrs['_FillValue'] = np.float32(np.nan)
    ds_out['ELEV'].attrs['long_name'] = 'elevation'
    ds_out['ELEV'].attrs['units'] = 'm'

    # Add global attributes describing the output dataset.
    ds_out.attrs['name'] = 'MADIS METAR surface data'
    ds_out.attrs['long_name'] = 'Meteorological Assimilation Data Ingest System METAR data'
    ds_out.attrs['description'] = (
        'MADIS METAR is a National Oceanic and Atmospheric Administration '
        '(NOAA) dataset containing aviation routine and special surface '
        'weather reports, including ASOS, AWOS, and non-automated reports.'
    )
    ds_out.attrs['source'] = 'National Centers for Environmental Prediction (NCEP)'
    ds_out.attrs['URL'] = 'https://madis-data.ncep.noaa.gov'
    if 'region' in ds.attrs:
        ds_out.attrs['region'] = ds.attrs['region']
    ds_out.attrs['processed_with'] = 'https://github.com/jankazil/madis-data'

    # Add a global attribute with the list of files from which the dataset was constructed
    #    if 'files_orig' in ds.attrs:
    #        files_orig = ds.attrs['files_orig']
    #        if isinstance(files_orig, str):
    #            files_orig = [files_orig]
    #        ds_out.attrs['files_orig'] = list(files_orig)
    #    ds_out.attrs.pop('file_orig', None)

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


def filter_by_valid_fraction(
    ds: xr.Dataset,
    vars: list[str],
    min_valid_fraction: float,
) -> xr.Dataset:
    '''
    Select stations with a minimum fraction of concurrently valid observations.

    At each time and station, an observation is considered valid only if the
    values of all specified variables are finite. The function retains stations
    for which the fraction of such times is at least ``min_valid_fraction``.

    Parameters
    ----------
    ds
        Dataset containing the ``station`` and ``time`` dimensions.
    vars
        Names of variables to assess. Each variable must have the dimensions
        ``station`` and ``time``.
    min_valid_fraction
        Minimum required fraction of valid times, between 0 and 1.

    Returns
    -------
    xr.Dataset
        Dataset containing only the stations satisfying the validity criterion.
    '''

    # Construct an array holding the variable values with the shape (variable, time, station)
    values = ds[vars].to_array(dim='variable')

    # Construct a boolean array with the shape (variable, time, station) that is
    # True where the values are finite, logically collapse it with 'and' along the
    # variable dimension, and calculate the fraction of True values along the time dimension
    valid_fraction = np.isfinite(values).all(dim='variable').mean(dim='time')

    # Boolean mask that is true at station indices that have a valid fraction, and False elsewhere
    valid_station_mask = valid_fraction.values >= min_valid_fraction

    # Give the indices of the True values in the array, skipping the False values
    valid_station_ixs = np.flatnonzero(valid_station_mask).tolist()

    # Construct a dataset with only the stations that have a valid fraction of values
    ds_valid = ds.isel(station=valid_station_mask)

    ds_valid.attrs['processing_note_1'] = (
        f'{len(valid_station_ixs)} out of {ds.sizes["station"]} stations are included that have at least {100 * min_valid_fraction} % of the time concurrently valid values for the variables '
        + ', '.join(vars)
        + '.'
    )

    return ds_valid


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
                selected_values = selected_values.copy()
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
    station_table_ids = np.asarray(station_ids2strings(ds['stationID']))
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
def _metar_station_names() -> dict[str, str]:
    '''Return station names from the packaged UCAR METAR station table.'''

    # Access the station table through the installed madis_data package.
    resource = files('madis_data') / 'data' / 'UCAR' / 'stations.txt'
    with as_file(resource) as station_table_file:
        # Read the fixed-width 16-character station name and ICAO identifier.
        madis_stations = pd.read_fwf(
            station_table_file,
            header=None,
            colspecs=[(3, 19), (20, 24)],
            names=['stationName', 'stationId'],
            dtype=str,
            keep_default_na=False,
        )

    # Remove fixed-width padding and normalize underscores in station names.
    madis_stations['stationId'] = madis_stations['stationId'].str.strip()
    madis_stations['stationName'] = madis_stations['stationName'].str.replace('_', ' ', regex=False).str.strip()

    # Ignore headings and other nonstation rows in the mixed-format source file.
    usable = madis_stations['stationId'].str.fullmatch(r'[A-Z0-9]{4}') & madis_stations['stationName'].ne('')
    madis_stations = madis_stations.loc[usable].drop_duplicates('stationId', keep='first')

    # Build the cached station identifier-to-name lookup.
    return dict(zip(madis_stations['stationId'], madis_stations['stationName'], strict=True))
