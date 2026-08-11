'''
Shared helpers for downloading and processing MADIS station observations.

Product-specific modules provide archive URL construction, variable metadata,
and single-file preprocessing through :class:`StationDataSource`.
'''

import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr

from madis_data.web import download_threaded


@dataclass(frozen=True)
class StationDataSource:
    '''Describe the product-specific operations used by shared station helpers.'''

    name: str
    file_prefix: str
    urls: Callable[[datetime, datetime], list[str]]
    variables: Callable[[], tuple[list[str], list[str], list[str], list[str]]]
    extract_file: Callable[[Path], xr.Dataset]


def local_file_paths(data_dir: Path, urls: list[str], file_prefix: str) -> list[Path]:
    '''
    Construct local paths for station-data URLs.

    Parameters
    ----------
    data_dir : Path
        Directory in which the files are or will be stored.
    urls : list[str]
        Station-data file URLs.
    file_prefix : str
        Prefix identifying the source product in each local filename.

    Returns
    -------
    list[Path]
        Local paths formed from the prefixed URL filenames.
    '''

    return [data_dir / f'{file_prefix}_{PurePosixPath(urlparse(file_url).path).name}' for file_url in urls]


def local_file_paths_preprocessed(file_paths: list[Path]) -> list[Path]:
    '''
    Construct paths for preprocessed station-data files.

    Parameters
    ----------
    file_paths : list[Path]
        Paths of original station-data files.

    Returns
    -------
    list[Path]
        Corresponding paths with the suffix changed to ``.nc``.
    '''

    return [file_path.with_suffix('.nc') for file_path in file_paths]


def station_ids2strings(station_id: xr.DataArray) -> np.ndarray:
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
    encoded_ids = _pack_station_ids(station_id)

    if encoded_ids is not None:
        # Cleaning before Unicode conversion is substantially faster for the
        # normal ASCII station IDs. Preserve replacement behavior for
        # unexpected non-ASCII or invalid UTF-8 bytes.
        try:
            return encoded_ids.astype(str)
        except UnicodeDecodeError:
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


def download_station_data(
    start_date: datetime,
    end_date: datetime,
    data_dir: Path,
    source: StationDataSource,
    refresh: bool = False,
    remove_original: bool = False,
    verbose: bool = False,
    n_jobs: int = 8,
) -> list[Path]:
    '''
    Download and preprocess files from a MADIS station-data source.

    Existing valid output files are reused unless ``refresh`` is True.
    Preprocessing quality-controls the selected observations, converts them to
    the product's output variables, and saves them as NetCDF files.

    Parameters
    ----------
    start_date : datetime
        First date/time to download.
    end_date : datetime
        Last date/time to download.
    data_dir : Path
        Directory used for downloaded and preprocessed files.
    source : StationDataSource
        Product-specific URL, variable, filename, and extraction operations.
    refresh : bool, optional
        If True, download files even when local copies exist. The default is
        False.
    remove_original : bool, optional
        If True, remove original files after successful preprocessing. The
        default is False.
    verbose : bool, optional
        Passed to the downloader to control messages. The default is False.
    n_jobs : int, optional
        Number of files downloaded and preprocessed concurrently. The default
        is 8.

    Returns
    -------
    list[Path]
        Sorted paths of the downloaded or reused preprocessed files.
    '''

    # Require at least one worker.
    if n_jobs < 1 or not np.isfinite(n_jobs):
        raise ValueError('n_jobs must be a positive finite integer.')

    # Create the data directory if needed.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Construct the archive URLs for the requested dates.
    file_urls = source.urls(start_date, end_date)

    # Collect files that can be reused instead of downloaded.
    skip_urls = []
    skip_files = []
    skip_files_preprocessed = []

    # Always preprocess
    preprocess = True

    if not refresh:
        # Construct the expected local original MADIS station-data file paths.
        file_paths = local_file_paths(data_dir, file_urls, source.file_prefix)

        if preprocess:
            # Construct the expected preprocessed paths.
            file_paths_preprocessed = local_file_paths_preprocessed(file_paths)

            # Obtain the variables required in a complete preprocessed file.
            _, _, _, mm_vars_pp = source.variables()

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
            # Reuse existing original MADIS station-data files when preprocessing is disabled.
            for file_url, file_path in zip(file_urls, file_paths, strict=True):
                if file_path.is_file():
                    skip_urls.append(file_url)

        # Remove reusable URLs from the download list.
        skip_urls = list(set(skip_urls))
        skip_files = local_file_paths(data_dir, skip_urls, source.file_prefix)
        skip_files_preprocessed = local_file_paths_preprocessed(skip_files)

        # Remove original MADIS station-data files whose valid preprocessed files are being reused.
        if preprocess and remove_original:
            for file_path in skip_files:
                remove_source_file(file_path)

        file_urls = [url for url in file_urls if url not in skip_urls]

    # Collect the paths produced during this call.
    downloaded_files = []

    # Print initial progress information.
    print(f'Downloading {source.name} data files ...')
    print(f'\rProgress: {0:.2f} %', end='', flush=True)

    for ii in range(0, len(file_urls), n_jobs):
        # Process one worker-sized batch of URLs.
        file_urls_short_list = file_urls[ii : ii + n_jobs]

        # Construct prefixed source paths and reuse any that already exist.
        file_paths_short_list = local_file_paths(data_dir, file_urls_short_list, source.file_prefix)
        urls_to_download = [
            file_url
            for file_url, file_path in zip(file_urls_short_list, file_paths_short_list, strict=True)
            if refresh or not file_path.is_file()
        ]

        failed_file_paths: set[Path] = set()

        if urls_to_download:
            with TemporaryDirectory(
                prefix=f'{source.file_prefix}_download_',
                dir=data_dir,
            ) as download_dir_name:
                download_dir = Path(download_dir_name)
                temporary_files = download_threaded(
                    urls_to_download,
                    download_dir,
                    refresh=True,
                    verbose=verbose,
                    n_jobs=n_jobs,
                )

                temporary_files_by_name = {path.name: path for path in temporary_files}

                for file_url, file_path in zip(
                    file_urls_short_list,
                    file_paths_short_list,
                    strict=True,
                ):
                    if file_url not in urls_to_download:
                        continue

                    source_name = PurePosixPath(urlparse(file_url).path).name
                    temporary_file = temporary_files_by_name.get(source_name)

                    if temporary_file is None:
                        failed_file_paths.add(file_path)
                        warnings.warn(
                            f'Skipping unavailable MADIS {source.name.upper()} file: {file_url}',
                            stacklevel=1,
                        )

                        continue

                    temporary_file.replace(file_path)

                    temporary_etag = temporary_file.with_name(temporary_file.name + '.etag')
                    if temporary_etag.is_file():
                        temporary_etag.replace(file_path.with_name(file_path.name + '.etag'))

        # Do not pass absent or failed files to preprocessing.
        file_paths_short_list = [path for path in file_paths_short_list if path not in failed_file_paths and path.is_file()]

        # Preprocess the downloaded files when requested.
        if preprocess:
            file_paths_preprocessed_short_list = local_file_paths_preprocessed(file_paths_short_list)

            dss = extract_station_files(file_paths_short_list, source.extract_file, n_jobs=n_jobs)
            try:
                if len(dss) < len(file_paths_short_list):
                    raise ValueError(
                        f'The number of downloaded original MADIS {source.name} files is greater than the '
                        'number of resulting processed datasets.'
                    )
                if len(dss) > len(file_paths_short_list):
                    raise ValueError(
                        f'The number of downloaded original MADIS {source.name} files is smaller than the '
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
                        # Remove the original MADIS station-data file and any accompanying ETAG.
                        remove_source_file(file_path)
            finally:
                # Release every processed dataset if validation or writing fails.
                for ds in dss:
                    ds.close()

        else:
            # Record the downloaded original MADIS station-data file paths.
            downloaded_files += file_paths_short_list

        # Print progress after completing the batch.
        progress = min(100, 100 * (ii + n_jobs) / len(file_urls))
        print(f'\rProgress: {progress:.2f} %', end='', flush=True)

    # End the progress line at 100 percent.
    print(f'\rProgress: {100:.2f} %')

    # Combine reused files with files produced during this call.
    if preprocess:
        result_files = skip_files_preprocessed + downloaded_files
    else:
        result_files = skip_files + downloaded_files

    return sorted(result_files)


def extract_station_files(
    file_paths: list[Path],
    extract_file: Callable[[Path], xr.Dataset],
    n_jobs: int = 1,
) -> list[xr.Dataset]:
    '''
    Preprocess multiple MADIS station-data files in parallel.

    Parameters
    ----------
    file_paths : list[Path]
        Paths of original station-data files.
    extract_file : Callable[[Path], xr.Dataset]
        Product-specific single-file preprocessing function.
    n_jobs : int, optional
        Number of worker threads. The default is 1.

    Returns
    -------
    list[xr.Dataset]
        Preprocessed datasets in the same order as ``file_paths``.
    '''

    # Collect datasets as workers complete in input order.
    dss = []
    completed = False
    try:
        # Process files concurrently while preserving their order.
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for ds in executor.map(extract_file, file_paths):
                dss.append(ds)
        completed = True
    finally:
        if not completed:
            # Release datasets returned before another worker failed.
            for ds in dss:
                ds.close()

    return dss


def prepare_station_id_keys(
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

    encoded_ids = _pack_station_ids(station_id)

    if encoded_ids is not None and all(value.isascii() for value in requested_station_ids):
        return encoded_ids

    return np.asarray(station_ids2strings(station_id))


def remove_source_file(file_path: Path) -> None:
    '''Remove one original station-data file and its ETag file.'''

    file_path.unlink(missing_ok=True)
    Path(f'{file_path}.etag').unlink(missing_ok=True)


def select_station_records(
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


def _pack_station_ids(station_id: xr.DataArray) -> np.ndarray | None:
    '''Pack a MADIS station-identifier array into fixed-width byte strings.'''

    values = np.asarray(station_id.values)

    if values.ndim not in {1, 2}:
        raise ValueError(f'Unexpected stationId shape: {values.shape}')

    if values.dtype.kind != 'S':
        return None

    if values.ndim == 2:
        if values.shape[1] == 0:
            return np.full(values.shape[0], b'', dtype='S1')

        values = np.ascontiguousarray(values)
        row_width = values.shape[1] * values.dtype.itemsize
        encoded_ids = values.view(f'S{row_width}').reshape(values.shape[0])
    else:
        encoded_ids = values

    return np.char.strip(np.char.replace(encoded_ids, b'\x00', b''))


def _value2string(value: object) -> str:
    '''Convert one identifier value to text while preserving missingness.'''

    if value is None:
        return ''

    try:
        if bool(pd.isna(value)):
            return ''
    except (TypeError, ValueError):
        pass

    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode('utf-8', errors='replace')

    return str(value)
