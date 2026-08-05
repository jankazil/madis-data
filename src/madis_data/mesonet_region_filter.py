'''
Catalogue-first regional filtering for record-oriented MADIS Mesonet files.

The implementation identifies distinct locations reported under each station
identifier. It uses two explicit phases:

1. Scan station metadata in every file and build one location-aware catalogue.
2. Select regional station locations once, then extract their records from every file.

By default, the first phase retains compact integer station-location codes for
every record. This avoids decoding stationId and rereading coordinates during
extraction. Set retain_record_index=False if that memory/performance tradeoff
is undesirable.

When load_selected=True, each complete input file is loaded sequentially before
regional records are selected in memory. For the tested MADIS files, this is
substantially faster than sparse indexing through the netCDF backend, although
it temporarily uses more memory.
'''

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import xarray as xr

from madis_data import progress, region_codes
from madis_data.mesonet import mesonet_station_ids2strings

STATION_LATITUDE_CHANGE_THRESHOLD_DEG = 0.01
STATION_LONGITUDE_CHANGE_THRESHOLD_DEG = 0.01
STATION_ELEVATION_CHANGE_THRESHOLD_M = 25.0


def filter_by_region(
    madis_mesonet_files: list[Path],
    region_code: str,
    n_jobs: int = 1,
    *,
    retain_record_index: bool = True,
    load_selected: bool = True,
    show_progress: bool = True,
    latitude_change_threshold_deg: float = STATION_LATITUDE_CHANGE_THRESHOLD_DEG,
    longitude_change_threshold_deg: float = STATION_LONGITUDE_CHANGE_THRESHOLD_DEG,
    elevation_change_threshold_m: float = STATION_ELEVATION_CHANGE_THRESHOLD_M,
) -> xr.Dataset:
    '''
    Filter MADIS files using a global station catalogue and regional table.

    The first pass builds a catalogue of unique station locations across all
    files. Stations reported at multiple locations are renamed with ``_1``,
    ``_2``, and subsequent suffixes in first-encounter order. One bounding-box
    and exact polygon test then produces the regional station table. The second
    pass selects every record belonging to those station locations, and per-file
    results are concatenated in input order.

    Parameters
    ----------
    madis_mesonet_files
        Ordered paths of source MADIS Mesonet netCDF files.
    region_code
        State or RTO/ISO code recognized by ``region_codes``.
    n_jobs
        Number of extraction threads. The metadata pass remains ordered and
        sequential so that later files can skip metadata already catalogued.
    retain_record_index
        Whether to retain compact per-record station-location codes between
        passes. The default normally minimizes repeated decoding at a cost of
        approximately two or four bytes per input record.
    load_selected
        If True, load each complete source file sequentially, select regional
        records in memory, and close the source immediately. This is fast for
        the tested MADIS files but temporarily uses up to roughly ``n_jobs``
        uncompressed files of memory. If False, return lazy selected arrays and
        retain source files until the returned dataset is closed.
    show_progress
        Whether to display progress bars for the catalogue and extraction
        passes. tqdm is used when available; otherwise a built-in text bar is
        displayed.
    latitude_change_threshold_deg, longitude_change_threshold_deg
        Maximum coordinate differences, in degrees, assigned to one station
        location. A larger difference creates a new station location.
    elevation_change_threshold_m
        Maximum elevation difference, in meters, assigned to one station
        location when both elevations are finite.

    Returns
    -------
    region_ds
        Dataset containing selected records concatenated in input-file order and
        usable regional station metadata along the ``station`` dimension.
        Elevation may be missing.

    Raises
    ------
    ValueError
        If no files are supplied, ``n_jobs`` is invalid, the region is invalid,
        or input files use inconsistent record dimensions.
    '''

    # Reject an empty source-file sequence before regional preparation.
    if not madis_mesonet_files:
        # Report the missing required inputs.
        raise ValueError('No MADIS Mesonet files were provided.')

    # Reject nonpositive worker counts before creating an executor.
    if n_jobs < 1:
        # Report the valid lower bound.
        raise ValueError('n_jobs must be at least 1.')

    # Reject invalid station-location thresholds before reading source files.
    location_thresholds = (
        latitude_change_threshold_deg,
        longitude_change_threshold_deg,
        elevation_change_threshold_m,
    )
    if not all(np.isfinite(threshold) and threshold >= 0 for threshold in location_thresholds):
        raise ValueError('Station-location thresholds must be finite and nonnegative.')

    # Normalize all supplied path-like values to Path objects.
    madis_mesonet_files = [Path(path) for path in madis_mesonet_files]
    # Count files once for progress reporting.
    file_count = len(madis_mesonet_files)

    # Read, transform, dissolve, and prepare the regional geometry.
    region = _prepare_region(region_code)

    # Initialize the ordered per-file station indexes.
    file_indices: list[_FileStationIndex] = []
    # Initialize the cross-file station-location catalogue.
    location_catalogue = _StationLocationCatalogue(locations=[], codes_by_station={})

    # Add progress reporting around the ordered file sequence.
    catalogue_paths = progress.iterate_with_progress(
        madis_mesonet_files,
        total=file_count,
        description='Scanning station metadata',
        units='files',
        enabled=show_progress,
    )

    # Scan every file in input order so later scans can skip known metadata.
    for path in catalogue_paths:
        # Build the file index while extending the station-location catalogue.
        file_index = _scan_file_station_metadata(
            path,
            location_catalogue=location_catalogue,
            retain_record_index=retain_record_index,
            latitude_change_threshold_deg=latitude_change_threshold_deg,
            longitude_change_threshold_deg=longitude_change_threshold_deg,
            elevation_change_threshold_m=elevation_change_threshold_m,
        )
        # Preserve the per-file index in input order.
        file_indices.append(file_index)

    # Construct final public names after every location of each station is known.
    location_names = _build_station_location_names(location_catalogue)
    # Identify locations for which records with missing elevation are ambiguous.
    elevation_ambiguous = _find_elevation_ambiguous_locations(
        location_catalogue,
        latitude_change_threshold_deg=latitude_change_threshold_deg,
        longitude_change_threshold_deg=longitude_change_threshold_deg,
    )

    # Finalize retained codes after vertical ambiguities are known.
    for file_index in file_indices:
        if file_index.record_location_codes is not None:
            file_index.record_location_codes = _finalize_record_location_codes(
                file_index.record_location_codes,
                elevation_ambiguous,
            )

    # Construct the public catalogue used by the regional geometry test.
    station_catalogue = {
        str(location_names[location_code]): location for location_code, location in enumerate(location_catalogue.locations)
    }

    # Apply bounding-box and exact polygon tests to unique stations only.
    region_station_table = _select_region_station_table(
        station_catalogue,
        region,
    )
    # Map regional membership to compact global station-location codes.
    location_in_region = np.isin(
        location_names,
        region_station_table['stationId'].to_numpy(dtype=str),
    )

    def extract_file(file_index: _FileStationIndex) -> _FilteredFile:
        '''
        Extract one indexed file using the selected regional station set.

        Parameters
        ----------
        file_index
            First-pass index for one source file.

        Returns
        -------
        _FilteredFile
            Regional subset of the indexed file.
        '''

        # Apply the shared regional identifiers to one indexed file.
        return _extract_region_from_file(
            file_index,
            region_code=region_code,
            location_names=location_names,
            location_in_region=location_in_region,
            location_catalogue=location_catalogue,
            elevation_ambiguous=elevation_ambiguous,
            load_selected=load_selected,
            latitude_change_threshold_deg=latitude_change_threshold_deg,
            longitude_change_threshold_deg=longitude_change_threshold_deg,
            elevation_change_threshold_m=elevation_change_threshold_m,
        )

    # Avoid executor overhead when only one worker is requested.
    if n_jobs == 1:
        # Store serial results as they are produced so failures can release them.
        filtered_files: list[_FilteredFile] = []
        extraction_succeeded = False
        try:
            # Create a lazy serial extraction iterator.
            extraction_results = (extract_file(file_index) for file_index in file_indices)
            # Consume results while advancing the extraction progress bar.
            for result in progress.iterate_with_progress(
                extraction_results,
                total=file_count,
                description='Extracting data in region',
                units='files',
                enabled=show_progress,
            ):
                filtered_files.append(result)  # noqa: PERF402
            extraction_succeeded = True
        finally:
            if not load_selected and not extraction_succeeded:
                # Release every lazy source returned before a later file failed.
                _close_datasets(tuple(result.dataset for result in filtered_files))
    # Extract independent files concurrently when multiple workers are allowed.
    else:
        # Import completion-order iteration locally for the concurrent path.
        from concurrent.futures import as_completed

        # Store completed results by input position for ordered concatenation.
        filtered_files_by_position: dict[int, _FilteredFile] = {}
        future_positions = {}
        extraction_succeeded = False
        try:
            # Ensure worker threads are joined before result validation begins.
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Submit every file independently and retain its input position.
                future_positions = {
                    executor.submit(extract_file, file_index): position for position, file_index in enumerate(file_indices)
                }
                # Yield futures as their extraction work finishes.
                completed_futures = as_completed(future_positions)
                # Advance progress in true completion order in the main thread.
                for future in progress.iterate_with_progress(
                    completed_futures,
                    total=file_count,
                    description='Extracting regional records',
                    units='files',
                    enabled=show_progress,
                ):
                    # Recover the completed file's original input position.
                    position = future_positions[future]
                    # Propagate extraction errors and retain successful results.
                    filtered_files_by_position[position] = future.result()

            # Restore input order before validation and concatenation.
            filtered_files = [filtered_files_by_position[position] for position in range(file_count)]
            extraction_succeeded = True
        finally:
            if not load_selected and not extraction_succeeded:
                # Collect each successful lazy result, including unconsumed futures.
                datasets_to_close = {id(result.dataset): result.dataset for result in filtered_files_by_position.values()}
                for future in future_positions:
                    if future.done() and not future.cancelled() and future.exception() is None:
                        result = future.result()
                        datasets_to_close[id(result.dataset)] = result.dataset
                _close_datasets(tuple(datasets_to_close.values()))

    # Collect the record dimension reported by every extracted file.
    record_dimensions = {result.record_dimension for result in filtered_files}

    # Reject files that cannot be concatenated along one common dimension.
    if len(record_dimensions) != 1:
        # Release lazy source files before reporting the structural error.
        if not load_selected:
            # Collect every source dataset created before validation failed.
            inconsistent_sources = tuple(result.dataset for result in filtered_files)
            # Close every retained lazy source dataset.
            _close_datasets(inconsistent_sources)
        # Report every conflicting dimension name.
        raise ValueError(f'Input files do not use one consistent record dimension: {sorted(record_dimensions)!r}.')

    # Retrieve the sole validated record dimension.
    record_dimension = filtered_files[0].record_dimension
    # Retain source datasets for concatenation and optional lazy closing.
    source_datasets = tuple(result.dataset for result in filtered_files)

    concatenation_succeeded = False
    try:
        # Concatenate record variables without unnecessary payload comparison.
        region_ds = xr.concat(
            source_datasets,
            dim=record_dimension,
            data_vars='minimal',
            coords='minimal',
            compat='override',
            join='exact',
            combine_attrs='override',
        )
        concatenation_succeeded = True
    finally:
        if not load_selected and not concatenation_succeeded:
            # Release every lazy source dataset after failed concatenation.
            _close_datasets(source_datasets)

    result_succeeded = False
    try:
        # Attach provenance and the regional station table to the concatenated data.
        region_ds = _attach_region_metadata(
            region_ds,
            file_indices=file_indices,
            source_datasets=source_datasets,
            region_station_table=region_station_table,
            region_code=region_code,
        )

        # Attach a single close callback for every retained lazy source dataset.
        if not load_selected:
            # Ensure that region_ds.close() releases all underlying input files.
            region_ds.set_close(partial(_close_datasets, source_datasets))

        result_succeeded = True
        # Return the regional dataset with usable station metadata.
        return region_ds
    finally:
        if not load_selected and not result_succeeded:
            # Release lazy inputs if any post-concatenation operation fails.
            _close_datasets(source_datasets)


'''
Internal helper classes
'''


@dataclass(frozen=True)
class _PreparedRegion:
    '''
    Store one dissolved region in the coordinate system used by MADIS.

    Attributes
    ----------
    geometry
        Dissolved and prepared Shapely geometry in EPSG:4326.
    min_longitude, min_latitude, max_longitude, max_latitude
        Bounding coordinates used for the inexpensive preliminary test.
    '''

    # Retain the exact dissolved polygon or multipolygon.
    geometry: object
    # Retain the western bounding coordinate.
    min_longitude: float
    # Retain the southern bounding coordinate.
    min_latitude: float
    # Retain the eastern bounding coordinate.
    max_longitude: float
    # Retain the northern bounding coordinate.
    max_latitude: float


@dataclass
class _StationMetadata:
    '''
    Store one representative metadata record for a fixed station location.

    Attributes
    ----------
    longitude, latitude
        Stable station coordinates in EPSG:4326.
    elevation
        Station elevation, or NaN when no finite value has been encountered.
    '''

    # Retain the station longitude.
    longitude: float
    # Retain the station latitude.
    latitude: float
    # Retain the station elevation when it is available.
    elevation: float = np.nan


@dataclass
class _StationLocationCatalogue:
    '''Store global station locations grouped by original station identifier.'''

    # Retain every station location in global integer-code order.
    locations: list[_StationMetadata]
    # Map each original station identifier to its ordered global location codes.
    codes_by_station: dict[str, list[int]]


@dataclass
class _FileStationIndex:
    '''
    Store the compact station index produced for one input file.

    Attributes
    ----------
    path
        Path of the indexed netCDF file.
    record_dimension
        Name of the file's record dimension.
    record_location_codes
        One global station-location code per record, or None in low-memory mode.
        Unusable or ambiguous locations use code -1.
    '''

    # Retain the input file path.
    path: Path
    # Retain the record dimension name.
    record_dimension: str
    # Optionally retain one compact station-location code for every input record.
    record_location_codes: np.ndarray | None


@dataclass
class _FilteredFile:
    '''
    Store the regional subset extracted from one input file.

    Attributes
    ----------
    dataset
        Selected eager or lazy xarray dataset.
    record_dimension
        Name of the dataset's record dimension.
    '''

    # Retain the selected dataset.
    dataset: xr.Dataset
    # Retain the record dimension name for consistency validation.
    record_dimension: str


'''
Internal helper functions
'''


def _attach_region_metadata(
    region_ds: xr.Dataset,
    *,
    file_indices: list['_FileStationIndex'],
    source_datasets: tuple[xr.Dataset, ...],
    region_station_table: pd.DataFrame,
    region_code: str,
) -> xr.Dataset:
    '''Attach source provenance and usable regional station metadata.'''

    # Collect source provenance in input-file order before overriding attributes.
    files_orig: list[str] = []
    for file_index, source_dataset in zip(file_indices, source_datasets, strict=True):
        file_orig = source_dataset.attrs.get('file_orig')
        if file_orig is not None:
            files_orig.append(str(file_orig))
            continue

        inherited_files = source_dataset.attrs.get('files_orig')
        if inherited_files is not None:
            if isinstance(inherited_files, str):
                files_orig.append(inherited_files)
            else:
                files_orig.extend(str(path) for path in inherited_files)
            continue

        files_orig.append(str(file_index.path))

    # Preserve the selected region and complete ordered provenance.
    region_ds.attrs['region'] = region_code
    region_ds.attrs.pop('file_orig', None)
    region_ds.attrs['files_orig'] = files_orig

    # Require finite coordinates while retaining stations with missing elevation.
    usable_location = np.isfinite(region_station_table['longitude'].to_numpy()) & np.isfinite(
        region_station_table['latitude'].to_numpy()
    )
    # Reject stations and their records when no usable location is available.
    station_records = region_station_table.loc[usable_location].copy()

    # Convert sorted station identifiers to the public list.
    station_ids = station_records['stationId'].tolist()
    # Construct the station-to-latitude mapping.
    latitudes = dict(zip(station_records['stationId'], station_records['latitude'], strict=True))
    # Construct the station-to-longitude mapping.
    longitudes = dict(zip(station_records['stationId'], station_records['longitude'], strict=True))
    # Construct the station-to-elevation mapping.
    elevations = dict(zip(station_records['stationId'], station_records['elevation'], strict=True))

    # Store usable station metadata in station-identifier order.
    region_ds['stationID'] = ('station', np.asarray(station_ids, dtype=str))
    region_ds['latitude'] = (
        'station',
        np.asarray([latitudes[station_id] for station_id in station_ids], dtype=np.float32),
    )
    region_ds['longitude'] = (
        'station',
        np.asarray([longitudes[station_id] for station_id in station_ids], dtype=np.float32),
    )
    region_ds['elevation'] = (
        'station',
        np.asarray([elevations[station_id] for station_id in station_ids], dtype=np.float32),
    )

    return region_ds


def _close_datasets(datasets: tuple[xr.Dataset, ...]) -> None:
    '''
    Close every source dataset retained by a lazy concatenated result.

    Parameters
    ----------
    datasets
        Source datasets whose netCDF file handles must be released.
    '''

    # Close every source independently.
    for dataset in datasets:
        # Release the source dataset's backend resources.
        dataset.close()


def _compact_station_codes(station_codes: np.ndarray) -> np.ndarray:
    '''
    Store station codes using the smallest practical signed integer type.

    Approximately 30,000 station locations fit in int16, reducing the retained
    index to two bytes per record while preserving negative provisional codes.

    Parameters
    ----------
    station_codes
        Local station code for every record.

    Returns
    -------
    numpy.ndarray
        Codes converted to int16, int32, or int64 as required.
    '''

    # Determine the code range, or -1 for an empty array.
    minimum_code = int(station_codes.min()) if station_codes.size else -1
    maximum_code = int(station_codes.max()) if station_codes.size else -1

    # Use two-byte codes when the station count permits.
    if minimum_code >= np.iinfo(np.int16).min and maximum_code <= np.iinfo(np.int16).max:
        # Convert without copying when the source already has the target type.
        return station_codes.astype(np.int16, copy=False)

    # Use four-byte codes for larger station catalogues.
    if minimum_code >= np.iinfo(np.int32).min and maximum_code <= np.iinfo(np.int32).max:
        # Convert without copying when the source already has the target type.
        return station_codes.astype(np.int32, copy=False)

    # Retain eight-byte codes only for exceptionally large catalogues.
    return station_codes.astype(np.int64, copy=False)


def _build_station_location_names(
    location_catalogue: _StationLocationCatalogue,
) -> np.ndarray:
    '''Construct unique public names for every station location.'''

    # Allocate names in global location-code order.
    location_names = np.empty(len(location_catalogue.locations), dtype=object)
    assigned_names: set[str] = set()

    for station_id, location_codes in location_catalogue.codes_by_station.items():
        split_station = len(location_codes) > 1

        for location_number, location_code in enumerate(location_codes, start=1):
            location_name = f'{station_id}_{location_number}' if split_station else station_id

            # Reject a generated name that would merge two unrelated stations.
            if location_name in assigned_names:
                raise ValueError(f'Station-location name collision for {location_name!r}.')

            location_names[location_code] = location_name
            assigned_names.add(location_name)

    # Use one fixed-width Unicode array for vectorized record renaming.
    return np.asarray(location_names.tolist(), dtype=str)


def _find_elevation_ambiguous_locations(
    location_catalogue: _StationLocationCatalogue,
    *,
    latitude_change_threshold_deg: float,
    longitude_change_threshold_deg: float,
) -> np.ndarray:
    '''Identify locations that require elevation to distinguish them.'''

    # Initialize every location as horizontally unambiguous.
    elevation_ambiguous = np.zeros(len(location_catalogue.locations), dtype=bool)

    for location_codes in location_catalogue.codes_by_station.values():
        if len(location_codes) < 2:
            continue

        latitudes = np.asarray([location_catalogue.locations[code].latitude for code in location_codes])
        longitudes = np.asarray([location_catalogue.locations[code].longitude for code in location_codes])

        # Find distinct locations whose horizontal tolerance regions overlap.
        horizontal_overlap = (np.abs(latitudes[:, np.newaxis] - latitudes) <= latitude_change_threshold_deg) & (
            np.abs(longitudes[:, np.newaxis] - longitudes) <= longitude_change_threshold_deg
        )
        np.fill_diagonal(horizontal_overlap, False)
        elevation_ambiguous[np.asarray(location_codes)] = horizontal_overlap.any(axis=1)

    return elevation_ambiguous


def _finalize_record_location_codes(
    record_location_codes: np.ndarray,
    elevation_ambiguous: np.ndarray,
) -> np.ndarray:
    '''Resolve provisional codes used for records with missing elevation.'''

    # Copy before replacing provisional negative codes.
    finalized_codes = record_location_codes.copy()
    provisional_positions = np.flatnonzero(finalized_codes < -1)

    if provisional_positions.size:
        # Decode provisional code ``-(location_code + 2)`` using a safe wide type.
        location_codes = -finalized_codes[provisional_positions].astype(np.int64) - 2
        usable_location = ~elevation_ambiguous[location_codes]
        finalized_codes[provisional_positions] = np.where(
            usable_location,
            location_codes,
            -1,
        )

    return _compact_station_codes(finalized_codes)


def _assign_changed_station_records(
    station_id: str,
    station_positions: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    elevations: np.ndarray,
    record_location_codes: np.ndarray,
    *,
    location_catalogue: _StationLocationCatalogue,
    latitude_change_threshold_deg: float,
    longitude_change_threshold_deg: float,
    elevation_change_threshold_m: float,
    allow_new_locations: bool,
) -> None:
    '''Assign records from one changed station to its individual locations.'''

    if allow_new_locations:
        location_codes = location_catalogue.codes_by_station.setdefault(station_id, [])
    else:
        location_codes = location_catalogue.codes_by_station.get(station_id, [])
    remaining_positions = station_positions

    while remaining_positions.size:
        if location_codes:
            reference_latitudes = np.asarray([location_catalogue.locations[code].latitude for code in location_codes])
            reference_longitudes = np.asarray([location_catalogue.locations[code].longitude for code in location_codes])

            # Compare this changed station with all of its known horizontal locations.
            horizontal_match = (
                np.abs(latitudes[remaining_positions, np.newaxis] - reference_latitudes) <= latitude_change_threshold_deg
            ) & (np.abs(longitudes[remaining_positions, np.newaxis] - reference_longitudes) <= longitude_change_threshold_deg)

            if allow_new_locations:
                # Fill a missing reference elevation from its first horizontal match.
                finite_record_elevation = np.isfinite(elevations[remaining_positions])
                for column, location_code in enumerate(location_codes):
                    location = location_catalogue.locations[location_code]
                    if np.isfinite(location.elevation):
                        continue

                    candidates = np.flatnonzero(horizontal_match[:, column] & finite_record_elevation)
                    if candidates.size:
                        location.elevation = float(elevations[remaining_positions[candidates[0]]])

            reference_elevations = np.asarray([location_catalogue.locations[code].elevation for code in location_codes])
            record_elevations = elevations[remaining_positions, np.newaxis]

            # Ignore vertical separation when either elevation is missing.
            vertical_match = (
                ~np.isfinite(record_elevations)
                | ~np.isfinite(reference_elevations)
                | (np.abs(record_elevations - reference_elevations) <= elevation_change_threshold_m)
            )
            location_match = horizontal_match & vertical_match
            matched = location_match.any(axis=1)

            if matched.any():
                # Resolve overlapping tolerances deterministically by first encounter.
                matched_columns = np.argmax(location_match[matched], axis=1)
                matched_codes = np.asarray(location_codes, dtype=np.intp)[matched_columns]
                record_location_codes[remaining_positions[matched]] = matched_codes
                remaining_positions = remaining_positions[~matched]
                continue

        if not allow_new_locations:
            raise ValueError(f'Records for station {station_id!r} no longer match the metadata pass.')

        # Anchor a new location at the first remaining record.
        anchor_position = remaining_positions[0]
        anchor_latitude = float(latitudes[anchor_position])
        anchor_longitude = float(longitudes[anchor_position])
        anchor_horizontal_match = (
            np.abs(latitudes[remaining_positions] - anchor_latitude) <= latitude_change_threshold_deg
        ) & (np.abs(longitudes[remaining_positions] - anchor_longitude) <= longitude_change_threshold_deg)
        elevation_candidates = remaining_positions[anchor_horizontal_match & np.isfinite(elevations[remaining_positions])]
        anchor_elevation = float(elevations[elevation_candidates[0]]) if elevation_candidates.size else np.nan

        # Append the new location and retry vectorized assignment.
        location_code = len(location_catalogue.locations)
        location_catalogue.locations.append(
            _StationMetadata(
                longitude=anchor_longitude,
                latitude=anchor_latitude,
                elevation=anchor_elevation,
            )
        )
        location_codes.append(location_code)


def _assign_station_location_codes(
    station_codes: np.ndarray,
    unique_station_ids: list[str],
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    elevations: np.ndarray,
    *,
    location_catalogue: _StationLocationCatalogue,
    latitude_change_threshold_deg: float,
    longitude_change_threshold_deg: float,
    elevation_change_threshold_m: float,
    allow_new_locations: bool,
) -> np.ndarray:
    '''Assign records through a vectorized stable path and a changed-station fallback.'''

    # Validate one-to-one alignment of identifiers and location variables.
    record_count = station_codes.size
    if any(values.shape != (record_count,) for values in (longitudes, latitudes, elevations)):
        raise ValueError('Station identifiers and location variables must align by record.')

    # Initialize unusable identifiers and coordinates with code -1.
    record_location_codes = np.full(record_count, -1, dtype=np.intp)
    valid_location = (station_codes >= 0) & np.isfinite(longitudes) & np.isfinite(latitudes)

    if not valid_location.any():
        return record_location_codes

    station_count = len(unique_station_ids)
    location_codes_by_station = [location_catalogue.codes_by_station.get(station_id) for station_id in unique_station_ids]

    if allow_new_locations and any(not codes for codes in location_codes_by_station):
        first_locations = _first_valid_record_by_station(
            station_codes,
            valid_location,
            station_count,
        )
        # Add one first-encounter anchor for every previously unseen usable station.
        for station_code, station_id in enumerate(unique_station_ids):
            if location_codes_by_station[station_code]:
                continue

            anchor_position = first_locations[station_code]
            if anchor_position >= record_count:
                continue

            anchor_elevation = float(elevations[anchor_position])
            location_code = len(location_catalogue.locations)
            location_catalogue.locations.append(
                _StationMetadata(
                    longitude=float(longitudes[anchor_position]),
                    latitude=float(latitudes[anchor_position]),
                    elevation=anchor_elevation,
                )
            )
            location_codes = [location_code]
            location_catalogue.codes_by_station[station_id] = location_codes
            location_codes_by_station[station_code] = location_codes

    # Map each local station to its first global location and current split state.
    base_location_codes = np.fromiter(
        (codes[0] if codes else -1 for codes in location_codes_by_station),
        dtype=np.intp,
        count=station_count,
    )
    requires_general_matching = np.fromiter(
        (not codes or len(codes) > 1 for codes in location_codes_by_station),
        dtype=bool,
        count=station_count,
    )
    stations_with_base = base_location_codes >= 0
    reference_latitudes = np.full(station_count, np.nan)
    reference_longitudes = np.full(station_count, np.nan)
    reference_elevations = np.full(station_count, np.nan)

    for station_code in np.flatnonzero(stations_with_base):
        location = location_catalogue.locations[base_location_codes[station_code]]
        reference_latitudes[station_code] = location.latitude
        reference_longitudes[station_code] = location.longitude
        reference_elevations[station_code] = location.elevation

    # Compare all records with their station's first location in compiled NumPy code.
    valid_positions = np.flatnonzero(valid_location)
    valid_station_codes = station_codes[valid_positions]
    horizontal_match = (
        np.abs(latitudes[valid_positions] - reference_latitudes[valid_station_codes]) <= latitude_change_threshold_deg
    ) & (np.abs(longitudes[valid_positions] - reference_longitudes[valid_station_codes]) <= longitude_change_threshold_deg)

    if allow_new_locations:
        # Fill missing base elevations before testing for vertical changes.
        missing_reference_elevation = stations_with_base & ~np.isfinite(reference_elevations)
        if missing_reference_elevation.any():
            base_elevation_candidate = np.zeros(record_count, dtype=bool)
            base_elevation_candidate[valid_positions] = np.isfinite(elevations[valid_positions]) & horizontal_match
            first_elevations = _first_valid_record_by_station(
                station_codes,
                base_elevation_candidate,
                station_count,
            )
            for station_code in np.flatnonzero(missing_reference_elevation):
                elevation_position = first_elevations[station_code]
                if elevation_position >= record_count:
                    continue

                elevation = float(elevations[elevation_position])
                location_code = base_location_codes[station_code]
                location_catalogue.locations[location_code].elevation = elevation
                reference_elevations[station_code] = elevation

    record_elevations = elevations[valid_positions]
    station_elevations = reference_elevations[valid_station_codes]
    vertical_match = (
        ~np.isfinite(record_elevations)
        | ~np.isfinite(station_elevations)
        | (np.abs(record_elevations - station_elevations) <= elevation_change_threshold_m)
    )
    base_match = horizontal_match & vertical_match

    # Escalate only stations containing a record that violates the base location.
    violating_station_codes = valid_station_codes[~base_match]
    requires_general_matching[violating_station_codes] = True
    direct_match = base_match & ~requires_general_matching[valid_station_codes]
    direct_positions = valid_positions[direct_match]
    record_location_codes[direct_positions] = base_location_codes[station_codes[direct_positions]]

    # Sort and group only records belonging to changed or unresolved stations.
    general_positions = valid_positions[requires_general_matching[valid_station_codes]]
    if general_positions.size:
        general_order = np.argsort(station_codes[general_positions], kind='stable')
        grouped_positions = general_positions[general_order]
        grouped_station_codes = station_codes[grouped_positions]
        group_boundaries = np.flatnonzero(np.diff(grouped_station_codes)) + 1

        for station_positions in np.split(grouped_positions, group_boundaries):
            station_code = int(station_codes[station_positions[0]])
            _assign_changed_station_records(
                unique_station_ids[station_code],
                station_positions,
                longitudes,
                latitudes,
                elevations,
                record_location_codes,
                location_catalogue=location_catalogue,
                latitude_change_threshold_deg=latitude_change_threshold_deg,
                longitude_change_threshold_deg=longitude_change_threshold_deg,
                elevation_change_threshold_m=elevation_change_threshold_m,
                allow_new_locations=allow_new_locations,
            )

    # Mark missing-elevation assignments provisionally until all locations are known.
    missing_elevation = (record_location_codes >= 0) & ~np.isfinite(elevations)
    record_location_codes[missing_elevation] = -record_location_codes[missing_elevation] - 2

    return record_location_codes


def _extract_region_from_file(
    file_index: _FileStationIndex,
    *,
    region_code: str,
    location_names: np.ndarray,
    location_in_region: np.ndarray,
    location_catalogue: _StationLocationCatalogue,
    elevation_ambiguous: np.ndarray,
    load_selected: bool,
    latitude_change_threshold_deg: float,
    longitude_change_threshold_deg: float,
    elevation_change_threshold_m: float,
) -> _FilteredFile:
    '''
    Extract records belonging to regional stations from one input file.

    When retained location codes are available, extraction does not decode
    station identifiers or reload location variables. With ``load_selected=True``,
    the complete source file is loaded sequentially and selection then occurs
    in memory. This counterintuitive order avoids slow sparse indexing through
    the tested netCDF backend.

    Parameters
    ----------
    file_index
        First-pass index for the source file.
    region_code
        Region code added to the output attributes.
    location_names
        Public station identifier for every global location code.
    location_in_region
        Regional membership for every global location code.
    location_catalogue
        Complete station-location catalogue built during the metadata pass.
    elevation_ambiguous
        Locations that cannot accept records with missing elevation.
    load_selected
        Whether to return materialized data and close the source immediately.

    Returns
    -------
    _FilteredFile
        Selected dataset and its record dimension.

    Raises
    ------
    ValueError
        If a reloaded station identifier variable has unexpected dimensions.
    '''

    # Open the source without constructing unnecessary default indexes.
    ds = xr.open_dataset(
        file_index.path,
        create_default_indexes=False,
    )

    extraction_succeeded = False
    try:
        # Reconstruct location codes only when low-memory mode omitted them.
        if file_index.record_location_codes is None:
            # Load the identifier variable for low-memory reconstruction.
            station_id_variable = ds['stationId'].load()

            # Validate consistency with the first-pass record dimension.
            if station_id_variable.dims != (file_index.record_dimension,):
                # Report a file whose structure changed between passes.
                raise ValueError(f'Unexpected stationId dimensions in {file_index.path!s}.')

            # Decode and factorize station identifiers a second time.
            station_codes, unique_station_ids = _factorize_station_ids(station_id_variable)

            # Reload record locations only in the low-memory extraction path.
            longitudes = _load_record_array(ds, 'longitude', file_index.record_dimension)
            latitudes = _load_record_array(ds, 'latitude', file_index.record_dimension)
            elevations = _load_record_array(ds, 'elevation', file_index.record_dimension)
            record_location_codes = _assign_station_location_codes(
                station_codes,
                unique_station_ids,
                longitudes,
                latitudes,
                elevations,
                location_catalogue=location_catalogue,
                latitude_change_threshold_deg=latitude_change_threshold_deg,
                longitude_change_threshold_deg=longitude_change_threshold_deg,
                elevation_change_threshold_m=elevation_change_threshold_m,
                allow_new_locations=False,
            )
            record_location_codes = _finalize_record_location_codes(
                record_location_codes,
                elevation_ambiguous,
            )
        # Reuse retained compact location codes in the normal fast mode.
        else:
            record_location_codes = file_index.record_location_codes

        # Map regional station locations to this file's record positions.
        record_positions = _record_positions_for_region(
            record_location_codes,
            location_in_region,
        )
        selected_location_codes = record_location_codes[record_positions].astype(
            np.intp,
            copy=False,
        )

        # Load sequentially before indexing when an eager result is requested.
        if load_selected:
            # Materialize every source variable using efficient sequential reads.
            ds.load()

        # Select regional records, in memory when eager loading was requested.
        ds_out = ds.isel({file_index.record_dimension: record_positions})
        # Replace original identifiers with their location-specific public names.
        station_id_attrs = ds_out['stationId'].attrs.copy()
        ds_out['stationId'] = (
            file_index.record_dimension,
            location_names[selected_location_codes],
        )
        ds_out['stationId'].attrs = station_id_attrs
        # Add the selected region to the per-file global attributes.
        ds_out.attrs['region'] = region_code
        extraction_succeeded = True
    finally:
        if load_selected or not extraction_succeeded:
            # Close eager sources and every source whose extraction failed.
            ds.close()

    # Package the selected dataset and dimension for later validation.
    return _FilteredFile(
        dataset=ds_out,
        record_dimension=file_index.record_dimension,
    )


def _factorize_station_ids(
    station_id_variable: xr.DataArray,
) -> tuple[np.ndarray, list[str]]:
    '''
    Decode station identifiers and replace strings with local integer codes.

    Code -1 represents a missing or empty identifier. Geometry and membership
    operations later use the less expensive integer codes.

    Parameters
    ----------
    station_id_variable
        One-dimensional MADIS station identifier variable.

    Returns
    -------
    station_codes
        One integer station code per record.
    unique_station_ids
        Identifier corresponding to each nonnegative station code.

    Raises
    ------
    ValueError
        If decoding does not produce one identifier per input record.
    '''

    # Decode fixed-width MADIS identifiers once.
    decoded = np.asarray(mesonet_station_ids2strings(station_id_variable))

    # Validate the decoded array against the source record count.
    if decoded.ndim != 1 or decoded.size != station_id_variable.size:
        # Report malformed decoder output.
        raise ValueError('Decoded station identifiers do not form one value per record.')

    # Wrap identifiers for efficient missing-value normalization.
    identifiers = pd.Series(decoded, dtype='object', copy=False)
    # Convert null and empty identifiers to one missing representation.
    identifiers = identifiers.where(identifiers.notna() & identifiers.ne(''))

    # Factorize repeated identifiers into compact local integer codes.
    codes, unique_ids = pd.factorize(identifiers, sort=False)
    # Return integer codes and normalized Python string identifiers.
    return (
        np.asarray(codes, dtype=np.intp),
        [str(value) for value in unique_ids],
    )


def _first_valid_record_by_station(
    station_codes: np.ndarray,
    valid_records: np.ndarray,
    station_count: int,
) -> np.ndarray:
    '''
    Find the first valid record position for every local station code.

    ``numpy.minimum.at`` performs grouping in compiled NumPy code and avoids
    constructing a record-sized DataFrame solely to remove duplicates.

    Parameters
    ----------
    station_codes
        Local station code for every record.
    valid_records
        Boolean mask identifying records eligible for selection.
    station_count
        Number of unique nonmissing stations.

    Returns
    -------
    numpy.ndarray
        First valid record position for each station. Missing stations receive
        the sentinel ``station_codes.size``.
    '''

    # Reserve the record count as the missing-position sentinel.
    missing_position = station_codes.size
    # Initialize every station with the missing-position sentinel.
    first_positions = np.full(
        station_count,
        missing_position,
        dtype=np.intp,
    )
    # Convert the validity mask to sorted integer record positions.
    record_positions = np.flatnonzero(valid_records)

    # Skip grouping when no record is eligible.
    if record_positions.size:
        # Retain the smallest valid position associated with each station code.
        np.minimum.at(
            first_positions,
            station_codes[record_positions],
            record_positions,
        )

    # Return one representative record position per local station.
    return first_positions


def _load_record_array(
    ds: xr.Dataset,
    variable_name: str,
    record_dimension: str,
) -> np.ndarray:
    '''
    Load and validate one one-dimensional record variable.

    Parameters
    ----------
    ds
        Open source dataset.
    variable_name
        Name of the variable to load.
    record_dimension
        Expected sole dimension of the variable.

    Returns
    -------
    numpy.ndarray
        Materialized record values.

    Raises
    ------
    ValueError
        If the variable does not use exactly the expected record dimension.
    '''

    # Retrieve the requested variable without loading unrelated data.
    variable = ds[variable_name]

    # Ensure that values align one-for-one with input records.
    if variable.dims != (record_dimension,):
        # Report the unexpected variable dimensions.
        raise ValueError(f'Expected {variable_name!r} to use only the {record_dimension!r} dimension; found {variable.dims!r}.')

    # Materialize the validated values as a NumPy array.
    return np.asarray(variable.load().values)


def _prepare_region(region_code: str) -> _PreparedRegion:
    '''
    Read, transform, dissolve, and prepare one regional geometry.

    The region is transformed to EPSG:4326, the coordinate reference system
    used by MADIS longitude and latitude. This avoids repeatedly transforming
    tens of thousands of station points.

    Parameters
    ----------
    region_code
        State or RTO/ISO region code recognized by ``region_codes``.

    Returns
    -------
    _PreparedRegion
        Prepared geometry and its bounding coordinates.

    Raises
    ------
    ValueError
        If the region is unavailable, lacks a coordinate reference system, or
        contains no usable geometry.
    '''

    # Retrieve the requested region from the host project's registry.
    region_gdf = region_codes.get_region_gdf(region_code)

    # Reject an unknown region code explicitly.
    if region_gdf is None:
        # Report the unavailable region code.
        raise ValueError(f'Region code {region_code!r} is not available.')

    # Reject geometry whose coordinate system cannot be interpreted.
    if region_gdf.crs is None:
        # Report the missing coordinate reference system.
        raise ValueError(f'Region {region_code!r} has no coordinate reference system.')

    # Transform the regional geometry once to the MADIS coordinate system.
    region_wgs84 = region_gdf.to_crs('EPSG:4326')

    # Use the current GeoPandas union method when it is available.
    if hasattr(region_wgs84.geometry, 'union_all'):
        # Dissolve all component polygons into one geometry.
        geometry = region_wgs84.geometry.union_all()
    # Support older GeoPandas versions through their legacy union property.
    else:
        # Dissolve all component polygons into one geometry.
        geometry = region_wgs84.geometry.unary_union

    # Reject an absent or empty dissolved result.
    if geometry is None or geometry.is_empty:
        # Report that no valid regional area is available.
        raise ValueError(f'Region {region_code!r} has no usable geometry.')

    # Build Shapely's internal segment index for repeated point predicates.
    shapely.prepare(geometry)
    # Extract the inexpensive preliminary bounding box.
    min_lon, min_lat, max_lon, max_lat = geometry.bounds

    # Return the prepared polygon and numeric bounding coordinates.
    return _PreparedRegion(
        geometry=geometry,
        min_longitude=float(min_lon),
        min_latitude=float(min_lat),
        max_longitude=float(max_lon),
        max_latitude=float(max_lat),
    )


def _record_positions_for_region(
    record_location_codes: np.ndarray,
    location_in_region: np.ndarray,
) -> np.ndarray:
    '''
    Map regional station-location codes back to sorted record positions.

    Parameters
    ----------
    record_location_codes
        Global station-location code for every record in one file.
    location_in_region
        Regional membership for every global station-location code.

    Returns
    -------
    numpy.ndarray
        Sorted integer positions of records belonging to regional stations.
    '''

    # Identify records with usable station locations.
    valid_location = record_location_codes >= 0
    # Initialize the record-level regional membership mask.
    record_mask = np.zeros(record_location_codes.size, dtype=bool)
    # Map valid record codes through the compact location-level membership array.
    record_mask[valid_location] = location_in_region[record_location_codes[valid_location]]

    # Return sorted integer positions rather than a record-sized Boolean index.
    return np.flatnonzero(record_mask)


def _scan_file_station_metadata(
    madis_mesonet_file: Path,
    *,
    location_catalogue: _StationLocationCatalogue,
    retain_record_index: bool,
    latitude_change_threshold_deg: float,
    longitude_change_threshold_deg: float,
    elevation_change_threshold_m: float,
) -> _FileStationIndex:
    '''
    Scan one file and assign every usable record to a fixed station location.

    Station identifiers and the three location variables are loaded from every
    file. Records whose latitude or longitude is missing receive code -1.

    Parameters
    ----------
    madis_mesonet_file
        Source netCDF file.
    location_catalogue
        Global location catalogue extended in first-encounter order.
    retain_record_index
        Whether to retain compact per-record location codes for extraction.
    latitude_change_threshold_deg, longitude_change_threshold_deg
        Maximum coordinate differences assigned to one location.
    elevation_change_threshold_m
        Maximum finite elevation difference assigned to one location.

    Returns
    -------
    _FileStationIndex
        Per-file mapping from records to global station-location codes.

    Raises
    ------
    ValueError
        If required record variables do not have the expected dimensions.
    KeyError
        If a required variable is absent.
    '''

    # Open the source without constructing unnecessary default indexes.
    ds = xr.open_dataset(
        madis_mesonet_file,
        create_default_indexes=False,
    )

    # Ensure that the input file closes after metadata have been copied.
    try:
        # Load station identifiers without loading the complete dataset.
        station_id_variable = ds['stationId'].load()

        # Require exactly one record dimension.
        if station_id_variable.ndim != 1:
            # Report an incompatible station identifier layout.
            raise ValueError('Expected stationId to have exactly one record dimension.')

        # Record the sole record dimension name.
        record_dimension = station_id_variable.dims[0]
        # Decode identifiers and construct one local code per record.
        station_codes, unique_station_ids = _factorize_station_ids(station_id_variable)

        # Load every location variable required to identify station moves.
        longitudes = _load_record_array(ds, 'longitude', record_dimension)
        latitudes = _load_record_array(ds, 'latitude', record_dimension)
        elevations = _load_record_array(ds, 'elevation', record_dimension)

        # Assign global location codes, using the fast path for unchanged stations.
        record_location_codes = _assign_station_location_codes(
            station_codes,
            unique_station_ids,
            longitudes,
            latitudes,
            elevations,
            location_catalogue=location_catalogue,
            latitude_change_threshold_deg=latitude_change_threshold_deg,
            longitude_change_threshold_deg=longitude_change_threshold_deg,
            elevation_change_threshold_m=elevation_change_threshold_m,
            allow_new_locations=True,
        )
    # Release the netCDF handle even when validation or decoding fails.
    finally:
        # Close the metadata-pass source dataset.
        ds.close()

    # Package the record mapping required by the extraction pass.
    file_index = _FileStationIndex(
        path=madis_mesonet_file,
        record_dimension=record_dimension,
        record_location_codes=(_compact_station_codes(record_location_codes) if retain_record_index else None),
    )

    return file_index


def _select_region_station_table(
    station_catalogue: dict[str, _StationMetadata],
    region: _PreparedRegion,
) -> pd.DataFrame:
    '''
    Select catalogue stations that intersect the requested region.

    An inexpensive bounding-box test first removes stations that cannot
    intersect the state or RTO/ISO. ``shapely.intersects_xy`` then performs the
    exact boundary-inclusive polygon test without constructing station Point
    objects or a GeoDataFrame.

    Parameters
    ----------
    station_catalogue
        Unique station-location metadata accumulated from all input files.
    region
        Prepared regional geometry and bounding coordinates.

    Returns
    -------
    pandas.DataFrame
        Sorted regional station table containing station identifier, longitude,
        latitude, and elevation.
    '''

    # Define a stable schema for both empty and nonempty results.
    columns = ['stationId', 'longitude', 'latitude', 'elevation']

    # Return an empty table immediately when the catalogue has no stations.
    if not station_catalogue:
        # Preserve the expected result column names.
        return pd.DataFrame(columns=columns)

    # Preserve catalogue insertion order while creating identifier values.
    station_ids = np.asarray(list(station_catalogue), dtype=object)
    # Materialize catalogue metadata once for compiled array construction.
    metadata = list(station_catalogue.values())
    # Construct a contiguous longitude array.
    longitudes = np.fromiter(
        (item.longitude for item in metadata),
        dtype=np.float64,
        count=len(metadata),
    )
    # Construct a contiguous latitude array.
    latitudes = np.fromiter(
        (item.latitude for item in metadata),
        dtype=np.float64,
        count=len(metadata),
    )
    # Construct a contiguous elevation array.
    elevations = np.fromiter(
        (item.elevation for item in metadata),
        dtype=np.float64,
        count=len(metadata),
    )

    # Apply the inexpensive rectangular preliminary filter.
    in_bbox = (
        (longitudes >= region.min_longitude)
        & (longitudes <= region.max_longitude)
        & (latitudes >= region.min_latitude)
        & (latitudes <= region.max_latitude)
    )
    # Convert bounding-box matches to integer catalogue positions.
    candidate_positions = np.flatnonzero(in_bbox)
    # Initialize exact region membership for all catalogue stations.
    in_region = np.zeros(len(station_ids), dtype=bool)

    # Avoid the exact spatial predicate when no station enters the bounding box.
    if candidate_positions.size:
        # Mark candidates that intersect the exact boundary-inclusive geometry.
        in_region[candidate_positions] = np.asarray(
            shapely.intersects_xy(
                region.geometry,
                longitudes[candidate_positions],
                latitudes[candidate_positions],
            ),
            dtype=bool,
        )

    # Construct, sort, and return the regional station table.
    return (
        pd.DataFrame(
            {
                'stationId': station_ids[in_region],
                'longitude': longitudes[in_region],
                'latitude': latitudes[in_region],
                'elevation': elevations[in_region],
            }
        )
        .sort_values('stationId')
        .reset_index(drop=True)
    )
