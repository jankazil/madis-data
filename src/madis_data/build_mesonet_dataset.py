#!/usr/bin/env python

'''
Build Meteorological Assimilation Data Ingest System (MADIS) Mesonet datasets
for a specified U.S. state, territory, RTO/ISO region, CONUS, or individual
station. The module automates downloading, preprocessing and optionally removing
the original downloaded MADIS Mesonet files to reduce storage requirements, filtering
observations by region or station identifier, constructing full-hourly UTC time
series, saving NetCDF results, and plotting station locations. It can be executed
as a command-line tool or used programmatically through the ``run_build()`` function.

Workflow:

1) Construct the requested start and end timestamps at 00:00 UTC and extend the
   download interval by one day at each end for interpolation support.
2) Download MADIS Mesonet files and preprocess them. Existing files
   are reused unless refresh is requested.
3) Load the requested region geometry and filter observations spatially, or
   extract one station when a station identifier is supplied.
4) Save the record-oriented observations for the selected region or station.
5) Plot the locations of the selected stations on a contiguous United States
   map.
6) Extract a time-dependent dataset for each station and interpolate the data
   to full-hourly UTC timestamps over the requested interval.
7) Merge the station time series and save them as a NetCDF file.

Output files:

- Downloaded MADIS Mesonet files and, by default, their preprocessed counterparts.
- A NetCDF file containing record-oriented observations for the region or
  station.
- A map showing the selected station locations.
- A NetCDF file containing the merged full-hourly UTC station time series.

Assumptions:

- Network access to the MADIS archive is available.
- A spatial identifier that is not recognized as a region code is treated as a station ID.

Example usage:

    build-mesonet-dataset 2024 1 1 2024 1 31 CO /path/to/data -n 8
    build-mesonet-dataset 2024 6 1 2024 6 30 ERCOT /path/to/data --remove-original
    build-mesonet-dataset 2024 1 1 2024 12 31 SMPC2 /path/to/data -v
'''

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from madis_data import region_codes
from madis_data.mesonet import (
    download_mesonet,
    extract_station_data,
    interpolate_to_full_hour,
    merge_hourly,
    plot_locations_conus,
)
from madis_data.mesonet_spatial_filter import filter_by_id, filter_by_region


def run_build(
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
    spatial_identifier: str,
    data_dir: Path,
    n_jobs: int = 1,
    remove_original: bool = False,
    refresh: bool = False,
    verbose: bool = False,
) -> tuple[Path, Path, Path]:
    '''
    Download, filter, and assemble MADIS Mesonet data.

    The requested date interval begins and ends at 00:00 UTC. Files from one
    additional day before and after the interval are downloaded so observations
    can bracket its boundary hours during interpolation.

    Parameters
    ----------
    start_year, start_month, start_day
        Components of the first requested timestamp.
    end_year, end_month, end_day
        Components of the last requested timestamp.
    spatial_identifier
        Spatial identifier, such as a U.S. state or territory code, ``CONUS``, an
        RTO/ISO code, or a MADIS Mesonet station identifier.
    data_dir
        Directory where downloaded data and generated outputs are stored.
    n_jobs
        Number of parallel workers used for downloading, spatial extraction,
        and interpolation. The default is 1.
    remove_original
        If True, remove downloaded original MADIS Mesonet files after successful preprocessing.
        The default is False.
    refresh
        If True, download files even when corresponding local files exist. The
        default is False.
    verbose
        If True, display detailed progress information. The default is False.

    Returns
    -------
    region_out_file, region_out_file_hourly, region_map_plot
        Paths to the record-oriented NetCDF file, full-hourly NetCDF file, and
        station-location plot.

    Raises
    ------
    ValueError
        If the end timestamp precedes the start timestamp.
    '''

    # Create the data directory unless it exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Construct timestamps at the beginning of the requested dates.
    start_date = datetime(
        year=start_year,
        month=start_month,
        day=start_day,
        hour=0,
        minute=0,
        second=0,
    )
    end_date = datetime(
        year=end_year,
        month=end_month,
        day=end_day,
        hour=23,
        minute=0,
        second=0,
    )

    # Reject a reversed time interval before downloading data.
    if end_date < start_date:
        raise ValueError('The end date must not precede the start date.')

    # Construct a common date tag for generated filenames.
    file_tag = f'{start_date.date()}_{end_date.date()}'

    # Extend the download interval so observations can bracket both boundary hours.
    data_start_date = start_date - timedelta(days=1)
    data_end_date = end_date + timedelta(days=1)

    # Download and preprocess the required MADIS Mesonet files.
    mesonet_files = download_mesonet(
        data_start_date,
        data_end_date,
        data_dir,
        refresh=refresh,
        remove_original=remove_original,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    # Determine whether the spatial identifier identifies a supported region or one station.
    region_gdf = region_codes.get_region_gdf(spatial_identifier)

    # Extract observations for the requested region or station.
    if region_gdf is None:
        # Treat an unrecognized spatial identifier as a station identifier.
        ds_region = filter_by_id(
            mesonet_files,
            spatial_identifier,
            n_jobs=n_jobs,
            show_progress=verbose,
        )
    else:
        # Filter records using the geometry associated with the spatial identifier.
        ds_region = filter_by_region(
            mesonet_files,
            spatial_identifier,
            n_jobs=n_jobs,
            show_progress=verbose,
        )

    # Define output files paths.
    region_out_file = data_dir / f'{spatial_identifier}.{file_tag}.nc'
    region_map_plot = data_dir / f'{spatial_identifier}.{file_tag}_stations.png'

    try:
        # Save the observations selected for the region or station.
        ds_region.to_netcdf(region_out_file)

        # Construct list of station identifiers
        station_ids = [str(station_id) for station_id in ds_region['stationID'].values]

        # Plot the selected stations on a contiguous United States map.
        plot_locations_conus(
            station_ids,
            list(ds_region['latitude'].values),
            list(ds_region['longitude'].values),
            region_map_plot,
            verbose=verbose,
        )

        # Construct a dictionary with one dataset per selected station.
        ds_station_dict = extract_station_data(
            ds_region,
            station_ids,
            show_progress=verbose,
        )
    finally:
        # Release any resources retained by the spatially filtered dataset.
        ds_region.close()

    # Create full-hourly time series by interpolation of the data for each station
    ds_station_hourly_dict = interpolate_to_full_hour(
        ds_station_dict,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
        max_workers=n_jobs,
    )

    # Merge single-station full-hourly time series into one dataset
    ds_hourly = merge_hourly(ds_station_hourly_dict)

    # Save the merged full-hourly station time series.
    region_out_file_hourly = data_dir / f'{spatial_identifier}.{file_tag}.hourly.nc'
    try:
        ds_hourly.to_netcdf(region_out_file_hourly, engine='h5netcdf', mode='w')
    finally:
        # Release arrays and backend resources associated with the merged dataset.
        ds_hourly.close()

    return region_out_file, region_out_file_hourly, region_map_plot


def arg_parse(
    argv: list[str] | None = None,
) -> tuple[int, int, int, int, int, int, str, Path, int, bool, bool, bool, bool]:
    '''
    Parse command-line arguments for the Mesonet dataset builder.

    Parameters
    ----------
    argv
        Argument tokens excluding the program name. If None, arguments are read
        from ``sys.argv[1:]``.

    Returns
    -------
    tuple
        Parsed start and end date components, spatial identifier, data
        directory, worker count, source-removal flag, refresh flag, and
        verbosity flag.

    Raises
    ------
    SystemExit
        If argparse rejects the supplied arguments.
    '''

    code_description = (
        'Download and preprocess MADIS Mesonet observations for a selected U.S. '
        'state, territory, the contiguous United States (CONUS), an RTO/ISO region, or one '
        'station identifier. The script optionally removes the original downloaded MADIS Mesonet data files, '
        'plots station locations, constructs full-hourly UTC time series, and writes the '
        'merged hourly dataset as a NetCDF file.\n\n'
        'The start and end dates represent 00:00 UTC. One additional day is downloaded before '
        'and after the requested interval to support interpolation at its boundaries.\n\n'
        'Existing downloaded or preprocessed files are reused unless --refresh is specified.\n\n'
        '\n\n'
        "Valid region or station arguments:\n\n"
        f"  - US states/territories: {', '.join(region_codes.us_states_territories)}\n\n"
        f"  - Special region: {region_codes.conus}\n\n"
        f"  - RTO/ISO regions: {', '.join(region_codes.rto_iso_regions)}\n\n"
        "  - Individual station: provide a station ID from the public MADIS station list available at\n"
        '    https://madis-data.bldr.ncep.noaa.gov/madisPublic1/data/stations/public_stntbl.csv'
    )

    parser = argparse.ArgumentParser(
        description=code_description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mandatory date, spatial identifier, and output-directory arguments.
    parser.add_argument('start_year', type=int, help='Start year.')
    parser.add_argument('start_month', type=int, help='Start month.')
    parser.add_argument('start_day', type=int, help='Start day.')
    parser.add_argument('end_year', type=int, help='End year.')
    parser.add_argument('end_month', type=int, help='End month.')
    parser.add_argument('end_day', type=int, help='End day.')
    parser.add_argument(
        'spatial_identifier',
        type=str,
        help=(
            'Region or station identifier: a U.S. state or territory code, CONUS, an RTO/ISO '
            'code, or a MADIS Mesonet station identifier.'
        ),
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory where downloaded data and generated outputs are stored.',
    )

    # Optional execution and file-handling arguments.
    parser.add_argument(
        '-n',
        '--n',
        type=int,
        default=1,
        help=(
            'Number of parallel workers. Values greater than one can accelerate downloads '
            'and processing but may increase memory use or trigger server errors.'
        ),
    )
    parser.add_argument(
        '--remove-original',
        action='store_true',
        help='Remove downloaded original MADIS Mesonet files after successful preprocessing.',
    )
    parser.add_argument(
        '-r',
        '--refresh',
        action='store_true',
        help='Download files even when corresponding local files already exist.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Print detailed progress information.',
    )

    args = parser.parse_args(argv)

    return (
        args.start_year,
        args.start_month,
        args.start_day,
        args.end_year,
        args.end_month,
        args.end_day,
        args.spatial_identifier,
        Path(args.data_dir),
        args.n,
        args.remove_original,
        args.refresh,
        args.verbose,
    )


def main(argv: list[str] | None = None) -> None:
    '''
    Run the command-line Mesonet dataset-building workflow.
    '''

    (
        start_year,
        start_month,
        start_day,
        end_year,
        end_month,
        end_day,
        spatial_identifier,
        data_dir,
        n_jobs,
        remove_original,
        refresh,
        verbose,
    ) = arg_parse(argv if argv is not None else sys.argv[1:])

    region_out_file, region_out_file_hourly, region_map_plot = run_build(
        start_year,
        start_month,
        start_day,
        end_year,
        end_month,
        end_day,
        spatial_identifier,
        data_dir,
        n_jobs=n_jobs,
        remove_original=remove_original,
        refresh=refresh,
        verbose=verbose,
    )

    print()
    print('Created the following files:')
    print()
    print(region_out_file)
    print(region_out_file_hourly)
    print(region_map_plot)


if __name__ == '__main__':
    main()
