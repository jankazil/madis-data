#!/usr/bin/env python

'''
Build Meteorological Assimilation Data Ingest System (MADIS) Mesonet datasets.

The requested observations may cover a U.S. state or territory, a regional
transmission organization or independent system operator (RTO/ISO) region, the
contiguous United States (CONUS), or one station. The module downloads and
preprocesses MADIS Mesonet files, filters observations spatially or by station
identifier, constructs full-hourly Coordinated Universal Time (UTC) series,
filters stations by data availability, writes NetCDF datasets, and plots station
locations. Original downloaded files can optionally be removed after successful
preprocessing.

The module can be executed as a command-line program or used programmatically
through ``run_build()``.

Workflow:

1) Construct the requested interval from 00:00 UTC on the start date through
   23:00 UTC on the end date. Extend the download interval by one day at each
   end to support interpolation at the requested boundaries.
2) Download and preprocess MADIS Mesonet files. Existing files are reused unless
   refresh is requested.
3) Load the requested region geometry and filter observations spatially, or
   extract one station when a station identifier is supplied.
4) Save the observations for the selected region or station.
5) Plot the locations of the selected stations on a contiguous United States
   map.
6) Extract a time-dependent dataset for each station, interpolate observations
   to full-hourly UTC timestamps, and merge the station time series.
7) For each configured validity threshold, retain stations meeting the minimum
   fraction of times at which temperature, dewpoint, and both 10 m wind
   components are concurrently valid, and save the filtered dataset.
8) Save the unfiltered merged full-hourly dataset.

Output files:

- Downloaded MADIS Mesonet files and their preprocessed counterparts. Original
  downloaded files are retained by default.
- A NetCDF file containing observations for the region or station.
- A map showing the selected station locations.
- Five NetCDF files containing full-hourly station time series filtered at the
  configured concurrent-validity thresholds.
- A NetCDF file containing the unfiltered merged full-hourly UTC station time
  series.

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
    MESONET_DATA_SOURCE,
    extract_station_data,
    filter_by_valid_fraction,
    interpolate_to_full_hour,
    merge_hourly,
    plot_locations_conus,
)
from madis_data.spatial_filter import filter_by_id, filter_by_region
from madis_data.stations import download_station_data


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
) -> None:
    '''
    Download, filter, and assemble MADIS Mesonet data.

    The requested interval runs from 00:00 UTC on the start date through 23:00
    UTC on the end date. Files from one additional day before and after the
    interval are downloaded so observations can bracket its boundary hours
    during interpolation.

    The function writes the spatially selected observations, a
    station-location map, the unfiltered merged hourly dataset, and hourly
    datasets filtered at concurrent-validity fractions of 0.500, 0.666, 0.750,
    0.800, and 0.900. At a given station and time, an observation is concurrently
    valid only when temperature, dewpoint, U10, and V10 are all finite.

    Parameters
    ----------
    start_year, start_month, start_day
        Calendar components of the first requested date. Its first included
        timestamp is 00:00 UTC.
    end_year, end_month, end_day
        Calendar components of the last requested date. Its last included
        timestamp is 23:00 UTC.
    spatial_identifier
        Spatial identifier, such as a U.S. state or territory code, ``CONUS``, an
        RTO/ISO code, or a MADIS Mesonet station identifier.
    data_dir
        Directory in which downloaded data, intermediate files, and generated
        outputs are stored. It is created if necessary.
    n_jobs
        Number of parallel workers used for downloading, spatial extraction,
        and interpolation. The default is 1.
    remove_original
        If True, remove original downloaded MADIS Mesonet files after successful
        preprocessing. The default is False.
    refresh
        If True, download files even when corresponding local files exist. The
        default is False.
    verbose
        If True, display detailed progress information. The default is False.

    Raises
    ------
    ValueError
        If the end timestamp precedes the start timestamp.

    Returns
    -------
    None
        Results are written to ``data_dir``.
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
    mesonet_files = download_station_data(
        data_start_date,
        data_end_date,
        data_dir,
        source=MESONET_DATA_SOURCE,
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
        # Filter data using the geometry associated with the spatial identifier.
        ds_region = filter_by_region(
            mesonet_files,
            spatial_identifier,
            n_jobs=n_jobs,
            show_progress=verbose,
        )

    # Define paths for the regional data file and station-location map.
    region_out_file = data_dir / f'{spatial_identifier}.{file_tag}.nc'
    region_map_plot = data_dir / f'{spatial_identifier}.{file_tag}_stations.png'

    try:
        # Save the observations selected for the region or station.
        ds_region.to_netcdf(region_out_file)
        print('Created', region_out_file)

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

        print('Created', region_map_plot)

        # Construct a dictionary with one dataset per selected station.
        ds_station_dict = extract_station_data(
            ds_region,
            station_ids,
            show_progress=verbose,
        )
    finally:
        # Release any resources retained by the spatially filtered dataset.
        ds_region.close()
        del ds_region

    # Interpolate each station dataset to full-hourly timestamps in the requested interval.
    ds_station_hourly_dict = interpolate_to_full_hour(
        ds_station_dict,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
        max_workers=n_jobs,
    )

    # Merge the single-station full-hourly time series into one dataset.
    ds_hourly = merge_hourly(ds_station_hourly_dict)

    # Release resources
    del ds_station_dict

    # Construct hourly datasets containing only stations that meet each concurrent-validity threshold.

    for min_valid_fraction in [0.5, 0.666, 0.75, 0.8, 0.9]:
        ds_valid_fraction = filter_by_valid_fraction(ds_hourly, ['temperature', 'dewpoint', 'U10', 'V10'], min_valid_fraction)
        try:
            # Save the data for the region or station.
            region_out_file_hourly = data_dir / f'{spatial_identifier}.{file_tag}.hourly.{min_valid_fraction:.3f}.nc'
            ds_valid_fraction.to_netcdf(region_out_file_hourly, engine='h5netcdf', mode='w')
            print('Created', region_out_file_hourly)
            # Construct list of station identifiers
            station_ids = [str(station_id) for station_id in ds_valid_fraction['station'].values]
            # Plot the selected stations on a contiguous United States map.
            region_map_plot = data_dir / f'{spatial_identifier}.{file_tag}.hourly.{min_valid_fraction:.3f}.png'
            plot_locations_conus(
                station_ids,
                list(ds_valid_fraction['LAT'].values),
                list(ds_valid_fraction['LON'].values),
                region_map_plot,
                verbose=verbose,
            )
            print('Created', region_map_plot)
        finally:
            ds_valid_fraction.close()
            # Release resources
            del ds_valid_fraction

    # Save the merged full-hourly station time series.
    region_out_file_hourly = data_dir / f'{spatial_identifier}.{file_tag}.hourly.nc'
    try:
        ds_hourly.to_netcdf(region_out_file_hourly, engine='h5netcdf', mode='w')
        print('Created', region_out_file_hourly)
    finally:
        # Release arrays and backend resources associated with the merged dataset.
        ds_hourly.close()
        # Release resources
        del ds_hourly

    return


def arg_parse(
    argv: list[str] | None = None,
) -> tuple[int, int, int, int, int, int, str, Path, int, bool, bool, bool]:
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
        Parsed start-year, start-month, start-day, end-year, end-month, end-day,
        spatial identifier, data directory, worker count, source-removal flag,
        refresh flag, and verbosity flag, in that order.

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
        'merged hourly datasets as NetCDF files.\n\n'
        'The requested interval runs from 00:00 UTC on the start date through 23:00 UTC '
        'on the end date. One additional day is downloaded before and after the requested '
        'interval to support interpolation at its boundaries.\n\n'
        'Existing downloaded or preprocessed files are reused unless --refresh is specified.\n\n'
        '\n\n'
        'Valid region or station arguments:\n\n'
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

    Parameters
    ----------
    argv
        Argument tokens excluding the program name. If None, arguments are read
        from ``sys.argv[1:]``.

    Returns
    -------
    None
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

    _ = run_build(
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


if __name__ == '__main__':
    main()
