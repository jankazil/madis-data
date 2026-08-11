'''
Tools for downloading things from the web.
'''

import gzip
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def gzip_file_is_valid(file_path: Path) -> bool:
    '''Check whether a gzip file can be read completely.'''

    try:
        with gzip.open(file_path, 'rb') as compressed_file:
            # Read through the end so gzip verifies the stream trailer and checksum.
            while compressed_file.read(1024 * 1024):
                pass
    except (EOFError, OSError, zlib.error):
        return False

    return True


def head_ok(url: str, timeout: float = 10.0) -> bool:
    """
    Check whether a remote file exists by issuing a lightweight HTTP request.

    This function first attempts an HTTP HEAD request to the given URL. If the
    server does not support HEAD (status 405) or returns certain 4xx errors
    other than 404, it falls back to a streaming GET request to verify
    availability.

    Args:
        url (str): Absolute URL of the resource to check.
        timeout (float): Timeout in seconds for the request. Defaults to 10.0.

    Returns:
        bool: True if the server responds with HTTP 200 (OK), False if the
        request fails, times out, or returns a non-200 status code.
    """
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        # Some servers may not support HEAD well; fall back to GET for 405/403 peculiarities
        if r.status_code == 405 or (400 <= r.status_code < 500 and r.status_code != 404):
            r = requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code == 200
    except requests.RequestException:
        return False


def download_threaded(
    urls: list[str],
    local_dir: Path,
    n_jobs=1,
    refresh: bool = False,
    max_retries: int = 5,
    delay_seconds: int = 3,
    verbose: bool = False,
):
    """
    Downloads a given number of files from given URLs to given local directory, in parallel.

    Args:
        urls (list[str]): List of URLs of files to download
        local_dir (Path): Local directory where the downloaded files will be saved.
        n_jobs (int): Maximum number of parallel downloads
        refresh (bool, optional): If True, download even if the file already exists. Defaults to False.
        verbose (bool): If True, print information. Defaults to False.
    Returns:
        list[Path]: List of local paths of the downloaded files.
    """

    if n_jobs is None:
        n_jobs = 1

    local_file_paths = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(download_file, url, local_dir, refresh, max_retries, delay_seconds, verbose) for url in urls]
        for future in as_completed(futures):
            try:
                local_file_path = future.result()
                local_file_paths.append(local_file_path)
            except Exception as exc:
                print(f"Download generated an exception: {exc}")

    return local_file_paths


def download_file(
    url: str, local_dir: Path, refresh: bool = False, max_retries: int = 5, delay_seconds: int = 3, verbose: bool = False
) -> Path:
    '''
    Downloads a file from a given URL to a given local path.

        Args:
        url (str): URL of file to download
        local_dir (Path): Local directory where files will be downloaded
        refresh (bool, optional): If True, download even if the file already exists. Defaults to False.
                                  When False:
                                  - if the local ETag of the file matches its ETag online, then the file will not be downloaded.
                                  - if the local ETag of the file differs from its ETag online, then the file will be downloaded.
        verbose (bool): If True, print information. Defaults to False.

        Returns:
        Path: Path to the downloaded file.

    '''

    # Local file path
    local_file_path = local_dir / Path(os.path.basename(url))

    # Get ETag with retry
    for attempt in range(max_retries):
        try:
            response = requests.head(url, timeout=10)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f'HEAD request failed ({e}), retrying in {delay_seconds} second(s)...')
                time.sleep(delay_seconds)
            else:
                raise

    etag = response.headers.get('ETag')
    if etag is None:
        message = '\n' + 'ETag not found of file at URL ' + url + '\n' + 'This could mean the file does not exist at this URL.'
        raise Exception(message)

    etag_file_path = local_file_path.with_name(local_file_path.name + '.etag')

    if not refresh and local_file_path.exists() and etag_file_path.exists():
        with open(etag_file_path) as f:
            local_etag = f.read().strip()
        if local_etag == etag:
            gzip_valid = local_file_path.suffix.lower() != '.gz' or gzip_file_is_valid(local_file_path)
            if gzip_valid:
                if verbose:
                    print(
                        url,
                        'available locally as',
                        str(local_file_path),
                        'and ETag matches ETag online. Skipping download.',
                    )
                return local_file_path
            elif verbose:
                print(
                    url,
                    'available locally as',
                    str(local_file_path),
                    'and ETag matches ETag online, but the gzip file is invalid. Proceeding to download.',
                )
        else:
            if verbose:
                print(
                    url,
                    'available locally as',
                    str(local_file_path),
                    'and ETag differs from ETag online. Proceeding to download.',
                )

    # Download with retry
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Reject a truncated gzip file before recording a successful download.
            if local_file_path.suffix.lower() == '.gz' and not gzip_file_is_valid(local_file_path):
                raise requests.exceptions.RequestException(f'Downloaded gzip file is invalid: {url}')

            break
        except requests.exceptions.RequestException as e:
            # Do not leave an incomplete file available for reuse.
            local_file_path.unlink(missing_ok=True)
            if attempt < max_retries - 1:
                if verbose:
                    print(f'Download failed ({e}), retrying in {delay_seconds} second(s)...')
                time.sleep(delay_seconds)
            else:
                raise

    if verbose:
        print('Downloaded', url, 'as', local_file_path)

    with open(etag_file_path, 'w') as f:
        f.write(etag)

    return local_file_path
