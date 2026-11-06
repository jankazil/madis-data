# Agent Development Guide

## Project Scope and Isolation

**The project root is the directory containing this `AGENTS.md` (`.` relative to this file). These instructions apply throughout that repository, including after it is moved.**

- Treat the repository root as the exclusive project boundary. Read, search, analyze, and modify project files only inside it. Run commands from the root unless an in-repository script requires another directory within the root.
- Do not inspect parent directories, sibling projects, unrelated workspaces, global configuration, installed dependency source, or other external locations for project context. Do not import assumptions from other projects or previous sessions.
- Check path containment before following symbolic links, Git worktree/submodule references, object alternates, configuration includes, or paths that may escape through `..`. Do not follow references outside the root.
- External URLs, paths, and credentials mentioned in repository files describe interfaces or dependencies; they do not authorize accessing resources outside the root. Report any required outside-root access instead of performing it.
- Keep development inputs, generated outputs, caches, fixtures, and temporary project files inside the root. Inspect scripts for filesystem, environment, network, or deployment effects before running them. Do not run workflows that cross the project boundary under these instructions.

## General Agent Instructions

- Base project-specific decisions on verified repository contents. Read the files relevant to the task; distinguish implementation facts, documentation claims, test results, and unverified assumptions.
- Before editing, inspect the relevant code and working-tree state. Make focused changes, preserve unrelated work, and retain established interfaces and data contracts unless the task requires otherwise.
- Use relevant checks that are safe in the available environment. Do not confuse a passing placeholder test, build, or lint check with end-to-end or scientific correctness. Report the checks actually run, failures, and verification gaps.
- Do not expose credentials, or perform publishing, deployment, destructive environment setup, or other unrelated side effects as routine development or validation.

## Project Overview and Repository Map

`madis-data` is a Python library and CLI for downloading NOAA MADIS Mesonet and METAR observations, selecting U.S. regions or individual stations, and producing quality-controlled observations, full-hourly UTC NetCDF datasets, and station maps. Supported observables are temperature, dewpoint, U10/V10 wind components, and Mesonet-only solar radiation.

Use [README.md](README.md) for user installation and CLI usage, and [notebooks/HowTo.ipynb](notebooks/HowTo.ipynb) for the METAR API workflow. Inspect notebook source rather than executing its large download workflow by default.

| Location | Responsibility |
| --- | --- |
| `src/madis_data/build_mesonet_dataset.py`, `build_metar_dataset.py` | `arg_parse`, `run_build`, and `main`; installed CLI entry points and output orchestration. |
| `src/madis_data/mesonet.py`, `metar.py` | Product URLs, variable/QC definitions, preprocessing, per-station extraction, hourly interpolation/merging, availability filtering, maps, and station-name lookup. |
| `src/madis_data/stations.py` | Shared `StationDataSource` descriptor, downloads/preprocessing, local paths, station-ID decoding and record selection. |
| `src/madis_data/web.py` | Threaded HTTP transfers, retries, ETags, and gzip integrity checks. |
| `src/madis_data/spatial_filter.py` | Two-pass station catalogue and region/ID selection with station-location consistency checks. |
| `src/madis_data/region_codes.py`, `rto_iso.py` | State/territory, CONUS, and RTO/ISO geometry selection. |
| `src/madis_data/xarray_tools.py`, `progress.py` | Missing-value metadata handling and optional progress display. |
| `src/madis_data/data/` | Packaged Census/EIA geometry and NOAA/UCAR station tables, with provenance documentation. |
| `tests/test_main.py`, `src/madis_data/test.py` | A single arithmetic test and its helper; no scientific workflow coverage. |
| `scripts/`, `Makefile`, `.github/workflows/ci.yml` | Development setup, example downloads, release automation, and CI. |
| `docs/`, `demos/`, `experiments/` | Currently placeholders containing `.gitkeep`. |

## Architecture and Data Contracts

1. Each CLI constructs an inclusive date interval from 00:00 UTC on the start date through 23:00 UTC on the end date, with one extra download day at each boundary.
2. `download_station_data(..., source=...)` delegates product details to `MESONET_DATA_SOURCE` or `METAR_DATA_SOURCE`. It reuses preprocessed files containing required variables unless refreshed, otherwise downloads and preprocesses worker-sized batches.
3. `filter_by_region` or `filter_by_id` produces record-oriented observations plus a station metadata table. `extract_station_data` groups by station and time, retaining the first nonmissing value independently for each variable at duplicate timestamps.
4. Per-station interpolation uses a process pool, restores input station order, and produces float32 observables with dimensions `(time, station)`. `merge_hourly` requires singleton input station dimensions and identical time coordinates (`join='exact'`).
5. Builders write selected observations, station maps, the merged hourly dataset, and station subsets at concurrent-validity fractions 0.500, 0.666, 0.750, 0.800, and 0.900. Validity requires finite temperature, dewpoint, U10, and V10 together; solar radiation is not part of that criterion.

Preserve these established behaviors when changing processing:

- Product differences matter: Mesonet uses `reportTime` and `stationId`; METAR uses `timeObs` and `stationName`, and renames `tempFromTenths`/`dpFromTenths` to temperature/dewpoint. Archive paths differ in both hierarchy and `netCDF`/`netcdf` capitalization.
- Preprocessing keeps screened/verified (`S`/`V`) QC values, rejects wind speeds outside 0–50 m/s, and derives meteorological wind components as `U10 = -speed*sin(direction)` and `V10 = -speed*cos(direction)`. Mesonet solar radiation is listed without a QC flag.
- Preserve NaN normalization, variable attributes, and NetCDF fill/time encodings. Source numeric times are checked against `[1970-01-01, 3000-01-01)` before CF decoding.
- Keep record `stationId` distinct from station-table `stationID` and the hourly `station` coordinate. Decode padded/byte station IDs through the shared helpers; avoid fixed-width truncation during comparisons.
- Region selection excludes stations whose location changes exceed default thresholds of 0.01 degrees latitude/longitude or 25 m elevation. ID selection retains such a station with mean distinct coordinates and `COORDINATE_WARNING`. Missing elevation is allowed.
- Region geometries are transformed to EPSG:4326. Unknown non-two-letter selectors fall back to station IDs; invalid two-letter state/territory codes raise `ValueError`.
- Interpolation requires unique, increasing times; treats naive times as UTC; copies finite exact-hour observations; otherwise needs finite bracketing neighbors separated by at most two hours by default. It does not extrapolate. Do not apply this algorithm unchanged to accumulated quantities.
- Close xarray datasets reliably, including error paths. Lazy spatial results with `load_selected=False` retain source handles until the result is closed; station extraction leaves ownership of its input with the caller.

## Environment and Setup

Verified configuration: `pyproject.toml` uses the `src/` layout, setuptools >=77 and wheel, and requires Python >=3.12. Runtime dependencies include NumPy/pandas/xarray/Dask, netCDF4/h5netcdf, GeoPandas/Shapely/Cartopy/Matplotlib, Requests, python-dateutil, timezonefinder, and tqdm. Development extras provide pytest, pytest-cov, Ruff, and pre-commit. `environment.yml` also includes Jupyter tooling and uses conda-forge.

`setup-dev-env.sh` generates `environment-dev.yml`, removes/recreates the `madis-data-dev` conda environment, replaces its Jupyter kernel, installs the editable package, and installs pre-commit hooks. It is not a read-only prerequisite check. `make clean` removes caches and build outputs. Release targets build and upload packages and must not be used as routine validation.

## Development Commands and Validation

The following commands are documented or implemented in this repository; availability and successful execution are not established by this guide. Dependency installation and environment management require resources outside the project boundary, so report those prerequisites and leave setup to an authorized manual operation.

| Command, from the root | Behavior |
| --- | --- |
| `python -m pip install -e '.[dev]'` | Editable development install, matching CI's installation approach. |
| `make setup-dev-env` | Runs `scripts/setup-dev-env.sh`; see side effects below. |
| `python -m ruff check .` | CI lint check without automatic fixes; add `--no-cache` when caches must not be written. |
| `python -m ruff format --check .` | Check formatting without rewriting source; add `--no-cache` to avoid Ruff cache writes. |
| `make test` or `pytest` | Runs tests selected by `pyproject.toml` under `tests/`. |
| `make fmt` | Rewrites Python formatting. |
| `make lint` | Runs `python -m ruff check --fix`; can modify source. |
| `make check` | Runs formatting and auto-fixing lint; does not run tests. |
| `build-mesonet-dataset --help`, `build-metar-dataset --help` | Installed CLI argument reference. |
| `python -m build` | Builds distributions; used by `scripts/upload2pypi.sh`, requires the separate `build` tool. |

`make type` has no active target; mypy is neither configured nor included in development extras. CI runs Ruff and pytest on Ubuntu with a Python `>=3.12` selector; it does not check formatting or test a fixed multi-version matrix.

## Coding Conventions

- Follow `.editorconfig`: UTF-8, LF, final newline, four-space indentation; YAML/JSON/TOML use two spaces and Make recipes use tabs.
- Follow Ruff settings in `pyproject.toml`: 128-character formatting width, preserved quote style, and lint families E/F/I/UP/B/SIM with the configured exclusions and ignores. Do not reformat excluded notebooks/data incidentally.
- Match nearby snake_case functions, typed signatures, descriptive constants, and NumPy-style docstrings. Existing source uses many explicit processing comments; keep comments focused on behavior and scientific reasoning.
- Put shared station/download logic in `stations.py`/`web.py`; express product-specific operations through `StationDataSource`. Review both product modules and both builders when changing parallel behavior, preserving their deliberate differences.
- Use `pathlib.Path` for paths and `importlib.resources.files`/`as_file` for packaged assets. Keep all shapefile companion files and the package-data declarations in `pyproject.toml` intact.
- Update the local README/notebook or relevant docstrings when changing public APIs, CLI arguments, output schemas, or time semantics. Version metadata comes from `pyproject.toml`; `__init__.py` reads installed distribution metadata with a local fallback.

## Configuration and External Interfaces

- Workflow inputs are function/CLI arguments: dates, spatial selector, data directory, workers, refresh, source removal, and verbosity. No application-specific runtime environment-variable reader was found in the Python source.
- NOAA archive URLs are constructed in the product modules under `https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/`. HTTP downloads require ETags, retry selected transient failures, and validate gzip streams. Missing downloads and certain invalid files can be skipped with warnings; verify completeness instead of assuming every requested hour was retrieved.
- Packaged NOAA and UCAR tables supply station names; Census and EIA files supply geometries. Preserve attribution and consult their in-repository README files before changing or redistributing assets.
- `scripts/Download_MADIS_*.sh` run real downloads with 16 workers and `--remove-original`, writing data under root `data/` and logs under `scripts/logs/`. These are substantial data-producing workflows, not smoke tests.
- PyPI release automation prompts for `TWINE_PASSWORD` and sets `TWINE_USERNAME=__token__`. Anaconda automation uses `ANACONDA_API_TOKEN`, `ANACONDA_USER_NAME`, and optional overrides including `PYPROJECT`, `IMPORT_NAME_OVERRIDE`, `LICENSE_ID`, and build-environment settings. It creates external conda environments and temporary paths. Report these external dependencies; never expose credentials or run uploads as validation.

## Change and Review Workflow

- Before editing, inspect repository status and the affected callers, parallel product implementation, configuration, and local documentation. Keep changes focused and preserve unrelated work.
- For code changes, run applicable non-fixing lint/format checks and pytest when the required environment is available within the authorized scope. If writes are prohibited, account for pytest/bytecode/cache output; report unavailable prerequisites rather than installing dependencies or crossing the boundary.
- For an explicitly authorized integration run, use a small interval and low worker count, keep outputs in the root, and inspect time endpoints, dimensions, station metadata, finite/missing values, NetCDF round trips, maps, and warning/completeness behavior. Large Mesonet runs have substantial RAM, CPU, storage, and network costs.
- Before review, inspect the complete diff and status for accidental data, logs, caches, notebook outputs, or release artifacts. Report checks actually run, skipped checks and reasons, changed public behavior, and remaining uncertainty.
- **Maintain `AGENTS.md`:** When making changes that affect the project's architecture, interfaces, dependencies, workflows, conventions, or documented constraints, update the relevant sections of `AGENTS.md` in the same change. Keep revisions focused and based on verified repository contents. Do not modify `AGENTS.md` for changes that do not affect its accuracy or usefulness.

## Known Limitations and Verification Gaps

- Existing tests only verify `add(2, 2) == 4`; a passing suite does not establish data-processing correctness. For scientific behavior changes, add focused synthetic xarray or mocked HTTP regression tests covering the affected contracts. Avoid live archive downloads in ordinary unit tests.
- **Documentation discrepancy:** README describes the end date as 00:00 UTC; both builders and CLI help implement 23:00 UTC on that date. Confirm intended public semantics before changing either. Example download scripts actually start December 31, 2025 despite their January 2026 descriptions.
- **Compatibility uncertainty:** Python metadata/environment require >=3.12 while Ruff targets `py311`. Most dependencies are unpinned; no dependency lockfile or scientific compatibility matrix is present. Confirm actual supported library/Python versions for compatibility changes.
- **Manual verification:** Environment availability, installation/build success, real NOAA responses, Cartopy map-data/cache requirements, and end-to-end numerical outputs are not established by static inspection. Do not assume plotting is entirely offline or confined to project-local caches.
- **Data terms:** The local UCAR station-table README explicitly leaves applicable redistribution/commercial-use terms unresolved. Preserve that uncertainty and the provider documentation; the software's BSD-3-Clause license does not grant rights to all bundled data.
