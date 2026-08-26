#!/usr/bin/env bash

########################################################################
# Download MADIS Metar surface observation data files from the MADIS
# archive and build full-hourly datasets for Colorado for January 2026.
#
# Requirements:
#
# - Python environment with the madis-data package
#
########################################################################

#
# Set up error logging
#

# Exit on errors, unset variables, and failed pipeline commands.
set -Eeuo pipefail

# Locate the script and change to its containing directory.
script_path="${BASH_SOURCE[0]}"
script_parent_dir="$(dirname -- "$script_path")"

cd -- "$script_parent_dir"
script_dir="$(pwd)"

# Create a directory for run logs and status files.
log_dir="$script_dir/logs"
mkdir -p -- "$log_dir"

# Generate unique filenames using the current UTC time.
run_id=$(date -u '+%Y%m%dT%H%M%SZ')
run_log="$log_dir/${0%*.sh}-$run_id.log"
status_file="$log_dir/${0%*.sh}-$run_id.status"

# Disconnect standard input and send all output to the run log.
exec </dev/null >>"$run_log" 2>&1

# Include the source file, line, and function in trace output.
export PS4='+ ${BASH_SOURCE}:${LINENO}:${FUNCNAME[0]:-main}: '

# Ignore hangup signals caused by terminal disconnection.
trap '' HUP

# Record the final exit status when the script terminates.
trap '
    rc=$?
    trap - EXIT
    printf "EXIT status=%d time=%s\n" "$rc" "$(date -Is)"
    printf "%d\n" "$rc" >"$status_file"
    exit "$rc"
' EXIT

# Report the command and line responsible for an error.
trap '
    rc=$?
    printf "ERROR status=%d line=%d command=%s\n" \
        "$rc" "$LINENO" "$BASH_COMMAND" >&2
' ERR

# Convert termination signals to conventional exit statuses.
trap 'printf "Received SIGTERM\n" >&2; exit 143' TERM
trap 'printf "Received SIGINT\n" >&2; exit 130' INT

# Log each command before executing it.
set -x

#
# Do your worst
#

REGION=CO

START_YEAR=2025
START_MONTH=12
START_DAY=31
END_YEAR=2026
END_MONTH=1
END_DAY=31

OUT_DIR='../data/Metar'

mkdir -p ${OUT_DIR}

# Record the download start time.
printf 'Started: %s\n' "$(date -Is)"

# Download the requested MADIS Metar surface data.
build-metar-dataset --remove-original -n 16 ${START_YEAR} ${START_MONTH} ${START_DAY} ${END_YEAR} ${END_MONTH} ${END_DAY} ${REGION} ${OUT_DIR}

# Record successful completion.
printf 'Completed: %s\n' "$(date -Is)"
