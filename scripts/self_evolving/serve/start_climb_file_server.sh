#!/usr/bin/env bash
# Launch the CLIMB media file server on the LOCAL node (mib), where
# /scratch/high_modality physically lives. Serves images/videos over
# authenticated HTTP to the remote gen server + trainer.
#
# The auth token is read from the environment only (never hardcoded). Export
# CLIMB_FILE_TOKEN to the SAME value as the teacher/chat API key before running:
#
#   export CLIMB_FILE_TOKEN=sk-...        # same key the teacher authenticates with
#   bash scripts/self_evolving/serve/start_climb_file_server.sh
#
# Defaults serve the whole high_modality tree on port 18080; override via env.

set -xeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

export CLIMB_DATA_ROOT="${CLIMB_DATA_ROOT:-/scratch/high_modality}"
# Name of the env var that holds the bearer token (the token itself stays in
# the environment, never on the command line).
export CLIMB_FILE_TOKEN_ENV="${CLIMB_FILE_TOKEN_ENV:-CLIMB_FILE_TOKEN}"

CLIMB_FILE_HOST="${CLIMB_FILE_HOST:-0.0.0.0}"
CLIMB_FILE_PORT="${CLIMB_FILE_PORT:-18080}"

if [ -z "${!CLIMB_FILE_TOKEN_ENV:-}" ]; then
    echo "ERROR: \$$CLIMB_FILE_TOKEN_ENV is empty. Export it to the teacher API key value first." >&2
    exit 1
fi

cd "$(dirname "$0")/../../.."

exec "$PYTHON_BIN" scripts/self_evolving/climb_file_server.py \
    --host "$CLIMB_FILE_HOST" \
    --port "$CLIMB_FILE_PORT" \
    "$@"
