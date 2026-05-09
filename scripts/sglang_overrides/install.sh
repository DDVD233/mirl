#!/bin/bash
# Copy patched sglang model files over the installed sglang in $CONDA_PREFIX.
# Usage:  bash scripts/sglang_overrides/install.sh
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: CONDA_PREFIX not set. Activate the target conda env first." >&2
    exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGL_MODELS_DIR="${CONDA_PREFIX}/lib/python3.12/site-packages/sglang/srt/models"

if [ ! -d "$SGL_MODELS_DIR" ]; then
    echo "ERROR: sglang models dir not found at $SGL_MODELS_DIR" >&2
    exit 1
fi

for f in "$SRC_DIR"/*.py; do
    base="$(basename "$f")"
    target="$SGL_MODELS_DIR/$base"
    cp "$f" "$target"
    echo "installed: $target"
done
