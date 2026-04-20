#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Prepare per-dataset JSONLs for zero-shot inference (EATD-Corpus and MVSA).
# Run this once before run_inference.sh.
#
# Usage:
#   bash prepare_data.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/zero_shot_data"

EATD_JSONL="$DATA_DIR/test_eatd_prompts.jsonl"
MVSA_JSONL="$DATA_DIR/test_mvsa_prompts.jsonl"

if [[ ! -f "$EATD_JSONL" ]]; then
    echo "[prepare] Generating EATD JSONL..."
    python "$SCRIPT_DIR/prepare_eatd.py" \
        --data_dir "$DATA_DIR/EATD-Corpus" \
        --output   "$EATD_JSONL"
else
    echo "[prepare] EATD JSONL: $EATD_JSONL (already exists)"
fi

if [[ ! -f "$MVSA_JSONL" ]]; then
    echo "[prepare] Generating MVSA JSONL..."
    python "$SCRIPT_DIR/prepare_mvsa.py" \
        --data_dir "$DATA_DIR/MVSA" \
        --output   "$MVSA_JSONL"
else
    echo "[prepare] MVSA JSONL: $MVSA_JSONL (already exists)"
fi

echo ""
echo "Data preparation complete."
echo "  EATD : $EATD_JSONL"
echo "  MVSA : $MVSA_JSONL"
