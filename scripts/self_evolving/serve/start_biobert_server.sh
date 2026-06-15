#!/usr/bin/env bash
# Start the BioBERT similarity server.
#
# The reward function calls this server to compute semantic similarity
# between the model's extracted answer and the ground-truth diagnosis,
# adding a smooth signal alongside exact-match accuracy and the
# LLM-judge scores.
#
# Usage::
#
#     CUDA_VISIBLE_DEVICES=3 BIOBERT_PORT=8003 \
#         bash scripts/self_evolving/start_biobert_server.sh
#
# Defaults: model=pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb,
# port=8003, device=cuda if available.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export BIOBERT_PORT="${BIOBERT_PORT:-8003}"
export BIOBERT_HOST="${BIOBERT_HOST:-0.0.0.0}"
export BIOBERT_MODEL="${BIOBERT_MODEL:-pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb}"
export BIOBERT_DEVICE="${BIOBERT_DEVICE:-cuda}"

exec "$PYTHON_BIN" scripts/self_evolving/biobert_server.py "$@"
