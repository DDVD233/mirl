#!/usr/bin/env bash
# Evaluate Gemini (or any SoTA via the Gemini API) on the MIMIC-IV rare
# dataset's test split, scoring with the same compute_score the trainer
# uses so numbers are directly comparable to wandb val-* curves.
#
# Required:
#   - GEMINI_API_KEY env var (Google Generative Language API key)
#   - vLLM chat server reachable at API_BASE (Qwen judge endpoint)
#   - BioBERT similarity server at BIOBERT_API_BASE
#
# On point.dd.works the launch script auto-finds the verl env's python.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/dvdai/miniconda3/envs/verl/bin/python}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export API_BASE="${API_BASE:-http://localhost:8002/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
export BIOBERT_API_BASE="${BIOBERT_API_BASE:-http://localhost:8003}"

VAL_FILE="${VAL_FILE:-/home/dvdai/scratch/dvdai/self_evolving_datasets/mimiciv_rare/test.jsonl}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"
CONCURRENCY="${CONCURRENCY:-8}"
LIMIT="${LIMIT:-0}"
# Stable filename per model so re-running resumes the prior run on interrupt.
# Override OUTPUT_JSONL or pass a new --output_jsonl to force a fresh run.
OUTPUT_JSONL="${OUTPUT_JSONL:-/home/dvdai/scratch/dvdai/self_evolving_datasets/logs/eval_gemini_${GEMINI_MODEL//\//_}.jsonl}"

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is required" >&2
    exit 1
fi

cd "$(dirname "$0")/../../.."
# Unbuffered stdout so progress lines show up under `tee` / piped capture.
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" scripts/self_evolving/eval_sota_gemini.py \
    --val_file "$VAL_FILE" \
    --model_name "$GEMINI_MODEL" \
    --judge_model_name "$MODEL_NAME" \
    --api_base "$API_BASE" \
    --biobert_api_base "$BIOBERT_API_BASE" \
    --concurrency "$CONCURRENCY" \
    --limit "$LIMIT" \
    --output_jsonl "$OUTPUT_JSONL" \
    "$@"
