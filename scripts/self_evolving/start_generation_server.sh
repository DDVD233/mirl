#!/usr/bin/env bash
# Launch the self-evolving question generation server.
#
# Run this BEFORE the trainer (e.g. in a separate tmux pane). The trainer's
# SelfEvolvingDataset blocks on /healthz at startup until this is reachable.
#
# Defaults match the mimiciv_rare layout on point.dd.works; override via env.

set -xeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/dvdai/miniconda3/envs/verl/bin/python}"

DATA_DIR="${DATA_DIR:-/scratch/self_evolving_datasets/mimiciv_rare}"
SEEDS_PATH="${SEEDS_PATH:-$DATA_DIR/train.jsonl}"
LOG_DIR="${LOG_DIR:-/scratch/self_evolving_datasets/logs}"

API_BASE="${API_BASE:-http://localhost:8002/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
EMBED_API_BASE="${EMBED_API_BASE:-http://localhost:8001/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
MILVUS_TOKEN="${MILVUS_TOKEN:-root:Milvus}"
MILVUS_COLLECTION="${MILVUS_COLLECTION:-medical_knowledge}"
MILVUS_TOP_K="${MILVUS_TOP_K:-16}"

# Generation knobs
N_QUERIES="${N_QUERIES:-10}"
QUESTIONS_PER_QUERY="${QUESTIONS_PER_QUERY:-1}"
ACCURACY_WINDOW="${ACCURACY_WINDOW:-64}"
MAX_POOL_SIZE="${MAX_POOL_SIZE:-200}"
GEN_WORKERS="${GEN_WORKERS:-8}"

GEN_SERVER_PORT="${GEN_SERVER_PORT:-8004}"
GEN_SERVER_HOST="${GEN_SERVER_HOST:-0.0.0.0}"

NO_LABEL_FLAG=""
if [ "${NO_LABEL:-0}" = "1" ] || [ "${NO_LABEL:-false}" = "true" ]; then
    NO_LABEL_FLAG="--no_label"
fi

cd "$(dirname "$0")/../.."

exec "$PYTHON_BIN" scripts/self_evolving/generation_server.py \
    --seeds_path "$SEEDS_PATH" \
    --api_base "$API_BASE" \
    --api_key "$API_KEY" \
    --model_name "$MODEL_NAME" \
    --embed_api_base "$EMBED_API_BASE" \
    --embed_model "$EMBED_MODEL" \
    --milvus_uri "$MILVUS_URI" \
    --milvus_token "$MILVUS_TOKEN" \
    --milvus_collection "$MILVUS_COLLECTION" \
    --milvus_top_k "$MILVUS_TOP_K" \
    --n_queries "$N_QUERIES" \
    --questions_per_query "$QUESTIONS_PER_QUERY" \
    --accuracy_window "$ACCURACY_WINDOW" \
    --max_pool_size "$MAX_POOL_SIZE" \
    --workers "$GEN_WORKERS" \
    --log_dir "$LOG_DIR" \
    --host "$GEN_SERVER_HOST" \
    --port "$GEN_SERVER_PORT" \
    $NO_LABEL_FLAG \
    "$@"
