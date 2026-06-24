#!/usr/bin/env bash
# Launch the self-evolving question generation server.
#
# Run this BEFORE the trainer (e.g. in a separate tmux pane). The trainer's
# SelfEvolvingDataset blocks on /healthz at startup until this is reachable.
#
# Defaults match the mimiciv_rare layout on point.dd.works; override via env.

set -xeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/dvdai/miniconda3/envs/verl/bin/python}"

DATA_DIR="${DATA_DIR:-$HOME/scratch/dvdai/self_evolving_datasets/mimiciv_rare}"
SEEDS_PATH="${SEEDS_PATH:-$DATA_DIR/train.jsonl}"
# Optional test seeds (labels stripped, used only to drive question generation
# against test-like distributions; never inserted into the pool verbatim).
# Auto-pick test.jsonl if it exists in DATA_DIR; override with empty string to
# disable.
if [ -z "${TEST_SEEDS_PATH+x}" ]; then
    if [ -f "$DATA_DIR/test.jsonl" ]; then
        TEST_SEEDS_PATH="$DATA_DIR/test.jsonl"
    else
        TEST_SEEDS_PATH=""
    fi
fi
DIRECT_TARGET="${DIRECT_TARGET:-0.30}"
GEN_TRAIN_TARGET="${GEN_TRAIN_TARGET:-0.35}"
GEN_TEST_TARGET="${GEN_TEST_TARGET:-0.35}"
LOG_DIR="${LOG_DIR:-$HOME/scratch/dvdai/self_evolving_datasets/logs}"

API_BASE="${API_BASE:-http://node2500:8002/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
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

# SFT-distillation mode: attach a verified teacher reasoning trace to every
# accepted entry (set GEN_MODE=sft). Off by default (RL pipeline).
SFT_MODE_FLAG=""
if [ "${GEN_MODE:-}" = "sft" ]; then
    SFT_MODE_FLAG="--sft_mode"
fi
TEACHER_RETRIES="${TEACHER_RETRIES:-2}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-4096}"

# CLIMB multimodal generation (gen_mm mode). Enabled when CLIMB_SEEDS_PATH +
# CLIMB_FILE_BASE are set; the file-server token is read from CLIMB_FILE_TOKEN
# in the server process (never passed on the CLI). For CLIMB retrieval set
# MILVUS_COLLECTION=medical_knowledge_v2 (the collection that indexes CLIMB).
CLIMB_SEEDS_PATH="${CLIMB_SEEDS_PATH:-}"
CLIMB_FILE_BASE="${CLIMB_FILE_BASE:-}"
GEN_MM_TARGET="${GEN_MM_TARGET:-0.0}"
MM_IMAGES_PER_QUERY="${MM_IMAGES_PER_QUERY:-3}"
MM_VIDEO_FRAMES="${MM_VIDEO_FRAMES:-2}"
MM_MAX_PIXELS="${MM_MAX_PIXELS:-1048576}"
MM_DIRECT_PROB="${MM_DIRECT_PROB:-0.3}"

cd "$(dirname "$0")/../../.."

TEST_SEEDS_FLAG=()
if [ -n "$TEST_SEEDS_PATH" ]; then
    TEST_SEEDS_FLAG=(--test_seeds_path "$TEST_SEEDS_PATH")
fi

CLIMB_FLAGS=()
if [ -n "$CLIMB_SEEDS_PATH" ] && [ -n "$CLIMB_FILE_BASE" ]; then
    CLIMB_FLAGS=(
        --climb_seeds_path "$CLIMB_SEEDS_PATH"
        --climb_file_base "$CLIMB_FILE_BASE"
        --gen_mm_target "$GEN_MM_TARGET"
        --mm_images_per_query "$MM_IMAGES_PER_QUERY"
        --mm_video_frames "$MM_VIDEO_FRAMES"
        --mm_max_pixels "$MM_MAX_PIXELS"
        --mm_direct_prob "$MM_DIRECT_PROB"
    )
fi

exec "$PYTHON_BIN" scripts/self_evolving/generation_server.py \
    --seeds_path "$SEEDS_PATH" \
    "${TEST_SEEDS_FLAG[@]}" \
    --direct_target "$DIRECT_TARGET" \
    --gen_train_target "$GEN_TRAIN_TARGET" \
    --gen_test_target "$GEN_TEST_TARGET" \
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
    --teacher_retries "$TEACHER_RETRIES" \
    --teacher_max_tokens "$TEACHER_MAX_TOKENS" \
    "${CLIMB_FLAGS[@]}" \
    $NO_LABEL_FLAG \
    $SFT_MODE_FLAG \
    "$@"
