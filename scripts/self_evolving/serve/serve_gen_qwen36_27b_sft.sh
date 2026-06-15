#!/usr/bin/env bash
# SFT-mode generation server for the Qwen3.6-27B SELF-IMPROVEMENT SFT run
# (server3, 8x B200). Same model is proposer / generator / validator AND the
# teacher that solves each accepted question to produce a verified
# <think>…</think>\boxed{answer} trace (--sft_mode). Backend = the local vLLM
# :8100 serving Qwen3.6-27B (CHAT_PROVIDER=vllm), so every role is the model
# itself. The teacher solve is grounded in the retrieved medical knowledge the
# question was synthesized from (see attach_teacher_trace), not the bare
# question.
#
# Start the vLLM :8100 server FIRST, then this, then run_qwen36_27b_selfimprove_sft.sh.
set -xeuo pipefail

export CHAT_PROVIDER="${CHAT_PROVIDER:-vllm}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
REPO_ROOT="${REPO_ROOT:-/scratch/sheng/self_evolving/verl}"
DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/self_evolving/generation_server.py \
    --seeds_path "$DATA_DIR/train.jsonl" \
    --test_seeds_path "$DATA_DIR/test.jsonl" \
    --direct_target 0.30 \
    --gen_train_target 0.35 \
    --gen_test_target 0.35 \
    --api_base "${API_BASE:-http://localhost:8100/v1}" \
    --api_key "${API_KEY:-EMPTY}" \
    --model_name "${MODEL_NAME:-Qwen/Qwen3.6-27B}" \
    --embed_api_base "${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}" \
    --embed_model "${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}" \
    --milvus_uri "${MILVUS_URI:-http://mib.media.mit.edu:19531}" \
    --milvus_token "${MILVUS_TOKEN:-root:Milvus}" \
    --milvus_collection "${MILVUS_COLLECTION:-medical_knowledge}" \
    --milvus_top_k "${MILVUS_TOP_K:-16}" \
    --n_queries "${N_QUERIES:-10}" \
    --questions_per_query "${QUESTIONS_PER_QUERY:-1}" \
    --accuracy_window 64 \
    --max_pool_size "${MAX_POOL_SIZE:-512}" \
    --workers "${GEN_WORKERS:-16}" \
    --log_dir "${LOG_DIR:-/scratch/sheng/self_evolving/logs_sft_qwen36}" \
    --host 0.0.0.0 \
    --port "${GEN_PORT:-8005}" \
    --sft_mode \
    --teacher_retries "${TEACHER_RETRIES:-2}" \
    --teacher_max_tokens "${TEACHER_MAX_TOKENS:-4096}" \
    --milvus_history_collection "${MILVUS_HISTORY_COLLECTION:-gen_history_qwen36_sft}" \
    "$@"
