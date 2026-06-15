#!/usr/bin/env bash
# Evaluate the Qwen3.5-397B-A17B-FP8 vLLM teacher (vps3.dd.works:18005) on
# the MIMIC-IV rare-disease test split, scored with the same compute_score
# the trainer uses so numbers are directly comparable to wandb val-* curves.
#
# The teacher is itself an OpenAI-compatible vLLM server with
# `--reasoning-parser qwen3`; --provider vllm pulls the answer from
# message.reasoning_content if message.content is empty (e.g. truncation
# mid-thinking). The judge endpoint is the same server unless overridden.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TEACHER_BASE="${TEACHER_BASE:-http://vps3.dd.works:18005/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
# Judge can be the same 397B (self-judge) or a separate model. Default to
# whatever the training run is using via API_BASE/MODEL_NAME so val-aux is
# apples-to-apples.
export API_BASE="${API_BASE:-$TEACHER_BASE}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-$TEACHER_MODEL}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"

VAL_FILE="${VAL_FILE:-/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl}"
CONCURRENCY="${CONCURRENCY:-16}"
LIMIT="${LIMIT:-0}"
VLLM_THINKING="${VLLM_THINKING:-True}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-16384}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/scratch/sheng/self_evolving/logs/eval_vllm_${TEACHER_MODEL//\//_}.jsonl}"

cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" scripts/self_evolving/eval_sota.py \
    --provider vllm \
    --val_file "$VAL_FILE" \
    --model_name "$TEACHER_MODEL" \
    --openai_base_url "$TEACHER_BASE" \
    --vllm_thinking "$VLLM_THINKING" \
    --vllm_max_tokens "$VLLM_MAX_TOKENS" \
    --judge_model_name "$MODEL_NAME" \
    --api_base "$API_BASE" \
    --judge_api_key "$API_KEY" \
    --embed_api_base "$EMBED_API_BASE" \
    --embed_api_key "$EMBED_API_KEY" \
    --embed_model "$EMBED_MODEL" \
    --concurrency "$CONCURRENCY" \
    --limit "$LIMIT" \
    --output_jsonl "$OUTPUT_JSONL" \
    "$@"
