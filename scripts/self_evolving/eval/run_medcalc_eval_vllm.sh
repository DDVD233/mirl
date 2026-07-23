#!/usr/bin/env bash
# Evaluate one of OUR trained checkpoints on MedCalc-Bench-Verified via an
# OpenAI-compatible vLLM server. Run this once a GPU server is free and the
# checkpoint is served (e.g. `vllm serve <ckpt> --port 8000 ...`).
# Same verbatim parser/scorer as the TRAPI path — only the endpoint differs.
#
#   MODEL=Qwen/Qwen3.6-27B BASE=http://localhost:8000/v1 bash run_medcalc_eval_vllm.sh
#   MODEL=/scratch/sheng/.../global_step_120/hf BASE=http://localhost:8000/v1 \
#       THINKING_OFF=1 LIMIT=50 bash run_medcalc_eval_vllm.sh
#
# THINKING_OFF=1 forces Qwen enable_thinking=false (leave unset to let the model
# reason — reasoning models usually score higher here but need more max-tokens).
set -xeuo pipefail

MODEL="${MODEL:?set MODEL to the served model name / checkpoint path}"
BASE="${BASE:-http://localhost:8000/v1}"
KEY="${MODEL_KEY:-EMPTY}"
PROMPT="${PROMPT:-one_shot}"
CONC="${CONC:-16}"
MAXTOK="${MAXTOK:-4096}"
TEMP="${TEMP:-0.0}"
LIMIT="${LIMIT:-0}"
OUT="${OUT:-/scratch/sheng/self_evolving/logs/medcalc_eval}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

cd "$(dirname "$0")/../../.."

THINK_FLAG=()
[ "${THINKING_OFF:-0}" = "1" ] && THINK_FLAG=(--thinking-off)

WANDB_FLAG=(--no-wandb)
[ -n "$WANDB_PROJECT" ] && WANDB_FLAG=(--wandb-project "$WANDB_PROJECT")

exec python scripts/self_evolving/eval/medcalc_bench_eval.py \
    --provider vllm --model-base "$BASE" --model-name "$MODEL" --model-key "$KEY" \
    --prompt "$PROMPT" --max-tokens "$MAXTOK" --temperature "$TEMP" \
    --concurrency "$CONC" --limit "$LIMIT" --output-dir "$OUT" \
    "${THINK_FLAG[@]}" "${WANDB_FLAG[@]}" "$@"
