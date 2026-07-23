#!/usr/bin/env bash
# Evaluate a TRAPI-served model on MedCalc-Bench-Verified using the benchmark's
# own verbatim parser + scorer (see medcalc_bench_eval.py). TRAPI-only: needs no
# GPU, so use this while the training servers are busy. Serves as a frontier
# reference baseline before we score our trained checkpoints via vLLM.
#
#   MODEL=gpt-5.3-chat_2026-03-03 LIMIT=20 bash run_medcalc_eval_trapi.sh
#   MODEL=gpt-5.3-chat_2026-03-03 bash run_medcalc_eval_trapi.sh          # full 1k
#
# reasoning-effort is left unset by default (gpt-chat models reject low/none).
set -xeuo pipefail

KEY="${TRAPI_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key)}"
MODEL="${MODEL:-gpt-5.3-chat_2026-03-03}"
TRAPI_BASE="${TRAPI_BASE:-http://point.dd.works:18890/v1}"
PROMPT="${PROMPT:-one_shot}"
CONC="${CONC:-8}"
MAXTOK="${MAXTOK:-4096}"
EFFORT="${EFFORT:-}"
LIMIT="${LIMIT:-0}"
OUT="${OUT:-/scratch/sheng/self_evolving/logs/medcalc_eval}"

cd "$(dirname "$0")/../../.."

EFFORT_FLAG=()
[ -n "$EFFORT" ] && EFFORT_FLAG=(--reasoning-effort "$EFFORT")

exec /usr/local/bin/python scripts/self_evolving/eval/medcalc_bench_eval.py \
    --provider trapi --model-base "$TRAPI_BASE" --model-name "$MODEL" --model-key "$KEY" \
    --prompt "$PROMPT" --max-tokens "$MAXTOK" --concurrency "$CONC" \
    --limit "$LIMIT" --no-wandb --output-dir "$OUT" \
    "${EFFORT_FLAG[@]}" "$@"
