#!/usr/bin/env bash
# Eval a TRAPI-served model on HealthBench Professional via the official
# simple-evals pipeline, graded by gpt-chat-latest (the SAME grader as the
# in-loop validation) — a frontier reference point + pipeline sanity check.
# TRAPI-only: does NOT load the training teacher on server 5.
#
#   MODEL=gpt-5.3-chat_2026-03-03 bash run_hbpro_eval_trapi.sh
#
# grader-effort is "" because gpt-chat-latest rejects reasoning_effort=none/low
# (only accepts its default), so we omit it.
set -xeuo pipefail

KEY="${TRAPI_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key)}"
MODEL="${MODEL:-gpt-5.3-chat_2026-03-03}"
TRAPI_BASE="${TRAPI_BASE:-http://point.dd.works:18890/v1}"
GRADER_MODEL="${GRADER_MODEL:-gpt-chat-latest_2026-05-28}"
CONC="${CONC:-8}"
MAXTOK="${MAXTOK:-4096}"
LIMIT="${LIMIT:-0}"

export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
cd "$(dirname "$0")/../../.."

LIMIT_FLAG=()
[ "$LIMIT" -gt 0 ] && LIMIT_FLAG=(--limit "$LIMIT")

exec /usr/local/bin/python scripts/self_evolving/eval/healthbench_professional_eval.py \
    --model-provider trapi --model-base "$TRAPI_BASE" --model-name "$MODEL" --model-key "$KEY" \
    --max-tokens "$MAXTOK" \
    --grader trapi --grader-base "$TRAPI_BASE" --grader-key "$KEY" \
    --grader-model "$GRADER_MODEL" --grader-effort "" \
    --concurrency "$CONC" --no-wandb \
    --output-dir /scratch/sheng/self_evolving/logs/hbpro_eval \
    "${LIMIT_FLAG[@]}" "$@"
