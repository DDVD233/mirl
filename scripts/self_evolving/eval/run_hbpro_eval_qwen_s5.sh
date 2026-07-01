#!/usr/bin/env bash
# Eval the server-5 Qwen3.6-27B through the OFFICIAL simple-evals HealthBench
# Professional pipeline (same as the gpt-5.3-chat run), graded by gpt-chat-latest.
# This is the apples-to-apples cross-check for the in-loop val (~0.59): if the
# official number matches, the in-loop val is sound; if it's far lower, the
# in-loop val pipeline has a bug.
#
# Matches the TRAINING/val model conditions: thinking ON, the clinical-assistant
# solver system prompt (identical to RUBRIC_SOLVER_SYSTEM in generation_server.py),
# max 8192 tokens. Runs at LOW concurrency because server 5 is shared with the
# live training teacher/judge.
set -xeuo pipefail

KEY="${TRAPI_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key)}"
MODEL_BASE="${MODEL_BASE:-http://point.dd.works:18184/v1}"   # server 5 Qwen
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
TRAPI_BASE="${TRAPI_BASE:-http://point.dd.works:18890/v1}"    # grader
GRADER_MODEL="${GRADER_MODEL:-gpt-chat-latest_2026-05-28}"
CONC="${CONC:-6}"
MAXTOK="${MAXTOK:-8192}"
LIMIT="${LIMIT:-0}"

# IDENTICAL to RUBRIC_SOLVER_SYSTEM (generation_server.py) so the eval reflects
# the trained model's real prompting.
SYS="You are a knowledgeable, careful medical AI assistant helping a clinician. Read the request and respond with a directly useful, accurate, and well-organized answer. Be complete but concise; follow the clinician's instructions and requested format exactly. Ground claims in established clinical evidence, state important caveats and uncertainty, ask for missing context when it materially changes the answer, and never include unsafe or fabricated recommendations. Prioritize patient safety."

export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
cd "$(dirname "$0")/../../.."

LIMIT_FLAG=()
[ "$LIMIT" -gt 0 ] && LIMIT_FLAG=(--limit "$LIMIT")

exec /usr/local/bin/python scripts/self_evolving/eval/healthbench_professional_eval.py \
    --model-provider vllm --model-base "$MODEL_BASE" --model-name "$MODEL" --model-key "$KEY" \
    --enable-thinking --max-tokens "$MAXTOK" \
    --system-message "$SYS" \
    --grader trapi --grader-base "$TRAPI_BASE" --grader-key "$KEY" \
    --grader-model "$GRADER_MODEL" --grader-effort "" \
    --concurrency "$CONC" --no-wandb \
    --output-dir /scratch/sheng/self_evolving/logs/hbpro_eval \
    "${LIMIT_FLAG[@]}" "$@"
