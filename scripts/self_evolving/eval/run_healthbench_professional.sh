#!/usr/bin/env bash
# Run HealthBench Professional (OpenAI, 2026) on one of our vLLM-served models
# and log the rubric scores to Weights & Biases.
#
# HealthBench Professional is a 525-example, text-only benchmark of real
# clinician chats, graded by physician-written rubrics. We drive OpenAI's
# reference grader from openai/simple-evals (cloned automatically). This is a
# different task/scoring scheme from our MIMIC-IV rare-disease eval, so it does
# NOT reuse verl compute_score.
#
# Two grader options:
#   GRADER=openai  -> GPT-5.4 at low reasoning effort  (needs OPENAI_API_KEY;
#                     numbers comparable to the paper / leaderboard)
#   GRADER=local   -> a vLLM judge endpoint of ours     (cheap, internal-only)
#
# Examples:
#   # Comparable score with the official GPT-5.4-low grader:
#   OPENAI_API_KEY=sk-... MODEL_BASE=http://localhost:8000/v1 \
#   MODEL_NAME=Qwen/Qwen3.6-27B \
#   scripts/self_evolving/eval/run_healthbench_professional.sh
#
#   # Internal-only, judged by our own Qwen3.6-27B (no OpenAI key):
#   GRADER=local GRADER_BASE=http://node2500:8002/v1 \
#   GRADER_MODEL_LOCAL=Qwen/Qwen3.6-27B \
#   MODEL_BASE=http://localhost:8000/v1 MODEL_NAME=Qwen/Qwen3.6-27B \
#   scripts/self_evolving/eval/run_healthbench_professional.sh --limit 50

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# --- model under test (OpenAI-compatible vLLM) ---
export MODEL_BASE="${MODEL_BASE:-http://localhost:8000/v1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export MODEL_KEY="${MODEL_KEY:-EMPTY}"

# --- grader ---
GRADER="${GRADER:-openai}"
export GRADER_MODEL="${GRADER_MODEL:-gpt-5.4-2026-03-05}"
GRADER_EFFORT="${GRADER_EFFORT:-low}"
export GRADER_BASE="${GRADER_BASE:-http://node2500:8002/v1}"
export GRADER_KEY="${GRADER_KEY:-EMPTY}"
export GRADER_MODEL_LOCAL="${GRADER_MODEL_LOCAL:-Qwen/Qwen3.6-27B}"

# --- eval / wandb ---
LIMIT="${LIMIT:-0}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-self_evolving_eval}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/sheng/self_evolving/logs/healthbench_professional}"
export SIMPLE_EVALS_DIR="${SIMPLE_EVALS_DIR:-$HOME/.cache/simple_evals_src}"

cd "$(dirname "$0")/../../.."
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" scripts/self_evolving/eval/healthbench_professional_eval.py \
    --model-base "$MODEL_BASE" \
    --model-name "$MODEL_NAME" \
    --model-key "$MODEL_KEY" \
    --grader "$GRADER" \
    --grader-model "$GRADER_MODEL" \
    --grader-effort "$GRADER_EFFORT" \
    --grader-base "$GRADER_BASE" \
    --grader-key "$GRADER_KEY" \
    --grader-model-local "$GRADER_MODEL_LOCAL" \
    --limit "$LIMIT" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --wandb-project "$WANDB_PROJECT" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
