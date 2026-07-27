#!/usr/bin/env bash
# v10 = v9's retrieval recipe, FRESH from stock, with FRONTIER models in the three
# LLM roles that were previously all played by the local Qwen teacher (or a lenient
# grader). Motivation (v6 step-65 + v9 diagnosis):
#   - the self-referential loop (Qwen writes tasks, grades them, diagnoses its own
#     failures and rewrites its own curriculum) narrowed the curriculum onto one
#     genre (premise-trap correction) while the real HB-Pro failures are
#     calculators/clinical scores, translation fidelity and trial recall;
#   - training rollouts scored mean 0.725 with 0.9% zeros and zero_var_frac 0.0 —
#     almost no gradient, because the judge was grading its own answers.
#
# Role allocation (measured 2026-07-27: ~1,750 tokens per grading call):
#   META-OPTIMIZER (/evolve, ~21 calls/step)  -> gpt-5.6-sol   (best model; 0.1% of its cap)
#   VALIDATION judge (~1,050 calls / val)     -> gpt-5.4       (THE official HealthBench
#                                                grader -> leaderboard-comparable numbers)
#   TRAIN reward judge (~3,600 calls/step)    -> gpt-chat-latest (only deployment whose
#                                                6.8M tok/min absorbs ~1M tok/min; sol and
#                                                5.4 cap at 850K and would throttle)
# All three run through the TRAPI proxy (:18890). Because that proxy died once
# (2026-07-27, silently zeroing a whole validation), the reward now falls back to the
# local Qwen teacher on any judge exception — a proxy outage degrades grading quality
# instead of destroying the training signal.
#
# Fresh checkpoint dir (new EXP) => no resume, per design: v9's policy was shaped by
# the self-graded curriculum.
set -uo pipefail
cd /scratch/sheng/self_evolving/verl_healthbench

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen3.6-27B}"   # stock, like v9
export EXP="${EXP:-healthbench_rubric_qwen36_27b_v10_frontier}"
export RETRIEVAL_URL="${RETRIEVAL_URL:-http://localhost:8006/retrieve}"

TRAPI_BASE="${TRAPI_BASE:-http://point.dd.works:18890/v1}"
TRAPI_KEY="${TRAPI_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key)}"

# --- validation judge: the official HealthBench grader ---
export VAL_JUDGE_BASE="$TRAPI_BASE"
export VAL_JUDGE_MODEL="${VAL_JUDGE_MODEL:-gpt-5.4_2026-03-05}"
export VAL_JUDGE_KEY="$TRAPI_KEY"

# --- training reward judge: external (breaks the self-grading loop) ---
export TRAIN_JUDGE_BASE="${TRAIN_JUDGE_BASE:-$TRAPI_BASE}"
export TRAIN_JUDGE_MODEL="${TRAIN_JUDGE_MODEL:-gpt-chat-latest_2026-05-28}"
export TRAIN_JUDGE_PROVIDER="${TRAIN_JUDGE_PROVIDER:-trapi}"
export TRAIN_JUDGE_KEY="$TRAPI_KEY"

# --- fallback judge: local teacher, used only when a primary call raises ---
export FALLBACK_JUDGE_BASE="${FALLBACK_JUDGE_BASE:-http://point.dd.works:18184/v1}"
export FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-Qwen/Qwen3.6-27B}"
export FALLBACK_JUDGE_PROVIDER="${FALLBACK_JUDGE_PROVIDER:-vllm}"

# Judge concurrency per reward worker (x8 workers). 6x8=48 measured clean at the
# proxy with zero 429s.
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-6}"

LOG="${LOG:-/scratch/sheng/self_evolving/logs_healthbench_rubric/v10_frontier_train.log}"
mkdir -p "$(dirname "$LOG")"

bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric_v7_retrieval.sh 2>&1 | tee "$LOG"
