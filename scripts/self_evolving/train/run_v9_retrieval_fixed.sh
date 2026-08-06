#!/usr/bin/env bash
# v9 = RL from STOCK with the REDESIGNED retrieval loop (retrieval decision phase +
# always-clean tool-free answer). This is the config where retrieval first became
# NET-POSITIVE at val (stock+redesign step0 = 0.563 > no-ret 0.559; empties 71->0).
# Self-contained launcher (avoids inline-ssh quoting issues): sets env + redirects log.
set -uo pipefail
cd /scratch/sheng/self_evolving/verl_healthbench

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen3.6-27B}"
export EXP="${EXP:-healthbench_rubric_qwen36_27b_v9_retrieval_fixed}"
export RETRIEVAL_URL="${RETRIEVAL_URL:-http://localhost:8006/retrieve}"
# Validation judge: gpt-chat-latest (higher-throughput TRAPI deployment; gpt-5.1 is
# rate-limited). It's a CHAT model so _call_api omits reasoning_effort (chat rejects it).
export VAL_JUDGE_MODEL="${VAL_JUDGE_MODEL:-gpt-chat-latest_2026-05-28}"
# Throttle judge (trapi) calls: per-worker cap x 8 reward workers = total concurrency.
# 6 x 8 = 48 concurrent, well under TRAPI's ~2000/60s shared cap. Prevents the val/step
# grading burst that hammered the rate limit.
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-6}"
LOG="${LOG:-/scratch/sheng/self_evolving/logs_healthbench_rubric/v9_retrieval_fixed_train.log}"

# tee (not exec-redirect) so the tmux pane shows live output AND the log is written.
bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric_v7_retrieval.sh 2>&1 | tee "$LOG"
