#!/usr/bin/env bash
# CONTROL for the 9B retrieval run: identical token budget, NO retrieval.
#
# WHY. At step 0 (before any gradient step) the retrieval run scored
# val-core/overall/acc 0.227 vs 0.145 for the previous no-retrieval run — but that
# comparison is confounded: the retrieval run also doubled max_response_length
# (4096 -> 8192) to make room for the loss-masked evidence, and the earlier analysis
# measured clip_ratio 0.51, i.e. HALF of rollouts were hitting the old 4096 cap. So
# part of that +0.082 is plausibly just answers no longer being truncated.
#
# This run holds the budget at the retrieval run's values and removes retrieval.
#   retrieval run  - retrieval + 8192  -> 0.227 (measured)
#   THIS run       - no retrieval + 8192 -> isolates the budget effect
#   prior run      - no retrieval + 4096 -> 0.145 (measured)
# The retrieval-attributable delta is (retrieval run - THIS run). Everything else is
# response budget.
#
# Runs on the preemptible pod at point.dd.works:2336 (server3), which is idle; it
# does not touch the retrieval run on 2335. /scratch/sheng is shared NFS across both
# pods, so the code deployed for 2335 is already here.
set -xeuo pipefail

S=/scratch/sheng/self_evolving
cd "$S/verl_healthbench"

export WANDB_API_KEY="$(cat $S/.wandb_key_dvd)"
export EXP="${EXP:-hb9b_ctrl_len8192}"

# Matched to run_9b_retrieval_fast.sh, EXCEPT no retrieval and no summarizer, so
# there is no need to hold a GPU slice back for one.
export MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
export ROLLOUT_MAX_LEN="${ROLLOUT_MAX_LEN:-14336}"
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-14336}"
export LOGPROB_MAX_TOKEN_LEN="${LOGPROB_MAX_TOKEN_LEN:-14336}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"

exec bash scripts/self_evolving/train/run_9b_trainval_fast.sh "$@"
