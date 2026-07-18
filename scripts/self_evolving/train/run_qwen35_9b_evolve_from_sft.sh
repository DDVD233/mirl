#!/usr/bin/env bash
# Scaling-law ablation, stage 2 of 2 for the Qwen3.5-9B line: the SAME data-gen
# RL recipe as the 27B main line (j75o3rrt / run_qwen36_27b_selfimprove_from_sft:
# gen server on :8006, teacher+judge = Qwen3.6-27B @ server 5, lr 2e-7,
# EVOLVE_ENABLE matches the main line), with the 9B student initialized from its
# own SFT-distill checkpoint (merged HF dir; see run_ablation_queue_s4.sh).
#
# Qwen3.5-9B quirks: sdpa + use_remove_padding=False + rollout TP=2 (head_dim=256).
set -xeuo pipefail

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:?set to the merged 9B SFT HF dir}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
export ATTN_SDPA="${ATTN_SDPA:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export EXP="${EXP:-mimiciv_rare_qwen35_9b_evolve_from_sft}"
export EVOLVE_ENABLE="${EVOLVE_ENABLE:-True}"
export TOTAL_STEPS="${TOTAL_STEPS:-500}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
export REPO="${REPO:-/scratch/sheng/self_evolving/verl_healthbench}"

exec bash "$(dirname "$0")/run_qwen36_27b_selfimprove_from_sft.sh" "$@"
