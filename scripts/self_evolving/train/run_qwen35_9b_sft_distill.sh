#!/usr/bin/env bash
# Scaling-law ablation, stage 1 of 2 for the Qwen3.5-9B line: the SAME SFT
# distillation recipe as the 27B main line (same distill_sft_train.jsonl teacher
# traces, lr 1e-6, bs 32, 1 epoch — mirrors run n5f46r0i), with the 9B student.
# SAVE_FREQ=30 so global_step_90 exists: the 27B RL stage initialized from its
# step-90 SFT checkpoint ("half distill"), i.e. the same number of SFT examples.
#
# Qwen3.5-9B quirks (see run_qwen35_9b_selfimprove.sh): text head_dim=256 breaks
# FlashAttention's varlen kernel -> sdpa + use_remove_padding=False; rollout TP=2.
set -xeuo pipefail

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
export ATTN_SDPA="${ATTN_SDPA:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export EXP="${EXP:-mimiciv_rare_qwen35_9b_sft_distill}"
export LR="${LR:-1e-6}"
export SAVE_FREQ="${SAVE_FREQ:-30}"
export REPO="${REPO:-/scratch/sheng/self_evolving/verl_healthbench}"

exec bash "$(dirname "$0")/run_qwen36_27b_sft_distill.sh" "$@"
