#!/usr/bin/env bash
# Scaling-law ablation, stage 4 (server 4): Qwen3.5-9B "normal RL" control —
# GRPO on the REAL mimiciv_rare train set (no gen server), initialized from the
# SAME 9B SFT-distill checkpoint as stage 3, use_kl_loss=False to match the
# main line. Completes the 2x2: {27B, 9B} x {data-gen RL, train-set RL}, all
# four from the same-data SFT inits, full val every 5 steps, judge = 27B @ s5.
#
# Run after run_ablation_queue_s4.sh (stage 2b produced the merged HF dir).
set -xeuo pipefail

REPO="${REPO:-/scratch/sheng/self_evolving/verl_healthbench}"
LOGROOT=/scratch/sheng/self_evolving/logs_ablation_s4
CKPT9B_SFT=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/mimiciv_rare_qwen35_9b_sft_distill
SFT9B_HF="$CKPT9B_SFT/global_step_90/actor/huggingface"

[ -f "$SFT9B_HF/config.json" ] || { echo "9B SFT HF dir missing ($SFT9B_HF) — did stage 2b run?"; exit 1; }
cd "$REPO"

echo "================ STAGE 4: 9B train-set RL from SFT ($(date)) ================"
MODEL_PATH="$SFT9B_HF" USE_KL_LOSS=False \
  USE_REMOVE_PADDING=False ATTN_SDPA=1 ROLLOUT_TP=2 \
  REPO="$REPO" EXP=mimiciv_rare_qwen35_9b_trainset_rl_from_sft \
  bash scripts/self_evolving/train/run_qwen36_27b_baseline.sh \
  2>&1 | tee -a "$LOGROOT/stage4_9b_trainset_rl.log"
