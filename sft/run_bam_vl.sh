#!/bin/bash
# BAM-on-Qwen3-VL test-run launcher.
#
# Trains three side-channel adapters (facial / pose / audio) on top of a (frozen,
# bam_only) Qwen3-VL-8B-Instruct backbone with the QA teacher-forcing objective.
# Native video is NOT wired yet — this is the text+BAM QA core.
#
# Most settings come from configs/config_bam_vl_accelerate.yaml; edit there or override
# the few flags below. Run from the sft/ directory.

set -euo pipefail

CONFIG="configs/config_bam_vl_accelerate.yaml"
ACCEL_CONFIG="configs/accelerate_config_qwen3vl.yaml"

MODE="train"
TASK_TYPE="qa"
TRAIN_FILE="/orcd/scratch/orcd/011/emorfin/ados/childplay_splits/train_for_VL/train_all_for_VL.jsonl"
VAL_FILE="/orcd/scratch/orcd/011/emorfin/ados/childplay_splits/val_for_VL/val_all_for_VL.jsonl"
SAVE_DIR="/orcd/scratch/orcd/011/emorfin/ados/checkpoints/bam_vl_qwen3vl32b_qa"

mkdir -p "$SAVE_DIR"

accelerate launch --config_file "$ACCEL_CONFIG" train_bam_vl.py \
  --config "$CONFIG" \
  --mode "$MODE" \
  --task_type "$TASK_TYPE" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --save_checkpoint_dir "$SAVE_DIR" \
  --bam_stage "bam_only"

echo "BAM-VL run finished."
