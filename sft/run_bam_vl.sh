#!/bin/bash
# BAM-on-Qwen3-VL test-run launcher.
#
# Trains three side-channel adapters (facial / pose / audio) on top of a (frozen,
# bam_only) backbone with the QA teacher-forcing objective. The backbone is whatever
# model.backbone_name in the config names (currently Qwen/Qwen3.5-27B).
# Native video frames feed the backbone alongside the BAM deltas when
# vl_use_native_video=true / modalities="videos" (set in the config).
#
# Most settings come from configs/config_bam_vl_accelerate.yaml; edit there or override
# the few flags below. Run from the sft/ directory.

set -euo pipefail

module load community-modules ffmpeg/5.1.4
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(command -v ffmpeg)")")/lib:$LD_LIBRARY_PATH"

# Let the CUDA caching allocator grow/return segments instead of failing on a fragmented
# pool. Targets the delayed ("after a good while") OOM caused by variable-size native-video
# tensors fragmenting GPU memory over many steps. Free, no code changes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/config_bam_vl_accelerate.yaml"
ACCEL_CONFIG="configs/accelerate_config_qwen3vl.yaml"

MODE="train"
TASK_TYPE="qa"

# NOTE: train_file / val_file / save_checkpoint_dir are intentionally NOT overridden here —
# they come from the config (data.train_file, data.val_file, train.save_checkpoint_dir) so
# both curriculum stages read one source of truth. CLI args beat the config in
# train_bam_vl.py, so overriding the splits here silently pointed stage 1 at the non-CoT
# data (1-token answers) while stage 2 trained on the CoT data (~300-token answers) — the
# warm-up was optimizing a different objective than the stage it feeds. The trainer creates
# save_checkpoint_dir/step_<N> itself, so no mkdir is needed.
accelerate launch --config_file "$ACCEL_CONFIG" train_bam_vl.py \
  --config "$CONFIG" \
  --mode "$MODE" \
  --task_type "$TASK_TYPE" \
  --bam_stage "bam_only"

echo "BAM-VL run finished."
