#!/bin/bash
# BAM-on-Qwen3-VL curriculum STAGE 2 launcher (full-model unfreeze via LoRA).
#
# Seeds from the stage-1 (bam_only) checkpoint as INITIALIZATION (--load_as_init):
# the model + trained BAM adapters are loaded, the step counter is reset to 0, and a
# fresh optimizer is built over this stage's param groups. With training_strategy=lora
# (set in the config), bam_and_full_model trains LoRA + heads + BAM adapters; for QA the
# heads get no gradient, so effectively LoRA + adapters.
#
# Prereq: stage 1 must have been run with training_strategy=lora (same config) so the
# checkpoint carries the LoRA module tree — otherwise the load key-mismatches.
#
# Run from the sft/ directory.

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

# train_file / val_file come from the config (data.train_file / data.val_file), same as
# stage 1, so the two stages can never drift onto different splits.

# Stage-1 (bam_only) save dir — must match the --save_checkpoint_dir stage 1 wrote to.
STAGE1_DIR="/orcd/scratch/orcd/011/emorfin/ados/checkpoints/bam_vl_qwen35_27b_qa_sft_stage2_full_lora"
# Stage-2 output dir (kept separate so stage 1 stays intact).
STAGE2_DIR="/orcd/scratch/orcd/011/emorfin/ados/checkpoints/bam_vl_qwen35_27b_qa_sft_stage2_full_lora_ep5"

# Stage-2 duration. EPOCHS is the hard bound; MAX_STEPS (micro-batches) caps it earlier if
# set. Set MAX_STEPS large (or equal to EPOCHS*steps_per_epoch) to run the full schedule —
# this OVERRIDES the small stage-1 cap baked into the config.
EPOCHS=1
MAX_STEPS=2100

# Pick the latest stage-1 checkpoint (step_<N> with the highest N) to seed from. Sort by the
# numeric suffix after "step_" only (the dir path also contains underscores, so a plain
# -t_ -k2 sort would key on the wrong field).
STAGE1_CKPT="$(ls -d "${STAGE1_DIR}"/step_* 2>/dev/null \
  | awk -F'/step_' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2- || true)"
if [[ -z "${STAGE1_CKPT}" || ! -f "${STAGE1_CKPT}/meta.json" ]]; then
  echo "ERROR: no stage-1 checkpoint with meta.json found under ${STAGE1_DIR}" >&2
  echo "       Run stage 1 (run_bam_vl.sh, training_strategy=lora) first." >&2
  exit 1
fi
echo "Seeding stage 2 from: ${STAGE1_CKPT}"

mkdir -p "$STAGE2_DIR"


accelerate launch --config_file "$ACCEL_CONFIG" train_bam_vl.py \
  --config "$CONFIG" \
  --mode "$MODE" \
  --task_type "$TASK_TYPE" \
  --bam_stage "bam_and_full_model" \
  --load_as_init \
  --load_checkpoint_path "$STAGE1_CKPT" \
  --save_checkpoint_dir "$STAGE2_DIR" \
  --epochs "$EPOCHS" \
  --max_steps "$MAX_STEPS"

echo "BAM-VL stage-2 run finished."