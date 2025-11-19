#!/usr/bin/env bash
set -euo pipefail

echo "Starting RHA profiling (subset-based)…"

########################
# USER CONFIG
########################

# Full train JSONL (the big one)
TRAIN_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_train.jsonl"

# Number of examples to sample for profiling
PROFILE_SAMPLES=200   # <- change to anything between 100–500

ACCEL_CFG="configs/accelerate_config_qwen.yaml"
SCRIPT="train_rha_multi_head.py"

BASE_SAVE_DIR="/scratch/keane/human_behaviour/reviewer_expts_rha_cls"
PROJECT_NAME="reviewer_rha_profile_subset"

LABEL_MAP_PATH="/home/keaneong/human-behavior/verl/multi_task_classification/label_maps/unified_label_map_v6.json"

# Environment
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

TMP_DIR="/scratch/keane/human_behaviour/human_behaviour_data"

########################################
# 1) Build random subset JSONL from TRAIN_JSONL
########################################

PROFILE_JSONL="${TMP_DIR}/rla_profile_subset_${PROFILE_SAMPLES}.jsonl"

echo "Sampling ${PROFILE_SAMPLES} lines from:"
echo "  ${TRAIN_JSONL}"
echo "→ Writing subset to:"
echo "  ${PROFILE_JSONL}"

if command -v shuf >/dev/null 2>&1; then
  # Use shuf if available (memory-efficient)
  shuf -n "${PROFILE_SAMPLES}" "${TRAIN_JSONL}" > "${PROFILE_JSONL}"
else
  # Python fallback with reservoir sampling (memory O(N), passes over entire file)
  python3 - "${TRAIN_JSONL}" "${PROFILE_SAMPLES}" "${PROFILE_JSONL}" <<'PY'
import sys, random

inp, n_str, outp = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(n_str)
reservoir = []
with open(inp, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.rstrip('\n')
        if not line:
            continue
        if i < n:
            reservoir.append(line)
        else:
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = line

with open(outp, 'w', encoding='utf-8') as g:
    for line in reservoir:
        g.write(line + "\n")
PY
fi

SUBSET_LINES=$(wc -l < "${PROFILE_JSONL}" || echo 0)
if [[ "${SUBSET_LINES}" -eq 0 ]]; then
  echo "ERROR: subset file ${PROFILE_JSONL} is empty. Exiting."
  exit 1
fi

echo "Subset created with ${SUBSET_LINES} lines."

########################################
# 2) Profile BASE model (RLA flags OFF, but rla_stage still residual_and_head)
########################################

SAVE_DIR_BASE="${BASE_SAVE_DIR}/profile_base_subset_${PROFILE_SAMPLES}"
VAL_DIR_BASE="${SAVE_DIR_BASE}/validation_results"
mkdir -p "${SAVE_DIR_BASE}" "${VAL_DIR_BASE}"

echo
echo "========================================"
echo "Profiling BASE-ONLY config (no use_rla_* flags)…"
echo "  save_dir: ${SAVE_DIR_BASE}"
echo "========================================"

accelerate launch --config_file "${ACCEL_CFG}" "${SCRIPT}" \
  --mode profile \
  --training_strategy lora \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --test_batch_size 2 \
  --lr 1e-4 \
  --hard_gamma 0.0 \
  --base_lr 1e-4 \
  --rla_lr 5e-4 \
  --epochs 1 \
  --train_file "${PROFILE_JSONL}" \
  --val_file   "${PROFILE_JSONL}" \
  --test_file  "${PROFILE_JSONL}" \
  --label_map_path "${LABEL_MAP_PATH}" \
  --save_every_n_epochs 9999999 \
  --save_every_n_steps 9999999 \
  --save_checkpoint_dir "${SAVE_DIR_BASE}" \
  --validation_result_dir "${VAL_DIR_BASE}" \
  --validate_every_n_epochs 1 \
  --validate_every_n_steps 999999 \
  --early_stopping_patience 99999 \
  --project "${PROJECT_NAME}_rha_profiler" \
  --gradient_accumulation_steps 8 \
  --rla_stage residual_and_head \
  --d_video_feat 3318 \
  --d_audio_feat 6373 \
  --rla_hidden_video 256 \
  --rla_hidden_audio 512 \
  --rla_p_moddrop_video 0.20 \
  --rla_p_moddrop_audio 0.20 \
  --rla_video_temporal meanstd \
  --rla_video_norm none \
  --rla_audio_norm l2 \
  --rla_audio_temporal none \
  --rla_video_alpha_init 4.0 \
  --rla_audio_alpha_init 4.0 \
  --format_prompt "" \
  --max_prompt_length 4096

########################################
# 3) Profile model WITH RHA adapters
#    (same rla_stage, but now use_rla_* flags ON)
########################################

SAVE_DIR_RHA="${BASE_SAVE_DIR}/profile_rha_subset_${PROFILE_SAMPLES}"
VAL_DIR_RHA="${SAVE_DIR_RHA}/validation_results"
mkdir -p "${SAVE_DIR_RHA}" "${VAL_DIR_RHA}"

echo
echo "========================================"
echo "Profiling RHA config (use_rla_audio/use_rla_video ON)…"
echo "  save_dir: ${SAVE_DIR_RHA}"
echo "========================================"

accelerate launch --config_file "${ACCEL_CFG}" "${SCRIPT}" \
  --mode profile \
  --training_strategy lora \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --test_batch_size 2 \
  --lr 1e-4 \
  --hard_gamma 0.0 \
  --base_lr 1e-4 \
  --rla_lr 5e-4 \
  --epochs 1 \
  --train_file "${PROFILE_JSONL}" \
  --val_file   "${PROFILE_JSONL}" \
  --test_file  "${PROFILE_JSONL}" \
  --label_map_path "${LABEL_MAP_PATH}" \
  --save_every_n_epochs 9999999 \
  --save_every_n_steps 9999999 \
  --save_checkpoint_dir "${SAVE_DIR_RHA}" \
  --validation_result_dir "${VAL_DIR_RHA}" \
  --validate_every_n_epochs 1 \
  --validate_every_n_steps 999999 \
  --early_stopping_patience 99999 \
  --project "${PROJECT_NAME}_rha_profiler" \
  --gradient_accumulation_steps 8 \
  --rla_stage residual_and_head \
  --d_video_feat 3318 \
  --d_audio_feat 6373 \
  --rla_hidden_video 256 \
  --rla_hidden_audio 512 \
  --rla_p_moddrop_video 0.20 \
  --rla_p_moddrop_audio 0.20 \
  --rla_video_temporal meanstd \
  --rla_video_norm none \
  --rla_audio_norm l2 \
  --rla_audio_temporal none \
  --rla_video_alpha_init 4.0 \
  --rla_audio_alpha_init 4.0 \
  --use_rla_audio \
  --use_rla_video \
  --rla_video_use_ln \
  --rla_audio_use_ln \
  --format_prompt "" \
  --max_prompt_length 4096

echo
echo "Profiling runs completed on subset of ${PROFILE_SAMPLES} examples."
echo "Compare the [PROFILE] logs from base vs RHA to get latency & VRAM overhead."