#!/usr/bin/env bash
set -euo pipefail

echo "Starting RHA profiling (per-dataset subset, possibly multi-dataset)…"

########################
# USER CONFIG
########################

# Full train JSONL (the big one)
TRAIN_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_train.jsonl"

# One or more target dataset names (must match .dataset exactly, case-sensitive)
# Example: DATASETS=("urfunny" "iemocap" "mosei_senti")
DATASETS=("urfunny" "mosei_senti")    # <- edit this list

# Number of examples to sample *per dataset* for profiling.
# If a dataset has <= PROFILE_SAMPLES_PER_DATASET, we just use all of its lines.
PROFILE_SAMPLES_PER_DATASET=200   # <- change to anything between 100–500

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
# helper: filter a JSONL by dataset
########################################
filter_jsonl() {
  local in_jsonl="$1"
  local dataset="$2"
  local out_jsonl="$3"
  if command -v jq >/dev/null 2>&1; then
    jq -c "select(.dataset? == \"$dataset\")" "$in_jsonl" > "$out_jsonl" || true
  else
    python3 - "$in_jsonl" "$dataset" "$out_jsonl" <<'PY'
import sys, json
inp, ds, outp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(inp, 'r', encoding='utf-8') as f, open(outp, 'w', encoding='utf-8') as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("dataset") == ds:
            g.write(json.dumps(obj, ensure_ascii=False) + "\n")
PY
  fi
}

########################################
# 1) Build per-dataset JSONLs + per-dataset subsets
########################################

# Tag for the combined profile file (e.g. "urfunny" or "urfunny_mosei_senti")
PROFILE_TAG="$(printf '%s_' "${DATASETS[@]}")"
PROFILE_TAG="${PROFILE_TAG%_}"   # strip trailing underscore

PROFILE_JSONL="${TMP_DIR}/rla_profile_${PROFILE_TAG}_subset_${PROFILE_SAMPLES_PER_DATASET}.jsonl"

echo "Using datasets: ${DATASETS[*]}"
echo "Filtering from:"
echo "  ${TRAIN_JSONL}"
echo

# Track all per-dataset subset files so we can concatenate them
PER_DATASET_SUBSETS=()

for TARGET_DATASET in "${DATASETS[@]}"; do
  echo "----------------------------------------"
  echo "Processing dataset '${TARGET_DATASET}'…"

  DATASET_JSONL="${TMP_DIR}/rla_profile_${TARGET_DATASET}.jsonl"
  DATASET_SUBSET_JSONL="${TMP_DIR}/rla_profile_${TARGET_DATASET}_subset_${PROFILE_SAMPLES_PER_DATASET}.jsonl"

  echo "→ Filtering dataset '${TARGET_DATASET}' to:"
  echo "   ${DATASET_JSONL}"

  filter_jsonl "${TRAIN_JSONL}" "${TARGET_DATASET}" "${DATASET_JSONL}"

  DATASET_LINES=$(wc -l < "${DATASET_JSONL}" || echo 0)
  if [[ "${DATASET_LINES}" -eq 0 ]]; then
    echo "WARNING: No lines found for dataset='${TARGET_DATASET}' in ${TRAIN_JSONL}. Skipping."
    continue
  fi

  echo "Found ${DATASET_LINES} lines for dataset='${TARGET_DATASET}'."
  echo "Building profiling subset of size ${PROFILE_SAMPLES_PER_DATASET} (or full dataset if smaller)…"

  if (( DATASET_LINES <= PROFILE_SAMPLES_PER_DATASET )); then
    echo "Dataset has ${DATASET_LINES} ≤ ${PROFILE_SAMPLES_PER_DATASET}; using all lines for profiling."
    cp "${DATASET_JSONL}" "${DATASET_SUBSET_JSONL}"
  else
    echo "Sampling ${PROFILE_SAMPLES_PER_DATASET} lines from per-dataset JSONL."
    if command -v shuf >/dev/null 2>&1; then
      # Use shuf if available (memory-efficient)
      shuf -n "${PROFILE_SAMPLES_PER_DATASET}" "${DATASET_JSONL}" > "${DATASET_SUBSET_JSONL}"
    else
      # Python fallback with reservoir sampling
      python3 - "${DATASET_JSONL}" "${PROFILE_SAMPLES_PER_DATASET}" "${DATASET_SUBSET_JSONL}" <<'PY'
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
  fi

  SUB_LINES=$(wc -l < "${DATASET_SUBSET_JSONL}" || echo 0)
  if [[ "${SUB_LINES}" -eq 0 ]]; then
    echo "WARNING: subset file ${DATASET_SUBSET_JSONL} is empty for dataset='${TARGET_DATASET}'. Skipping."
    continue
  fi

  echo "Subset created with ${SUB_LINES} lines for dataset='${TARGET_DATASET}'."
  PER_DATASET_SUBSETS+=("${DATASET_SUBSET_JSONL}")
done

# Combine all per-dataset subsets into one PROFILE_JSONL
if [[ ${#PER_DATASET_SUBSETS[@]} -eq 0 ]]; then
  echo "ERROR: No non-empty per-dataset subsets were created. Exiting."
  exit 1
fi

echo
echo "Combining per-dataset subsets into single profiling file:"
printf '  - %s\n' "${PER_DATASET_SUBSETS[@]}"
cat "${PER_DATASET_SUBSETS[@]}" > "${PROFILE_JSONL}"

SUBSET_LINES=$(wc -l < "${PROFILE_JSONL}" || echo 0)
if [[ "${SUBSET_LINES}" -eq 0 ]]; then
  echo "ERROR: combined subset file ${PROFILE_JSONL} is empty. Exiting."
  exit 1
fi

echo
echo "Combined profiling subset created with ${SUBSET_LINES} lines total."
echo "  Datasets: ${DATASETS[*]}"
echo "  Subset file: ${PROFILE_JSONL}"

########################################
# 2) Profile BASE model (RLA flags OFF, but rla_stage still residual_and_head)
########################################

SAVE_DIR_BASE="${BASE_SAVE_DIR}/profile_base_${PROFILE_TAG}_subset_${PROFILE_SAMPLES_PER_DATASET}"
VAL_DIR_BASE="${SAVE_DIR_BASE}/validation_results"
mkdir -p "${SAVE_DIR_BASE}" "${VAL_DIR_BASE}"

echo
echo "========================================"
echo "Profiling BASE-ONLY config (no use_rla_* flags)…"
echo "  datasets: ${DATASETS[*]}"
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
# 3) (Optional) Profile model WITH RHA adapters
#    (same rla_stage, but now use_rla_* flags ON)
########################################

# SAVE_DIR_RHA="${BASE_SAVE_DIR}/profile_rha_${PROFILE_TAG}_subset_${PROFILE_SAMPLES_PER_DATASET}"
# VAL_DIR_RHA="${SAVE_DIR_RHA}/validation_results"
# mkdir -p "${SAVE_DIR_RHA}" "${VAL_DIR_RHA}"

# echo
# echo "========================================"
# echo "Profiling RHA config (use_rla_audio/use_rla_video ON)…"
# echo "  datasets: ${DATASETS[*]}"
# echo "  save_dir: ${SAVE_DIR_RHA}"
# echo "========================================"

# accelerate launch --config_file "${ACCEL_CFG}" "${SCRIPT}" \
#   --mode profile \
#   --training_strategy lora \
#   --train_batch_size 2 \
#   --val_batch_size 2 \
#   --test_batch_size 2 \
#   --lr 1e-4 \
#   --hard_gamma 0.0 \
#   --base_lr 1e-4 \
#   --rla_lr 5e-4 \
#   --epochs 1 \
#   --train_file "${PROFILE_JSONL}" \
#   --val_file   "${PROFILE_JSONL}" \
#   --test_file  "${PROFILE_JSONL}" \
#   --label_map_path "${LABEL_MAP_PATH}" \
#   --save_every_n_epochs 9999999 \
#   --save_every_n_steps 9999999 \
#   --save_checkpoint_dir "${SAVE_DIR_RHA}" \
#   --validation_result_dir "${VAL_DIR_RHA}" \
#   --validate_every_n_epochs 1 \
#   --validate_every_n_steps 999999 \
#   --early_stopping_patience 99999 \
#   --project "${PROJECT_NAME}_rha_profiler" \
#   --gradient_accumulation_steps 8 \
#   --rla_stage residual_and_head \
#   --d_video_feat 3318 \
#   --d_audio_feat 6373 \
#   --rla_hidden_video 256 \
#   --rla_hidden_audio 512 \
#   --rla_p_moddrop_video 0.20 \
#   --rla_p_moddrop_audio 0.20 \
#   --rla_video_temporal meanstd \
#   --rla_video_norm none \
#   --rla_audio_norm l2 \
#   --rla_audio_temporal none \
#   --rla_video_alpha_init 4.0 \
#   --rla_audio_alpha_init 4.0 \
#   --use_rla_audio \
#   --use_rla_video \
#   --rla_video_use_ln \
#   --rla_audio_use_ln \
#   --format_prompt "" \
#   --max_prompt_length 4096

# echo
# echo "Profiling runs completed on datasets='${DATASETS[*]}' "
# echo "  total subset size: ${SUBSET_LINES} examples."
# echo "Compare the [PROFILE] logs from base vs RHA to get latency & VRAM overhead."