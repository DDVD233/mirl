#!/usr/bin/env bash
set -euo pipefail

echo "Starting LORA + RHA training (per-dataset)…"

# ==== USER CONFIG (keep these constant except the two file paths) ====
TRAIN_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_train.jsonl"
VAL_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_test.jsonl"

ACCEL_CFG="configs/accelerate_config_qwen.yaml"
SCRIPT="train_rha_multi_head.py"

# BASE_SAVE_DIR="/scratch/keane/human_behaviour/v6_rha_remaining_no_conf_no_gamma"
# PROJECT_NAME="v6-rha-nogamma_noconf_omni-classifier-multi-head-lora"
BASE_SAVE_DIR="/scratch/keane/human_behaviour/v6-rha-nogamma_noconf_omni-classifier-multi-head-lora"
PROJECT_NAME="v6-rha-omni-classifier-multi-head-lora"

# === NEW: Allowlist (exact match, case-sensitive). If non-empty, ONLY these run.
# INCLUDE_DATASETS=("meld_senti")
# CURRENTLY TRAINING:
# INCLUDE_DATASETS=("urfunny" "ptsd_in_the_wild" "tess" "ravdess")

# NOT TRAINING YET: (missed out: ravdess)
# INCLUDE_DATASETS=("mosei_emotion" "mosei_senti" "meld_senti" "chsimsv2" "cremad" "meld_emotion")


    # --use_scheduler \
    # --scheduler_type cosine \
    # --warmup_steps 50 \

# FULL LIST (minus ravdess)
# For old, with everything inside (conf, gamma):
# Completed ("mmsd" "urfunny" "mosei_emotion" "mosei_senti" "meld_senti" "chsimsv2" "meld_emotion")
# LEAVING "daicwoz" out for now as it doesn't have pose; we configure this differently later
# INCLUDE_DATASETS=("ptsd_in_the_wild" "tess" "cremad")

# For most baseline (no conf, no gamma):
# completed ("mmsd" "urfunny" "mosei_emotion" "mosei_senti")
# Leaving out "daicwoz" for now as it doesn't have pose; we configure this differently later
# INCLUDE_DATASETS=("meld_senti" "chsimsv2" "cremad" "meld_emotion" "ptsd_in_the_wild" "tess")
# INCLUDE_DATASETS=("tess" "ptsd_in_the_wild" "meld_emotion" "cremad" "chsimsv2" "meld_senti")
    # --use_rla_video \
INCLUDE_DATASETS=("daicwoz")

# Exclude list (used only when INCLUDE_DATASETS is empty)
# this is the list of all datasets, the only datasets that we do not have are literally expw, einterface, mmpsy, so exclude those
EXCLUDE_DATASETS=("einterface" "expw" "mmpsy_anxiety" "mmpsy_depression" "meld_emotion" "cremad" "chsimsv2" "meld_senti" "mosei_emotion" "mosei_senti" "ravdess" "tess" "ptsd_in_the_wild" "daicwoz" "urfunny" "mmsd")

# --train_file /scratch/keane/human_behaviour/human_behaviour_data/trunc_rla_fulltemp_train_daicwoz.jsonl
# /scratch/keane/human_behaviour/human_behaviour_data/trunc_rla_fulltemp_train_daicwoz.jsonl (need to use this for daicwoz)
#     --train_file "$TRAIN_OUT" \
# Environment
export CUDA_VISIBLE_DEVICES="4,5"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Temp directory for filtered JSONLs
TMP_DIR="/scratch/keane/human_behaviour/human_behaviour_data/"

# ---- helper: membership check for arrays ----
in_list() {
  local needle="$1"; shift || true
  for x in "$@"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}

# ==== helper: filter a JSONL by dataset ====
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
        line=line.strip()
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

# ==== helper: list unique dataset names present in a JSONL ====
list_datasets() {
  local in_jsonl="$1"
  if command -v jq >/dev/null 2>&1; then
    jq -r 'select(.dataset?) | .dataset' "$in_jsonl" | sort -u
  else
    python3 - "$in_jsonl" <<'PY'
import sys, json
seen=set()
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj=json.loads(line)
        except Exception:
            continue
        ds=obj.get("dataset")
        if isinstance(ds,str) and ds not in seen:
            seen.add(ds)
for s in sorted(seen):
    print(s)
PY
  fi
}

# ==== build the dataset list: appear in TRAIN and/or VAL ====
echo "Collecting dataset names…"
mapfile -t TRAIN_DS < <(list_datasets "$TRAIN_JSONL")
mapfile -t VAL_DS   < <(list_datasets "$VAL_JSONL")
mapfile -t ALL_DS_ARR < <(printf "%s\n%s\n" "${TRAIN_DS[@]}" "${VAL_DS[@]}" | sort -u)

if ((${#ALL_DS_ARR[@]} == 0)); then
  echo "No datasets found in the provided JSONLs. Exiting."
  exit 1
fi

# ==== NEW: Decide which datasets to process ====
PROCESS_DS=()
if ((${#INCLUDE_DATASETS[@]} > 0)); then
  echo "Using INCLUDE_DATASETS allowlist: ${INCLUDE_DATASETS[*]}"
  for DS in "${INCLUDE_DATASETS[@]}"; do
    if in_list "$DS" "${ALL_DS_ARR[@]}"; then
      PROCESS_DS+=("$DS")
    else
      echo "Warning: '$DS' not present in JSONLs; skipping."
    fi
  done
else
  # Fall back to exclude list
  for DS in "${ALL_DS_ARR[@]}"; do
    if in_list "$DS" "${EXCLUDE_DATASETS[@]}"; then
      echo "Skipping $DS (explicitly excluded)."
      continue
    fi
    PROCESS_DS+=("$DS")
  done
fi

if ((${#PROCESS_DS[@]} == 0)); then
  echo "No datasets to process after applying include/exclude. Exiting."
  exit 0
fi

# ==== main loop ====
for DS in "${PROCESS_DS[@]}"; do
  echo "-----------------------------------------------"
  echo "Dataset: $DS"
  TRAIN_OUT="$TMP_DIR/rla_fulltemp_train_${DS}.jsonl"
  VAL_OUT="$TMP_DIR/rla_fulltemp_test_${DS}.jsonl"

  filter_jsonl "$TRAIN_JSONL" "$DS" "$TRAIN_OUT"
  filter_jsonl "$VAL_JSONL"   "$DS" "$VAL_OUT"

  TRAIN_LINES=$(wc -l < "$TRAIN_OUT" || echo 0)
  VAL_LINES=$(wc -l < "$VAL_OUT" || echo 0)

  if [[ "$TRAIN_LINES" -eq 0 || "$VAL_LINES" -eq 0 ]]; then
    echo "Skipping $DS (train lines: $TRAIN_LINES, val lines: $VAL_LINES)."
    continue
  fi

  SAVE_DIR="${BASE_SAVE_DIR}/rla_${DS}"
  VAL_DIR="${SAVE_DIR}/validation_results"
  mkdir -p "$SAVE_DIR" "$VAL_DIR"

  echo "Launching training for dataset=$DS"
  echo "  train_file: $TRAIN_OUT  ($TRAIN_LINES lines)"
  echo "  val_file:   $VAL_OUT    ($VAL_LINES lines)"
  echo "  save_dir:   $SAVE_DIR"

  accelerate launch --config_file "$ACCEL_CFG" "$SCRIPT" \
    --mode train \
    --rla_resume_diff_training_stage \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --hard_gamma 0.0 \
    --base_lr 1e-4 \
    --rla_lr 5e-4 \
    --epochs 4 \
    --train_file "$TRAIN_OUT" \
    --val_file "$VAL_OUT" \
    --test_file "$VAL_OUT" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_v6.json" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/v6_multi_head_lora_training/step_43539" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 9999999 \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps 999999 \
    --early_stopping_patience 99999 \
    --project "${PROJECT_NAME}" \
    --gradient_accumulation_steps 1 \
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
    --rla_video_use_ln \
    --rla_audio_use_ln \
    --format_prompt "" \
    --max_prompt_length 4096 \

  echo "Finished dataset: $DS"
done

echo "All runs completed."
