#!/usr/bin/env bash
set -euo pipefail

echo "Starting LORA + RLA training (per-dataset)…"

# ==== USER CONFIG (keep these constant except the two file paths) ====
TRAIN_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/v5_train.jsonl"
VAL_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/v5_test.jsonl"

ACCEL_CFG="configs/accelerate_config_qwen.yaml"
SCRIPT="train_rla_multi_head.py"

BASE_SAVE_DIR="/scratch/keane/human_behaviour/new_full_joint_rla"
PROJECT_NAME="full-rla-omni-classifier-multi-head-lora"

# Environment
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Temp directory for filtered JSONLs
TMP_DIR="./jsonl_temps"
mkdir -p "$TMP_DIR"

# ==== helper: filter a JSONL by dataset ====
filter_jsonl() {
  local in_jsonl="$1"
  local dataset="$2"
  local out_jsonl="$3"
  # Prefer jq if available; else Python fallback
  if command -v jq >/dev/null 2>&1; then
    # Keep only lines where .dataset == "$dataset"
    # Also ensure the line parses as JSON to avoid junk
    jq -c "select(.dataset? == \"$dataset\")" "$in_jsonl" > "$out_jsonl" || true
  else
    # Python fallback
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

# Join and unique
ALL_DS=$(printf "%s\n%s\n" "${TRAIN_DS[@]}" "${VAL_DS[@]}" | sort -u)

if [[ -z "$ALL_DS" ]]; then
  echo "No datasets found in the provided JSONLs. Exiting."
  exit 1
fi

# ==== main loop ====
for DS in $ALL_DS; do
  echo "-----------------------------------------------"
  echo "Dataset: $DS"
  TRAIN_OUT="$TMP_DIR/train_${DS}.jsonl"
  VAL_OUT="$TMP_DIR/val_${DS}.jsonl"

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
    --lr 0 \
    --hard_gamma 5.0 \
    --base_lr 0 \
    --rla_lr 5e-4 \
    --epochs 2 \
    --train_file "$TRAIN_OUT" \
    --val_file "$VAL_OUT" \
    --test_file "$VAL_OUT" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_w_feats_v5_unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.json" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 9999999 \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/v5_multi_head_lora_training/step_20000" \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps 999999 \
    --early_stopping_patience 99999 \
    --project "${PROJECT_NAME}" \
    --gradient_accumulation_steps 4 \
    --rla_stage residual_only \
    --d_video_feat 3318 \
    --d_audio_feat 6373 \
    --rla_hidden_video 256 \
    --rla_hidden_audio 256 \
    --rla_p_moddrop_video 0.10 \
    --rla_p_moddrop_audio 0.10 \
    --rla_video_temporal meanstd \
    --rla_video_norm none \
    --rla_audio_norm l2 \
    --rla_audio_temporal none \
    --rla_video_alpha_init 3.0 \
    --rla_audio_alpha_init 3.0 \
    --use_rla_video \
    --use_rla_audio \
    --rla_video_use_ln \
    --rla_audio_use_ln

  echo "Finished dataset: $DS"
done

echo "All runs completed."