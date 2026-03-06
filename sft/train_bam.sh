#!/bin/bash
# BAM adapter training: loads a frozen multi-head checkpoint and trains
# per-dataset video/audio residual hidden adapters.
# Edit the paths below before running.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

TRAIN_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_train.jsonl"
VAL_JSONL="/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_test.jsonl"
LABEL_MAP="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map_v6.json"
LOAD_CHECKPOINT="/scratch/keane/human_behaviour/v6_multi_head_lora_training/step_43539"
BASE_SAVE_DIR="/scratch/keane/human_behaviour/rha_adapter_training"
PROJECT_NAME="v6-rha-omni-classifier-multi-head-lora"
TMP_DIR="/scratch/keane/human_behaviour/human_behaviour_data"

# Datasets to train adapters for (one run per dataset)
INCLUDE_DATASETS=("mosei_emotion" "mosei_senti" "meld_senti" "ptsd_in_the_wild")

# ---- helpers ----
in_list() {
  local needle="$1"; shift || true
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

filter_jsonl() {
  local in_jsonl="$1" dataset="$2" out_jsonl="$3"
  python3 - "$in_jsonl" "$dataset" "$out_jsonl" <<'PY'
import sys, json
inp, ds, outp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(inp, 'r') as f, open(outp, 'w') as g:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("dataset") == ds:
            g.write(json.dumps(obj) + "\n")
PY
}

list_datasets() {
  python3 - "$1" <<'PY'
import sys, json
seen = set()
with open(sys.argv[1], 'r') as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception: continue
        ds = obj.get("dataset")
        if isinstance(ds, str) and ds not in seen:
            seen.add(ds); print(ds)
PY
}

echo "Collecting dataset names from JSONL files..."
mapfile -t ALL_DS_ARR < <(
  { list_datasets "$TRAIN_JSONL"; list_datasets "$VAL_JSONL"; } | sort -u
)

PROCESS_DS=()
for DS in "${INCLUDE_DATASETS[@]}"; do
  if in_list "$DS" "${ALL_DS_ARR[@]}"; then
    PROCESS_DS+=("$DS")
  else
    echo "Warning: '$DS' not found in JSONLs; skipping."
  fi
done

if ((${#PROCESS_DS[@]} == 0)); then
  echo "No datasets to process. Exiting."
  exit 0
fi

for DS in "${PROCESS_DS[@]}"; do
  echo "-------------------------------------------"
  echo "Training RHA adapter for: $DS"

  TRAIN_OUT="$TMP_DIR/rha_train_${DS}.jsonl"
  VAL_OUT="$TMP_DIR/rha_val_${DS}.jsonl"
  filter_jsonl "$TRAIN_JSONL" "$DS" "$TRAIN_OUT"
  filter_jsonl "$VAL_JSONL"   "$DS" "$VAL_OUT"

  TRAIN_LINES=$(wc -l < "$TRAIN_OUT" || echo 0)
  VAL_LINES=$(wc -l < "$VAL_OUT" || echo 0)
  if [[ "$TRAIN_LINES" -eq 0 || "$VAL_LINES" -eq 0 ]]; then
    echo "Skipping $DS (train=$TRAIN_LINES, val=$VAL_LINES)."
    continue
  fi

  SAVE_DIR="${BASE_SAVE_DIR}/rha_${DS}"
  VAL_DIR="${SAVE_DIR}/validation_results"
  mkdir -p "$SAVE_DIR" "$VAL_DIR"

  echo "  train_file: $TRAIN_OUT  ($TRAIN_LINES lines)"
  echo "  val_file:   $VAL_OUT    ($VAL_LINES lines)"
  echo "  save_dir:   $SAVE_DIR"

  accelerate launch --config_file configs/accelerate_config_qwen.yaml train_bam.py \
    --mode train \
    --bam_resume_diff_training_stage \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --base_lr 1e-4 \
    --bam_lr 5e-4 \
    --hard_gamma 0.0 \
    --epochs 3 \
    --train_file "$TRAIN_OUT" \
    --val_file "$VAL_OUT" \
    --test_file "$VAL_OUT" \
    --label_map_path "$LABEL_MAP" \
    --load_checkpoint_path "$LOAD_CHECKPOINT" \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    --save_every_n_epochs None \
    --save_every_n_steps None \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps None \
    --early_stopping_patience 99999 \
    --gradient_accumulation_steps 4 \
    --bam_stage bam_and_classifier_heads_only \
    --use_bam_audio \
    --d_video_feat 3318 \
    --d_audio_feat 6373 \
    --bam_hidden_video 256 \
    --bam_hidden_audio 256 \
    --bam_p_moddrop_video 0.20 \
    --bam_p_moddrop_audio 0.20 \
    --bam_video_temporal meanstd \
    --bam_video_norm none \
    --bam_audio_norm l2 \
    --bam_audio_temporal none \
    --bam_video_alpha_init 4.0 \
    --bam_audio_alpha_init 4.0 \
    --bam_video_use_ln \
    --bam_audio_use_ln \
    --format_prompt "" \
    --max_prompt_length 4096 \
    --project "${PROJECT_NAME}"

  echo "Finished: $DS"
done

echo "All RHA adapter runs completed."
