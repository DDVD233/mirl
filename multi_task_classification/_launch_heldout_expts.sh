#!/usr/bin/env bash
# multi_experiment_cls_and_qa.sh
# Indexed arrays: TYPES[i], TRAINS[i], VALS[i], TESTS[i]

set -u
echo "Starting LoRA sweep (CLS + QA)…"

# GPUs / perf
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Shared configs
ACCEL_CFG="configs/accelerate_config_qwen.yaml"
PROJECT="all_heldout_expts"
LABEL_MAP="/home/keaneong/human-behavior/verl/multi_task_classification/seperate_unified_label_map_v6.json"
BASE_SAVE_DIR="/scratch/keane/human_behaviour/all_heldout_expts"
RESUME_FROM=""

# --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/seperate_unified_label_map_v6.json" \
# "/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_v6.json"

# Common args
COMMON_ARGS=(
  --mode train
  --training_strategy lora
  --train_batch_size 1
  --val_batch_size 1
  --test_batch_size 1
  --lr 1e-4
  --epochs 10
  --label_map_path "$LABEL_MAP"
  --save_every_n_epochs 999999
  --save_every_n_steps 999999
  --validate_every_n_epochs 1
  --validate_every_n_steps 9999999
  --early_stopping_patience 99999999
  --project "$PROJECT"
  --gradient_accumulation_steps 1
  --format_prompt ""
  --max_prompt_length 4096
)

# -----------------------------------------
# HARD-CODED EXPERIMENTS (keep indices aligned)
# TYPE ∈ {cls, qa}
# -----------------------------------------
TYPES=(
  "cls"   # daicwoz (train_4)
  "cls"   # daicwoz (train_32)
  "cls"   # meld_emotion (train_4)
  "cls"   # meld_emotion (train_32)
  "cls"   # mmsd (train_4)
  "cls"   # mmsd (train_32)
  "cls"   # mosei_senti (train_4)
  "cls"   # mosei_senti (train_32)
  "qa"    # mimeqa (train_4)
  "qa"    # mimeqa (train_32)
)

TRAINS=(
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_meld_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_meld_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_mosei_senti.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_mosei_senti.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_qa_mimeqa.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_qa_mimeqa.jsonl"
)

VALS=(
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
)

TESTS=(
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
)
# Sanity check
if [[ ${#TYPES[@]} -ne ${#TRAINS[@]} || ${#TYPES[@]} -ne ${#VALS[@]} || ${#TYPES[@]} -ne ${#TESTS[@]} ]]; then
  echo "[FATAL] Array length mismatch (TYPES/TRAINS/VALS/TESTS)."
  exit 2
fi

# -----------------------------------------
# Run loop
# -----------------------------------------
for i in "${!TYPES[@]}"; do
  EXP_TYPE="${TYPES[$i]}"
  TRAIN_FILE="${TRAINS[$i]}"
  VAL_FILE="${VALS[$i]}"
  TEST_FILE="${TESTS[$i]}"

  SAVE_DIR="${BASE_SAVE_DIR}/exp${i}"
  VAL_DIR="${SAVE_DIR}/validation_results"
  mkdir -p "$SAVE_DIR" "$VAL_DIR"

  SCRIPT="train_multi_head.py"
  EXTRA_ARGS=()

  if [[ "$EXP_TYPE" == "qa" ]]; then
    SCRIPT="addqa_train_multi_head.py"
    EXTRA_ARGS+=( --format_prompt "" --max_prompt_length 8096 )
  fi

  LOAD_ARG=()
  if [[ -n "${RESUME_FROM:-}" && -d "$RESUME_FROM" ]]; then
    LOAD_ARG=( --load_checkpoint_path "$RESUME_FROM" )
  fi

  LOG_FILE="${SAVE_DIR}/run_$(date +'%Y%m%d_%H%M%S').log"

  echo "------------------------------------------------------------"
  echo "[$((i+1))/${#TYPES[@]}] Experiment index $i  (type: $EXP_TYPE)"
  echo "Train: $TRAIN_FILE"
  echo "Val:   $VAL_FILE"
  echo "Test:  $TEST_FILE"
  echo "Save:  $SAVE_DIR"
  echo "Log:   $LOG_FILE"
  echo "Script: $SCRIPT"
  echo "------------------------------------------------------------"

  set +e
  accelerate launch --config_file "$ACCEL_CFG" "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    "${LOAD_ARG[@]}" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --test_file "$TEST_FILE" \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[WARN] Experiment index $i failed (exit $EXIT_CODE). Continuing."
  else
    echo "[OK] Experiment index $i completed."
  fi
done

echo "All experiments finished."