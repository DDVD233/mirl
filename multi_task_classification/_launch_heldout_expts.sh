#!/usr/bin/env bash
# multi_experiment_cls_and_qa.sh
# Indexed arrays: TYPES[i], TRAINS[i], VALS[i], TESTS[i], TRAIN_BS[i], GACC[i]

set -u
echo "Starting LoRA sweep (CLS + QA)…"

# GPUs / perf
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Shared configs
ACCEL_CFG="configs/accelerate_config_qwen.yaml"
PROJECT="all_heldout_expts"
LABEL_MAP="/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_v6.json"
BASE_SAVE_DIR="/scratch/keane/human_behaviour/all_heldout_expts/full_hbatlas_2epochs"
RESUME_FROM="/scratch/keane/human_behaviour/v6_heldout_multi_head_lora_training/step_38935"   # leave blank to start from scratch
# RESUME_FROM="/scratch/keane/human_behaviour/v6_heldout_multi_head_lora_training/step_38935"   # leave blank to start from scratch
  # --use_scheduler \
  # --scheduler_type cosine \
  # --warmup_steps 50 \

# Common args (no train_batch_size / gradient_accumulation_steps here; set per-index below)
COMMON_ARGS=(
  --mode train
  --training_strategy lora
  --val_batch_size 2
  --test_batch_size 2
  --lr 1e-4
  --epochs 2
  --label_map_path "$LABEL_MAP"
  --save_every_n_epochs 999999
  --save_every_n_steps 999999
  --validate_every_n_epochs 1
  --validate_every_n_steps 9999999
  --early_stopping_patience 99999999
  --project "$PROJECT"
  --format_prompt ""
  --max_prompt_length 4096
  --use_scheduler \
  --scheduler_type cosine \
  --warmup_steps 50 \
)

# -----------------------------------------
# HARD-CODED EXPERIMENTS (keep indices aligned)
# TYPE ∈ {cls, qa}
# -----------------------------------------
# TYPES=(
#   "cls"   # daicwoz (train_4)
#   "cls"   # daicwoz (train_32)
#   "cls"   # meld_emotion (train_4)
#   "cls"   # meld_emotion (train_32)
#   "cls"   # mmsd (train_4)
#   "cls"   # mmsd (train_32)
#   "cls"   # mosei_senti (train_4)
#   "cls"   # mosei_senti (train_32)
# )
#   "qa"    # mimeqa (train_4)
#   "qa"    # mimeqa (train_32)

# Few shot training files ; you should use vals of the same length
# TRAINS=(
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_daicwoz.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_daicwoz.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_meld_emotion.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_meld_emotion.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_mmsd.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_mmsd.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_mosei_senti.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_mosei_senti.jsonl"
# )
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_4_qa_mimeqa.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_32_qa_mimeqa.jsonl"
# VALS=(
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_emotion.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_emotion.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
# )
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
# TRAIN_BS=(
#   2  # daicwoz_4
#   2  # daicwoz_32
#   2  # meld_emo_4
#   2  # meld_emo_32
#   2  # mmsd_4
#   2  # mmsd_32
#   2  # mosei_senti_4
#   2  # mosei_senti_32
# )
# GACC=(
#   1  # daicwoz_4
#   1  # daicwoz_32
#   1  # meld_emo_4
#   1  # meld_emo_32
#   1  # mmsd_4
#   1  # mmsd_32
#   1  # mosei_senti_4
#   1  # mosei_senti_32
# )

TYPES=(
  "cls"   
  "cls"  
  "cls"   
  "cls"  
)

TRAINS=(
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_full_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_full_meld_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_full_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_train_full_mosei_senti.jsonl"
)

VALS=(
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_meld_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
)
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"

TESTS=(
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_daicwoz.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_meld_emotion.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mmsd.jsonl"
  "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_mosei_senti.jsonl"
)

#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"
#   "/scratch/keane/human_behaviour/human_behaviour_data/heldout_test_qa_mimeqa.jsonl"

# Per-index TRAIN BATCH SIZE and GRAD ACCUMULATION
# (Example values — tweak for your GPU/RAM budget)
TRAIN_BS=(
  2  # daicwoz_4
  2  # meld_emo_4
  2  # mmsd_4
  2  # mosei_senti_4
)
#   1  # mimeqa_4
#   2  # mimeqa_32

GACC=(
  2  # daicwoz_4
  8  # meld_emo_4
  4  # mmsd_4
  8  # mosei_senti_4
)

#   1  # mimeqa_4
#   1  # mimeqa_32

# Sanity checks
if [[ ${#TYPES[@]} -ne ${#TRAINS[@]} || ${#TYPES[@]} -ne ${#VALS[@]} || ${#TYPES[@]} -ne ${#TESTS[@]} ]]; then
  echo "[FATAL] Array length mismatch (TYPES/TRAINS/VALS/TESTS)."; exit 2
fi
if [[ ${#TYPES[@]} -ne ${#TRAIN_BS[@]} || ${#TYPES[@]} -ne ${#GACC[@]} ]]; then
  echo "[FATAL] Array length mismatch (TYPES vs TRAIN_BS/GACC)."; exit 2
fi

# -----------------------------------------
# Run loop
# -----------------------------------------
for i in "${!TYPES[@]}"; do
  EXP_TYPE="${TYPES[$i]}"
  TRAIN_FILE="${TRAINS[$i]}"
  VAL_FILE="${VALS[$i]}"
  TEST_FILE="${TESTS[$i]}"
  TB="${TRAIN_BS[$i]}"
  GA="${GACC[$i]}"

  SAVE_DIR="${BASE_SAVE_DIR}/exp${i}"
  VAL_DIR="${SAVE_DIR}/validation_results"
  mkdir -p "$SAVE_DIR" "$VAL_DIR"

  SCRIPT="train_heldout_multi_head.py"
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
  echo "Train BS: $TB   |   Grad Accum: $GA"
  echo "Save:  $SAVE_DIR"
  echo "Log:   $LOG_FILE"
  echo "Script: $SCRIPT"
  echo "------------------------------------------------------------"

  set +e
  accelerate launch --config_file "$ACCEL_CFG" "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    "${LOAD_ARG[@]}" \
    --train_batch_size "$TB" \
    --gradient_accumulation_steps "$GA" \
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