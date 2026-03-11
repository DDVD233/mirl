#!/bin/bash
# QA stage training: loads a frozen multi-head checkpoint, trains only the lm_head.
# Edit the paths below before running.

export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

TRAIN_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_train_cleaned_2.jsonl"
VAL_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_val_cleaned.jsonl"
TEST_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_test_cleaned.jsonl"
LABEL_MAP="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map.json"
LOAD_CHECKPOINT="/scratch/keane/human_behaviour/v6_multi_head_lora_training/step_43539"
SAVE_DIR="/scratch/keane/human_behaviour/qa_lm_head_training"
VAL_DIR="$SAVE_DIR/test_results"

echo "Launching QA lm_head training..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train_qa.py \
    --mode train \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --epochs 5 \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --test_file "$TEST_FILE" \
    --label_map_path "$LABEL_MAP" \
    --load_checkpoint_path "$LOAD_CHECKPOINT" \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 2000 \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps None \
    --early_stopping_patience 999999 \
    --gradient_accumulation_steps 8 \
    --use_scheduler \
    --scheduler_type cosine \
    --warmup_steps 50 \
    --format_prompt "" \
    --max_prompt_length 8096 \
    --qa_datasets intentqa mimeqa siq2 \
    --qa_loss_weight 1.0 \
    --project "qa-lm-head-omni-classifier"

echo "QA training completed!"
