#!/bin/bash
# Multi-head classification training with LoRA.
# Edit the paths below before running.

export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

TRAIN_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_train_cleaned_2.jsonl"
VAL_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_val_cleaned.jsonl"
TEST_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_test_cleaned.jsonl"
LABEL_MAP="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map.json"
SAVE_DIR="/scratch/keane/human_behaviour/v6_multi_head_lora_training_trial"
VAL_DIR="$SAVE_DIR/validation_results"

echo "Launching multi-head LoRA training..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train_classification.py \
    --mode train \
    --training_strategy lora \
    --train_batch_size 216 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --epochs 10 \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --test_file "$TEST_FILE" \
    --label_map_path "$LABEL_MAP" \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 200 \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps None \
    --early_stopping_patience 99999999 \
    --gradient_accumulation_steps 128 \
    --use_scheduler \
    --scheduler_type cosine \
    --warmup_steps 50 \
    --format_prompt "" \
    --max_prompt_length 4096 \
    --project "v6_omni-classifier-multi-head-lora"

echo "Multi-head training completed!"
