#!/bin/bash

# Shell script for head_only training strategy
# This script launches training with only the classification head being trained

echo "Starting head_only training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

#  --load_checkpoint_path "/scratch/keane/human_behaviour/2_head_only_training/checkpoint_epoch_1.pt" \
#     --load_checkpoint_path /scratch/keane/human_behaviour/2_debug_head_only_training/step_00000003/" \   
#  --load_checkpoint_path "/scratch/keane/human_behaviour/3_debug_head_only_training/step_6" \
#     --load_checkpoint_path "/scratch/keane/human_behaviour/_debug_head_only_training/step_3" \

    # --load_checkpoint_path "/scratch/keane/human_behaviour/v5_multi_head_lora_training/step_20000" \
# Launch training with accelerate for head_only strategy
    # --load_checkpoint_path "/scratch/keane/human_behaviour/v5_multi_head_lora_training/step_20000" \


echo "Launching multi head training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train_multi_head.py \
    --mode train \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --epochs 10 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_exclude_heldout_train.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_exclude_heldout_val.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_exclude_heldout_test.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_v6.json" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 5000 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/v6_heldout_multi_head_lora_training" \
    --validation_result_dir "/scratch/keane/human_behaviour/v6_heldout_multi_head_lora_training/validation_results" \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps 9999999 \
    --early_stopping_patience 99999999 \
    --project "v6_heldout_omni-classifier-multi-head-lora" \
    --gradient_accumulation_steps 128 \
    --use_scheduler \
    --scheduler_type cosine \
    --warmup_steps 50 \
    --format_prompt "" \

echo "Lora Multi Head training completed!"
