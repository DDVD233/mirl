#!/bin/bash

# Shell script for LoRA training strategy
# This script launches training with LoRA (Low-Rank Adaptation)

echo "Starting LoRA training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

    # --use_scheduler \
    # --scheduler_type cosine \
    # --warmup_steps 150

# Launch training with accelerate for LoRA strategy
echo "Launching LoRA training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_no_chalearn_no_expw_no_mosei_fixed_0.1_train.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_no_chalearn_no_expw_no_mosei_fixed_0.1_val.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_no_chalearn_no_expw_no_mosei_fixed_0.1_test.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/0.1_no_chalearn_no_expw_no_mosei_label_map.json" \
    --lr 3e-5 \
    --epochs 3 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/full_lora_training" \
    --validate_every_n_steps 1000 \
    --validate_every_n_epochs 1 \
    --early_stopping_patience 99999999 \
    --project "omni-classifier-lora" \
    --gradient_accumulation_steps 16 \

echo "LoRA training completed!"
