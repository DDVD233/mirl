#!/bin/bash

# Shell script for LoRA testing strategy
# This script launches testing with LoRA (Low-Rank Adaptation)

echo "Starting LoRA testing..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Launch testing with accelerate for LoRA strategy
echo "Launching LoRA testing with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --mode test \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_train_chsimsv2_only.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_val_chsimsv2_only.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_test_chsimsv2_only.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/chsimsv2_label_map.json" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/debug_chsimsv2_lora_training/checkpoint_epoch_1.pt" \
    --project "omni-classifier-lora-test" \
    --gradient_accumulation_steps 8

echo "LoRA testing completed!"
