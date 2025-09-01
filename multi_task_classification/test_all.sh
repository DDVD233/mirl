#!/bin/bash

# Shell script for full model testing strategy
# This script launches testing with all model parameters

echo "Starting full model testing..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Launch testing with accelerate for full strategy
echo "Launching full model testing with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --mode test \
    --training_strategy full \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_train_meld.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_val_meld.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_test_meld.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/meld_label_map.json" \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --load_checkpoint_path "/scratch/keane/human_behaviour/omni_full_training/checkpoint_epoch_1.pt" \
    --project "omni-classifier-full-test" \
    --gradient_accumulation_steps 64

echo "Full model testing completed!"
