#!/bin/bash

# Shell script for full training strategy
# This script launches training with all model parameters being updated

echo "Starting full model training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Launch training with accelerate for full strategy
echo "Launching full model training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --training_strategy full \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --lr 1e-6 \
    --epochs 15 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/full_training" \
    --validate_every_n_epochs 1 \
    --early_stopping_patience 7 \
    --project "omni-classifier-full" \
    --gradient_accumulation_steps 8

echo "Full model training completed!"
