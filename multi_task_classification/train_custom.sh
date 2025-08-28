#!/bin/bash

# Customizable training script with parameter overrides
# Usage: ./train_custom.sh --training_strategy head_only --lr 1e-4 --epochs 10

echo "Starting customizable training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Build the command with all arguments passed to this script
# This allows you to pass any parameter to the training script
echo "Launching training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py "$@"

echo "Training completed!"
