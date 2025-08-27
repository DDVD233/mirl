#!/bin/bash

# Launch script for Accelerate training with FSDP
# This script sets up the environment and launches multi-GPU training

echo "Setting up Accelerate training environment..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Check if accelerate config exists, if not create it
if [ ! -f "accelerate_config.yaml" ]; then
    echo "Creating accelerate config..."
    accelerate config --config_file accelerate_config.yaml
fi

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Launch training with accelerate
echo "Launching training with Accelerate..."
accelerate launch \
    --config_file accelerate_config.yaml \
    train_omni_classifier_accelerate.py

echo "Training completed!"
