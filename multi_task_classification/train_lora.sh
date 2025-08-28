#!/bin/bash

# Shell script for LoRA training strategy
# This script launches training with LoRA (Low-Rank Adaptation)

echo "Starting LoRA training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Launch training with accelerate for LoRA strategy
echo "Launching LoRA training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --training_strategy lora \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --lr 5e-5 \
    --epochs 10 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/lora_training" \
    --validate_every_n_epochs 1 \
    --early_stopping_patience 5 \
    --project "omni-classifier-lora" \
    --gradient_accumulation_steps 4

echo "LoRA training completed!"
