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
   
# Launch training with accelerate for head_only strategy
echo "Launching head_only training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --training_strategy head_only \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --lr 1e-4 \
    --epochs 2 \
    --save_every_n_epochs 9 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/2_head_only_training" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/2_head_only_training/checkpoint_epoch_1.pt" \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps None \
    --early_stopping_patience 99999 \
    --project "omni-classifier-head-only" \
    --gradient_accumulation_steps 1

echo "Head-only training completed!"
