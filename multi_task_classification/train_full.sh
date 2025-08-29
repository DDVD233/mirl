#!/bin/bash

# Shell script for full training strategy
# This script launches training with all model parameters being updated

echo "Starting full model training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Launch training with accelerate for full strategy
echo "Launching full model training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --training_strategy full \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_train_meld.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_val_meld.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/audio_sigs_test_meld.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/meld_label_map.json" \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --lr 1e-5 \
    --epochs 2 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/omni_full_training" \
    --validate_every_n_epochs None \
    --validate_every_n_steps 100 \
    --early_stopping_patience 99999999 \
    --project "omni-classifier-full" \
    --gradient_accumulation_steps 64

echo "Full model training completed!"
