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


# NOTE A WARMUP STEP HERE IS NOT A BATCH, IT IS A GRADIENT ACCUMULATION STEP

# Launch training with accelerate for LoRA strategy
echo "Launching LoRA training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --mode train \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/unified_scheme_train.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/unified_scheme_val.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/unified_scheme_test.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/final_unified_scheme_binarymmpsy_no_vptd_chalearn_lmvd_esconv_full_label_map.json" \
    --lr 1e-4 \
    --epochs 10 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/2_lr_unified_scheme_full_lora_training_resume" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/2_lr_unified_scheme_full_lora_training_resume/step_40000" \
    --validate_every_n_steps 10000 \
    --validate_every_n_epochs 1 \
    --save_every_n_steps 10000 \
    --early_stopping_patience 99999999 \
    --project "omni-classifier-lora-full-unified" \
    --gradient_accumulation_steps 128 \
    --use_scheduler \
    --scheduler_type cosine \
    --warmup_steps 50

echo "LoRA training completed!"
