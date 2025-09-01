#!/bin/bash

# Shell script for LoRA testing strategy
# This script launches testing with LoRA (Low-Rank Adaptation)

echo "Starting LoRA testing..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# NOTE IF A SCHEDULER IS USED FOR TRAINING, IT MUST ALSO BE PUT AS A PARAMETER FOR TESTING
# OTHERWISE THE ACCELERATOR WON'T LOAD PROPERLY

# Launch testing with accelerate for LoRA strategy
echo "Launching LoRA testing with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --mode test \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/0.01_audio_sigs_train_meld.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/0.01_audio_sigs_train_meld.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/0.01_audio_sigs_train_meld.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/meld_label_map.json" \
    --validation_result_dir "/scratch/keane/human_behaviour/2_lr_unified_scheme_full_lora_training/validation_results" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/test_debug_head_only_training" \
    --project "omni-classifier-lora-test" \
    --gradient_accumulation_steps 8 \
    --use_scheduler \

echo "LoRA testing completed!"
