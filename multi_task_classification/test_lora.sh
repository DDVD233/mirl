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

# NOTE ALL THE SCHEDULER PARAMETERS MUST BE THE SAME AS THE PARAMETERS USED FOR TRAINING
# OTHERWISE THE ACCELERATOR WON'T LOAD PROPERLY

# Launch testing with accelerate for LoRA strategy
echo "Launching LoRA testing with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train.py \
    --mode test \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/unified_scheme_train.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/unified_scheme_val.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/unified_scheme_test.jsonl" \
    --validation_result_dir "/scratch/keane/human_behaviour/2_lr_unified_scheme_full_lora_training/validation_results" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/final_unified_scheme_binarymmpsy_no_vptd_chalearn_lmvd_esconv_full_label_map.json" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/2_lr_unified_scheme_full_lora_training/step_26094" \
    --project "omni-classifier-lora-test" \
    --gradient_accumulation_steps 8 \
    --use_scheduler \
    
echo "LoRA testing completed!"
