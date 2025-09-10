#!/bin/bash

# This script launches training with only the classification head being trained

echo "Starting QA training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="0,1"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

    # --use_scheduler \
    # --scheduler_type cosine \
    # --warmup_steps 25

    # 0.001_qa_train.jsonl

echo "Launching QA training with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml addqa_train_multi_head.py \
    --mode train \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --epochs 5 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/qa_train.jsonl" \
    --val_file  "/scratch/keane/human_behaviour/human_behaviour_data/0.001_qa_train.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/qa_test.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_w_feats_v5_unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.json" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/v5_multi_head_lora_training/step_49500" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 9999999 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/freeze_base_qa_multi_task_model" \
    --validation_result_dir "/scratch/keane/human_behaviour/freeze_base_qa_multi_task_model/test_results" \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps 999999 \
    --early_stopping_patience 999999 \
    --project "qa-omni-classifier-multi-task-lora" \
    --gradient_accumulation_steps 8 \
    --format_prompt "" \

echo "QA training completed!"
