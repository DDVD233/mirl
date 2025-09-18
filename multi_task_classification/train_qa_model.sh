#!/bin/bash

# This script launches training with only the classification head being trained

echo "Starting QA training..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="2,3"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


    # 0.001_qa_train.jsonl
    # qa_val.jsonl
    #     --use_scheduler \
    # --scheduler_type cosine \
    # --warmup_steps 50 \

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
    --val_file  "/scratch/keane/human_behaviour/human_behaviour_data/qa_test.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/qa_test.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_v6.json" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/v6_multi_head_lora_training/step_43539" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 2000 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/4_freeze_base_qa_multi_task_model/" \
    --validation_result_dir "/scratch/keane/human_behaviour/4_freeze_base_qa_multi_task_model/test_results" \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps 999999 \
    --early_stopping_patience 999999 \
    --project "4_freeze-base-qa-omni-classifier-multi-task-lora" \
    --gradient_accumulation_steps 8 \
    --format_prompt "" \
    --max_prompt_length 8096 \

echo "QA training completed!"
