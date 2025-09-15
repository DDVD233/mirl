#!/bin/bash

# Shell script for head_only training strategy
# This script launches training with only the classification head being trained

echo "Starting Multi Head Testing..."

# Set CUDA_VISIBLE_DEVICES to use GPUs 2 and 3
export CUDA_VISIBLE_DEVICES="4,5"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Set environment variables for better performance
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

#  --load_checkpoint_path "/scratch/keane/human_behaviour/2_head_only_training/checkpoint_epoch_1.pt" \
#     --load_checkpoint_path /scratch/keane/human_behaviour/2_debug_head_only_training/step_00000003/" \   
#  --load_checkpoint_path "/scratch/keane/human_behaviour/3_debug_head_only_training/step_6" \
#     --load_checkpoint_path "/scratch/keane/human_behaviour/_debug_head_only_training/step_3" \

    # --use_scheduler \
    # --scheduler_type cosine \
    # --warmup_steps 25

#   --train_file "/scratch/keane/human_behaviour/human_behaviour_data/rla_ptsd_train_w_feats.jsonl" \
#   --val_file   "/scratch/keane/human_behaviour/human_behaviour_data/rla_ptsd_test_w_feats.jsonl" \
#   --test_file  "/scratch/keane/human_behaviour/human_behaviour_data/rla_ptsd_test_w_feats.jsonl" \
#   --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/v2_unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.json" \

# --train_file "/scratch/keane/human_behaviour/human_behaviour_data/v2_unified_scheme_train.jsonl" \
#     --val_file "/scratch/keane/human_behaviour/human_behaviour_data/v2_unified_scheme_val.jsonl" \
#     --test_file "/scratch/keane/human_behaviour/human_behaviour_data/v2_unified_scheme_test.jsonl" \
#     --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/v2_unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.json" \

# Launch training with accelerate for head_only strategy
echo "Launching Multi Head Testing with Accelerate..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train_multi_head.py \
    --mode test \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --epochs 10 \
    --train_file "/scratch/keane/human_behaviour/human_behaviour_data/v5_train.jsonl" \
    --val_file "/scratch/keane/human_behaviour/human_behaviour_data/v5_val.jsonl" \
    --test_file "/scratch/keane/human_behaviour/human_behaviour_data/v5_test.jsonl" \
    --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/unified_label_map_w_feats_v5_unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.json" \
    --load_checkpoint_path "/scratch/keane/human_behaviour/2_qa_multi_task_model/step_4500" \
    --save_every_n_epochs 999999 \
    --save_every_n_steps 999999 \
    --save_checkpoint_dir "/scratch/keane/human_behaviour/v5_multi_head_lora_training" \
    --validation_result_dir "/scratch/keane/human_behaviour/v5_multi_head_lora_training/test_results" \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps 15000 \
    --early_stopping_patience 99999999 \
    --project "test-omni-classifier-multi-head-lora" \
    --gradient_accumulation_steps 128 \
    --format_prompt "" \
    --max_prompt_length 4096 \


echo "Lora Multi Head Testing completed!"
