#!/bin/bash
# upload_to_hf.sh — Upload a trained SFT checkpoint to HuggingFace Hub

CKPT_DIR="/scratch/keane/human_behaviour/4_freeze_base_qa_multi_task_model/step_4578"
REPO_ID="keentomato/omnisapiens_sft"
BACKBONE_NAME="Qwen/Qwen2.5-Omni-7B"
LABEL_SCHEME="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map_v6.json"
SAVE_DIR="/scratch/keane/human_behaviour/hf_staging"

python upload_to_hf.py \
    --ckpt_dir   "$CKPT_DIR" \
    --repo_id    "$REPO_ID" \
    --backbone_name "$BACKBONE_NAME" \
    --label_scheme  "$LABEL_SCHEME" \
    --save_dir   "$SAVE_DIR" \
    --lora_scaling 2.0 \
    --dataset_repo "keentomato/human_behaviour_atlas" \
    --private
