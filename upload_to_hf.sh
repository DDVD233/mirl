#!/bin/bash
# upload_to_hf.sh — Upload a trained SFT checkpoint to HuggingFace Hub
#
# Full upload:
#   bash upload_to_hf.sh
#
# README-only update (model already on HF, just refresh the model card):
#   bash upload_to_hf.sh --readme-only

CKPT_DIR="/scratch/keane/human_behaviour/4_freeze_base_qa_multi_task_model/step_4578"
REPO_ID="keentomato/omnisapiens_sft"
BACKBONE_NAME="Qwen/Qwen2.5-Omni-7B"
LABEL_SCHEME="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map_v6.json"
SAVE_DIR="/scratch/keane/human_behaviour/hf_staging"

if [[ "$1" == "--readme-only" ]]; then
    python upload_to_hf.py \
        --ckpt_dir   "$CKPT_DIR" \
        --repo_id    "$REPO_ID" \
        --save_dir   "$SAVE_DIR" \
        --dataset_repo "keentomato/human_behavior_atlas" \
        --readme_only
else
    python upload_to_hf.py \
        --ckpt_dir   "$CKPT_DIR" \
        --repo_id    "$REPO_ID" \
        --backbone_name "$BACKBONE_NAME" \
        --label_scheme  "$LABEL_SCHEME" \
        --save_dir   "$SAVE_DIR" \
        --lora_scaling 2.0 \
        --dataset_repo "keentomato/human_behavior_atlas" \
        --private
fi
