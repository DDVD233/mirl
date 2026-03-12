#!/bin/bash
# upload_to_hf.sh — Upload a trained SFT checkpoint to HuggingFace Hub
#
# Full upload:
#   bash upload_to_hf.sh
#
# README-only update (model already on HF, just refresh the model card):
#   bash upload_to_hf.sh --readme-only

# /scratch/keane/human_behaviour/v6_rha_residual_nogamma_noconf/rla_mmsd/step_484
# /scratch/keane/human_behaviour/v6_rha_residual_nogamma_noconf/rla_urfunny/step_856

CKPT_DIR="/scratch/keane/human_behaviour/v6_rha_residual_nogamma_noconf/rla_mosei_senti/step_8164"
REPO_ID="keentomato/omnisapiens_bam_sentiment_polarity_mosei"
BACKBONE_NAME="Qwen/Qwen2.5-Omni-7B"
LABEL_SCHEME="/home/keaneong/human-behavior/verl/sft/label_maps/unified_label_map_v6.json"
SAVE_DIR="/scratch/keane/human_behaviour/hf_staging_bam"
# Task controls README title/tags/domain example.
# Choices: sarcasm | emotion | sentiment | humour | mental_health | generic
TASK="sentiment"

if [[ "$1" == "--readme-only" ]]; then
    python upload_to_hf.py \
        --ckpt_dir   "$CKPT_DIR" \
        --repo_id    "$REPO_ID" \
        --save_dir   "$SAVE_DIR" \
        --task       "$TASK" \
        --dataset_repo "keentomato/human_behavior_atlas" \
        --readme_only
else
    python upload_to_hf.py \
        --ckpt_dir   "$CKPT_DIR" \
        --repo_id    "$REPO_ID" \
        --backbone_name "$BACKBONE_NAME" \
        --label_scheme  "$LABEL_SCHEME" \
        --save_dir   "$SAVE_DIR" \
        --task       "$TASK" \
        --dataset_repo "keentomato/human_behavior_atlas" \
        --private
fi
