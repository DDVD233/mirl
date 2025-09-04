#!/bin/bash
echo "Starting LORA + RLA training..."
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# For videos, use this:
#   --use_rla_video \
#   --rla_stage joint \
#   --d_video_feat 210 \
#   --rla_hidden 128 \
#   --rla_p_moddrop_video 0.30

# For audio, use this:
  # --use_rla_audio \
  # --rla_stage residual_only \
  # --d_audio_feat 88 \
  # --rla_hidden 128 \
  # --rla_p_moddrop_audio 0.30

# If you want to 

accelerate launch --config_file configs/accelerate_config_qwen.yaml train_rla_multi_head.py \
  --mode train \
  --training_strategy lora \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --test_batch_size 2 \
  --lr 1e-4 \
  --epochs 10 \
  --train_file "/scratch/keane/human_behaviour/human_behaviour_data/feat_meld_train.jsonl" \
  --val_file   "/scratch/keane/human_behaviour/human_behaviour_data/feat_meld_val.jsonl" \
  --test_file  "/scratch/keane/human_behaviour/human_behaviour_data/feat_meld_test.jsonl" \
  --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/v2_unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.json" \
  --save_every_n_epochs 1 \
  --save_every_n_steps 999999 \
  --save_checkpoint_dir "/scratch/keane/human_behaviour/multi_task_lora_training" \
  --validation_result_dir "/scratch/keane/human_behaviour/multi_task_lora_training/validation_results" \
  --validate_every_n_epochs 1 \
  --validate_every_n_steps 999999 \
  --early_stopping_patience 99999 \
  --project "debug-rla-omni-classifier-multi-head-lora" \
  --gradient_accumulation_steps 128 \
  \
  --use_rla_video \
  --rla_stage residual_only \
  --d_video_feat 3318 \
  --rla_hidden 128 \
  --rla_p_moddrop_video 0.30 \
  --rla_video_temporal meanstd \
  --rla_video_use_conf \
  
  # NOTE: the d_video_feat will depend on the meanstd or mean modes etc. 
  # (optional) uncomment if you want the pre-MLP
  # --rla_video_use_mlp \
  # --rla_video_mlp_hidden 256 \
  # --rla_video_out_dim 256


echo "Run finished."