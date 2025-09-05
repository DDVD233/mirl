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

# Mean would be 1659
# meanstd would be 3318
# meanstdp25p75 would be 6636

# we have a few different options; base_only, residual_only, joint

# --load_checkpoint_path "/scratch/keane/human_behaviour/debug_rla/step_1998"
# for lora we put 1e-4
# but for rla we put 1e-3

# CONFIDENCE GAIN IS BASICALLY WHETHER OR NOT WE SCALE THE CONFIDENCE OF THE LOGITS THAT ARE FED INTO THE ADAPTERSS
  # --rla_video_use_ln \
  # --rla_audio_use_ln \
  # --rla_video_use_conf_gain \
  # --rla_video_conf_init_gain 3.0 \
  # --rla_audio_use_conf_gain \
  # --rla_audio_conf_init_gain 3.0 \
  # --rla_video_alpha_init 2.0 \
  # --rla_audio_alpha_init 2.0 \


accelerate launch --config_file configs/accelerate_config_qwen.yaml train_rla_multi_head.py \
  --mode train \
  --rla_resume_diff_training_stage \
  --training_strategy lora \
  --train_batch_size 1 \
  --val_batch_size 2 \
  --test_batch_size 2 \
  --lr 1e-4 \
  --hard_gamma 0.0 \
  --base_lr 1e-5 \
  --rla_lr  5e-4 \
  --epochs 10 \
  --train_file "/scratch/keane/human_behaviour/human_behaviour_data/0.2_feat_meld_train.jsonl" \
  --val_file   "/scratch/keane/human_behaviour/human_behaviour_data/feat_meld_val.jsonl" \
  --test_file  "/scratch/keane/human_behaviour/human_behaviour_data/feat_meld_test.jsonl" \
  --label_map_path "/home/keaneong/human-behavior/verl/multi_task_classification/unified_feat_meld_label_map.json" \
  --save_every_n_epochs 1 \
  --save_every_n_steps 99999999 \
  --load_checkpoint_path "/scratch/keane/human_behaviour/debug_rla/step_1998" \
  --save_checkpoint_dir "/scratch/keane/human_behaviour/resume_debug_rla_residual_res_only" \
  --validation_result_dir "/scratch/keane/human_behaviour/debug_rla_residual_res_only/validation_results" \
  --validate_every_n_epochs 1 \
  --validate_every_n_steps 999999 \
  --early_stopping_patience 99999 \
  --project "debug-rla-omni-classifier-multi-head-lora" \
  --gradient_accumulation_steps 8 \
  --rla_stage joint \
  --use_rla_video \
  --use_rla_audio \
  --d_video_feat 3318 \
  --d_audio_feat 6373 \
  --rla_hidden_video 256 \
  --rla_hidden_audio 512 \
  --rla_p_moddrop_video 0.20 \
  --rla_p_moddrop_audio 0.20 \
  --rla_video_temporal meanstd \
  --rla_video_norm none \
  --rla_audio_norm l2 \
  --rla_audio_temporal none \
  --rla_video_alpha_init 2.0 \
  --rla_audio_alpha_init 2.0 \
  --rla_video_use_ln \
  --rla_audio_use_ln \     

echo "Run finished."