set -x

unset ROCR_VISIBLE_DEVICES

# actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct
# actor_rollout_ref.model.path=Qwen/Qwen2.5-Omni-7B
# data.train_files=/scratch/keane/human_behaviour/human_behaviour_data/train_no_meld_no_chalearn_vision_v2_template_prompts.jsonl \
# data.val_files=/scratch/keane/human_behaviour/human_behaviour_data/val_no_meld_no_chalearn_vision_v2_template_prompts.jsonl \
# data.modalities=\'audio,videos\' \

# SETTING OF SAVE PATH: trainer.default_local_dir= /scratch/keane/human_behaviour/2_models_hb_vision_only
# SETTING OF THE LOAD PATH from directory of checkpoints is also: trainer.default_local_dir

# TRAINING FROM scratch: trainer.resume_mode ==  "disable" (default will save into default_local_dir)

# TRAINING AUTOMATICALLY (i.e. from scratch or from latest checkpoint) : 
    # trainer.resume_mode == "auto" and then the model will take the latest ckpt from trainer.default_hdfs_dir

# TRAINING from specific CHECKPOINT: trainer.resume_mode == "resume_path" and then specify trainer.resume_from_path
    # Setting of path to resume training from trainer.resume_from_path (exact path of checkpoint)
    # the model will take from resume_from_path directly (absolute path), and ignore default_hdfs_dir

# for validation, set val_before_train=True ; make sure that the checkpoint is loaded and put val_only=True
# the checkpoint should already be loaded before that
# and then we will just evaluate

# esconv
# /scratch/keane/human_behaviour/human_behaviour_data/esconv_only.jsonl
# /scratch/keane/human_behaviour/human_behaviour_data/no_esconv.jsonl

# chalearn
# /scratch/keane/human_behaviour/human_behaviour_data/chalearn_only.jsonl 
# /scratch/keane/human_behaviour/human_behaviour_data/no_chalearn.jsonl (Problematic)

# # chsimsv2
# /scratch/keane/human_behaviour/human_behaviour_data/chsimsv2_only.jsonl (works)
# /scratch/keane/human_behaviour/human_behaviour_data/no_chsimsv2.jsonl

# # tess
# /scratch/keane/human_behaviour/human_behaviour_data/tess_only.jsonl
# /scratch/keane/human_behaviour/human_behaviour_data/no_tess.jsonl

# # expw
# /scratch/keane/human_behaviour/human_behaviour_data/expw_only.jsonl
# /scratch/keane/human_behaviour/human_behaviour_data/no_expw.jsonl

# # meld
# /scratch/keane/human_behaviour/human_behaviour_data/meld_only.jsonl
# /scratch/keane/human_behaviour/human_behaviour_data/no_meld.jsonl

# # cremad
# /scratch/keane/human_behaviour/human_behaviour_data/cremad_only.jsonl
# /scratch/keane/human_behaviour/human_behaviour_data/no_cremad.jsonl

# old
# discretized_no_lmvd_no_chalearn_v3_template_prompts.jsonl

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 PYTHONPATH="/home/keaneong/human-behavior/verl:$PYTHONPATH" NCCL_ASYNC_ERROR_HANDLING=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/scratch/keane/human_behaviour/human_behaviour_data/no_chsimsv2.jsonl \
    data.val_files=/scratch/keane/human_behaviour/human_behaviour_data/subset_cremad_only.jsonl \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=3072 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.image_key=images \
    data.video_key=videos \
    data.prompt_key=problem \
    data.dataloader_num_workers=0 \
    data.modalities=\'audio,videos\' \
    data.format_prompt=/home/keaneong/human-behavior/verl/examples/format_prompt/default.jinja \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Omni-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1e-9 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=/home/keaneong/human-behavior/verl/examples/reward_function/medical.py \
    custom_reward_function.name=medical_compute_score_batch \
    reward_model.reward_manager=batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_hb' \
    trainer.experiment_name='vision_only' \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.val_before_train=False \
    trainer.test_freq=1 \
    trainer.total_epochs=15 $@ \
    trainer.default_local_dir=/scratch/keane/human_behaviour/verl_models_hb_vision_only