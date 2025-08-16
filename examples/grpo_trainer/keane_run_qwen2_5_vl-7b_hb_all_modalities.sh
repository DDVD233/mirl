set -x

unset ROCR_VISIBLE_DEVICES

# Qwen/Qwen2.5-VL-7B-Instruct
# Qwen/Qwen2.5-Omni-7B
# data.train_files=/scratch/keane/human_behaviour/human_behaviour_data/train_no_meld_no_chalearn_vision_v2_template_prompts.jsonl \
# data.val_files=/scratch/keane/human_behaviour/human_behaviour_data/val_no_meld_no_chalearn_vision_v2_template_prompts.jsonl \
# data.modalities=\'audio,videos\' \

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 PYTHONPATH="/home/keaneong/human-behavior/verl:$PYTHONPATH" python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/scratch/keane/human_behaviour/human_behaviour_data/old_train_template_prompts.jsonl \
    data.val_files=/scratch/keane/human_behaviour/human_behaviour_data/old_val_template_prompts.jsonl \
    data.train_batch_size=18 \
    data.val_batch_size=6 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.image_key=images \
    data.video_key=videos \
    data.prompt_key=problem \
    data.dataloader_num_workers=0 \
    data.modalities=\'videos\' \
    data.format_prompt=/home/keaneong/human-behavior/verl/examples/format_prompt/default.jinja \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Omni-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1e-8 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
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
    trainer.test_freq=2 \
    trainer.total_epochs=15 $@