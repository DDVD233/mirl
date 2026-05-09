set -x
ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CUDA_LAUNCH_BLOCKING=1

# Evaluate non-reasoning RL checkpoint on full test set — split: task

GEN_TP=${GEN_TP:-4}
CP=${CP:-2}
TP=${TP:-2}
PP=${PP:-1}

CKPT_DIR="checkpoints/tactile/qwen3vl_dapo_split_task_official"
LATEST_STEP=$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")
MODEL_PATH="${CKPT_DIR}/global_step_${LATEST_STEP}/actor/huggingface"
echo "Evaluating model at: ${MODEL_PATH}"

test_path="$HOME/scratch/raofu/3DHaptic/annotation_verl_split_task_test.json"

mkdir -p outputs/eval_qwen3vl_dapo_split_task

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$test_path" \
    data.val_files="$test_path" \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=2048 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='tactile' \
    trainer.experiment_name='eval_qwen3vl_dapo_split_task' \
    trainer.n_gpus_per_node= \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.validation_data_dir=outputs/eval_qwen3vl_dapo_split_task \
    trainer.total_epochs=15 $@
