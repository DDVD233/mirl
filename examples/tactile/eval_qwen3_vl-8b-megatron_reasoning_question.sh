#!/bin/bash
set -x

# Evaluate reasoning RL checkpoint on full test set — split: question

CKPT_DIR="checkpoints/tactile/qwen3vl_dapo_reasoning_split_question_official"
LATEST_STEP=$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")
MODEL_PATH="${CKPT_DIR}/global_step_${LATEST_STEP}/actor/huggingface"
echo "Evaluating model at: ${MODEL_PATH}"

test_path="$HOME/scratch/raofu/3DHaptic/annotation_verl_split_question_test.json"

mkdir -p outputs/eval_qwen3vl_dapo_reasoning_split_question

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$test_path" \
    data.val_files="$test_path" \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.experiment_name='eval_qwen3vl_dapo_reasoning_split_question' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_only=True \
    trainer.validation_data_dir=outputs/eval_qwen3vl_dapo_reasoning_split_question \
    trainer.total_epochs=15 $@