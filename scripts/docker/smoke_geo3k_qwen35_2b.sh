#!/usr/bin/env bash
# Smoke test for the zjdavid/verl-selfevolving:cu130-vllm0.22.1 image:
# Qwen3.5-2B GRPO on geo3k (text+image), 1 GPU, 3 training steps.
#
# Expects to run INSIDE the container with /root/data/geo3k populated:
#   docker run --gpus '"device=6"' --rm -it --shm-size=8g \
#       -v /home/dvd/docker_data:/root/data \
#       zjdavid/verl-selfevolving:cu130-vllm0.22.1 \
#       bash scripts/docker/smoke_geo3k_qwen35_2b.sh

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-2B}
DATA_DIR=${DATA_DIR:-/root/data/geo3k}

cd /workspace/verl

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.image_key=images \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    critic.enable=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.save_freq=100 \
    trainer.project_name=docker_smoke \
    trainer.experiment_name=qwen35_2b_geo3k_smoke \
    'trainer.logger=["console"]' \
    "+ray_init.address=local" \
    "$@"
