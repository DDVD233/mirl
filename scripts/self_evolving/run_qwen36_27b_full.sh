#!/usr/bin/env bash
# Full Qwen3.6-27B training on node3400 (4xH200) with self-evolving generation
# server. Chat server / judge runs separately on node2500 (currently
# Qwen3.6-35B-A3B MoE).

set -xeuo pipefail

export API_BASE="http://node2500:8002/v1"
export API_KEY="EMPTY"
# Judge / chat model — name string used by reward_function for the LLM judges.
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
export DATA_DIR="/home/dvdai/scratch/dvdai/self_evolving_datasets/mimiciv_rare"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_qwen36_27b}"
export BIOBERT_API_BASE="http://localhost:8003"
export GEN_SERVER_URL="http://localhost:8004"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# cu130 flashinfer-cubin needs conda's libstdc++ (GLIBCXX_3.4.26 not in /lib64)
export LD_LIBRARY_PATH="/home/dvdai/miniconda3/envs/cu130/lib:${LD_LIBRARY_PATH:-}"

# Fresh local Ray cluster (default /tmp/ray; AF_UNIX path stays under the
# 107-byte limit). Don't auto-attach to anyone else's cluster.
export RAY_ADDRESS="local"

PYTHON_BIN="/home/dvdai/miniconda3/envs/cu130/bin/python"

cd /home/dvdai/verl

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.shuffle=False \
    data.val_batch_size=64 \
    data.image_key=images \
    data.truncation=right \
    +data.self_evolving.gen_server_url="$GEN_SERVER_URL" \
    +data.self_evolving.dataset_length=100000 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$API_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$API_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$MODEL_NAME" \
    +reward.custom_reward_function.reward_kwargs.biobert_api_base="$BIOBERT_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="$GEN_SERVER_URL" \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=8192 \
    actor_rollout_ref.model.path=Qwen/Qwen3.6-27B \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir=/home/dvdai/scratch/dvdai/self_evolving_datasets/logs/val_generations/$EXPERIMENT_NAME \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "+ray_init.address=local" \
    "$@"
