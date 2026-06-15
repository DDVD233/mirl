#!/usr/bin/env bash
# Baseline: standard GRPO training on PubMedQA (no self-evolving proposer).
# Uses the same reward function, model, and hyperparameters as run_self_evolving.sh
# but trains directly on the PubMedQA train set with the default RLHFDataset.
#
# Prerequisites:
#   1. Start vLLM server: bash scripts/self_evolving/serve/start_vllm_server.sh
#   2. Preprocess data:   python scripts/self_evolving/preprocess_pubmedqa.py

set -xeuo pipefail

export API_BASE="${API_BASE:-http://localhost:8000/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
export DATA_DIR="${DATA_DIR:-/scratch/self_evolving_datasets/pubmedqa}"

# Force fresh Ray cluster
unset RAY_ADDRESS 2>/dev/null || true

# Preprocess PubMedQA if not already done
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "Preprocessing PubMedQA dataset..."
    python scripts/self_evolving/preprocess_pubmedqa.py --output_dir "$DATA_DIR"
fi

# Standard GRPO training on PubMedQA train set (no custom dataset class)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.train_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.shuffle=True \
    data.val_batch_size=200 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$API_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$API_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$MODEL_NAME" \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    critic.enable=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=1000 \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name=pubmedqa_baseline_grpo \
    'trainer.logger=["console","wandb"]' \
    "$@"
