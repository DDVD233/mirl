#!/usr/bin/env bash
# Run self-evolving medical agent training (NO LABEL variant).
# Same as run_self_evolving.sh but ground truth labels are not used.
# The reward function uses an LLM judge to determine correctness.
#
# Prerequisites:
#   1. Start vLLM server: bash scripts/self_evolving/serve/start_vllm_server.sh
#   2. Preprocess data:   python scripts/self_evolving/preprocess_pubmedqa.py
#
# Environment variables:
#   API_BASE       - vLLM server URL (default: http://localhost:8000/v1)
#   API_KEY        - API key (default: EMPTY)
#   MODEL_NAME     - Model name (default: Qwen/Qwen3-VL-2B-Instruct)
#   MASK_CONTEXT   - True=no context to proposer, False=context but no answer (default: True)
#   EXPERIMENT_NAME - wandb experiment name (default: pubmedqa_no_label)
#
# Variant 1 (no context): MASK_CONTEXT=True EXPERIMENT_NAME=pubmedqa_no_label_no_ctx bash ...
# Variant 2 (with context): MASK_CONTEXT=False EXPERIMENT_NAME=pubmedqa_no_label_with_ctx bash ...

set -xeuo pipefail

export API_BASE="${API_BASE:-http://localhost:8000/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
export DATA_DIR="${DATA_DIR:-/scratch/self_evolving_datasets/pubmedqa}"

# Force fresh Ray cluster (don't connect to existing stale clusters)
unset RAY_ADDRESS 2>/dev/null || true

# Step 1: Preprocess PubMedQA if not already done
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "Preprocessing PubMedQA dataset..."
    python scripts/self_evolving/preprocess_pubmedqa.py --output_dir "$DATA_DIR"
fi

# Step 2: Run GRPO training with self-evolving dataset
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.shuffle=False \
    data.val_batch_size=200 \
    +data.self_evolving.api_base="$API_BASE" \
    +data.self_evolving.api_key="$API_KEY" \
    +data.self_evolving.model_name="$MODEL_NAME" \
    +data.self_evolving.questions_per_target=5 \
    +data.self_evolving.accuracy_window=32 \
    +data.self_evolving.dataset_length=100000 \
    +data.self_evolving.no_label=True \
    +data.self_evolving.no_label_mask_context="${MASK_CONTEXT:-True}" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=20 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=262144 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    critic.enable=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1000 \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="${EXPERIMENT_NAME:-pubmedqa_no_label}" \
    'trainer.logger=["console","wandb"]' \
    "$@"
