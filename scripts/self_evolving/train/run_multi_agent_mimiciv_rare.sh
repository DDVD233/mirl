#!/usr/bin/env bash
# Multi-agent self-evolving training on MIMIC-IV rare-disease primary-diagnosis QA.
#
# The full QueryProposer / Generator / Validator / Milvus pipeline now lives in
# an external generation server (scripts/self_evolving/generation_server.py).
# Start it FIRST in another pane via start_generation_server.sh; this script
# only needs the server URL to fetch fresh samples and report accuracy back.
#
# Train: train.jsonl (used as seeds inside the generation server)
# Val:   test.jsonl  (real admissions, evaluated directly via RLHFDataset)
#
# Token budget:
#   - 8K context (max_model_len)
#   - 6K prompt cap (text + multimodal tokens)
#   - 2K response

set -xeuo pipefail

export API_BASE="${API_BASE:-http://localhost:8000/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
export DATA_DIR="${DATA_DIR:-/scratch/self_evolving_datasets/mimiciv_rare}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_multi_agent_qwen3vl8b}"
# BioBERT similarity server (optional; reward gracefully degrades to 0 if unset).
export BIOBERT_API_BASE="${BIOBERT_API_BASE:-http://localhost:8003}"
# Generation server: must be running before this script starts.
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8004}"

unset RAY_ADDRESS 2>/dev/null || true

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: $DATA_DIR/train.jsonl not found. Run preprocess_mimiciv_rare.py first." >&2
    exit 1
fi

# Fail fast if the generation server isn't up yet — saves Hydra/Ray startup time
# when the operator forgot to launch start_generation_server.sh.
if ! curl -fsS --max-time 5 "$GEN_SERVER_URL/healthz" >/dev/null 2>&1; then
    echo "ERROR: generation server not reachable at $GEN_SERVER_URL/healthz" >&2
    echo "       Start it first with scripts/self_evolving/serve/start_generation_server.sh" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.shuffle=False \
    data.val_batch_size=200 \
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
    +reward.reward_kwargs.overlong_buffer_cfg.len=256 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-8B-Instruct \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    critic.enable=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir=/home/dvdai/scratch/dvdai/self_evolving_datasets/logs/val_generations/mimiciv_rare \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "$@"
