#!/usr/bin/env bash
# Multi-agent self-evolving training on MIMIC-IV rare-disease primary-diagnosis QA.
#
# Train: train.jsonl (used as seeds for the self-evolving question-generation pipeline)
# Val:   test.jsonl  (real admissions with chest X-ray + 12-lead ECG, evaluated directly via RLHFDataset)
#
# Token budget:
#   - 8K context (max_model_len)
#   - 6K prompt cap (text + multimodal tokens)
#   - 2K response
#
# Prerequisites (on vps3):
#   - vLLM chat API on GPU 0
#   - vLLM embedding API on GPU 1
#   - Milvus reachable at mib.media.mit.edu:19531
#   - Dataset preprocessed under DATA_DIR

set -xeuo pipefail

export API_BASE="${API_BASE:-http://localhost:8000/v1}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://localhost:8001/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
export MILVUS_TOKEN="${MILVUS_TOKEN:-root:Milvus}"
export DATA_DIR="${DATA_DIR:-/scratch/self_evolving_datasets/mimiciv_rare}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_multi_agent_qwen3vl8b}"
# BioBERT similarity server (optional; reward gracefully degrades to 0 if unset).
export BIOBERT_API_BASE="${BIOBERT_API_BASE:-http://localhost:8003}"

unset RAY_ADDRESS 2>/dev/null || true

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "ERROR: $DATA_DIR/train.jsonl not found. Run preprocess_mimiciv_rare.py first." >&2
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
    +data.self_evolving.api_base="$API_BASE" \
    +data.self_evolving.api_key="$API_KEY" \
    +data.self_evolving.model_name="$MODEL_NAME" \
    +data.self_evolving.embed_api_base="$EMBED_API_BASE" \
    +data.self_evolving.embed_model="$EMBED_MODEL" \
    +data.self_evolving.milvus_uri="$MILVUS_URI" \
    +data.self_evolving.milvus_token="$MILVUS_TOKEN" \
    +data.self_evolving.milvus_collection=medical_knowledge \
    +data.self_evolving.milvus_top_k=16 \
    +data.self_evolving.n_queries=10 \
    +data.self_evolving.questions_per_query=1 \
    +data.self_evolving.accuracy_window=32 \
    +data.self_evolving.dataset_length=100000 \
    +data.self_evolving.no_label=False \
    +data.self_evolving.log_dir=/scratch/self_evolving_datasets/logs \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$API_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$API_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$MODEL_NAME" \
    +reward.custom_reward_function.reward_kwargs.biobert_api_base="$BIOBERT_API_BASE" \
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
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "$@"
