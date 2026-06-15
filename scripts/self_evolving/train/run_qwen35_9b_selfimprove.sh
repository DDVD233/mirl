#!/usr/bin/env bash
# Qwen3.5-9B RL "self-improvement" on B200 (server 2). The SAME model plays every
# role: proposer (gen server), solver (the RL student's own rollouts), and judge
# (reward). Trained with GRPO + KL-Cov entropy-collapse control + a vLLM
# sampling-time repetition penalty.
#
# Layout (server 2, 4x B200):
#   GPU 0,1 -> standalone vLLM OpenAI server (Qwen3.5-9B, TP=2) on :8100
#              => serves the gen-server proposer/validator AND the reward judges
#   GPU 2,3 -> this GRPO trainer (FSDP2 actor + colocated vLLM rollout, TP=2)
#   gen server (RL mode, no teacher trace) on :8005
#
# Qwen3.5-9B specifics on this stack:
#   - text head_dim=256 => the actor forward must avoid FlashAttention's
#     head_dim<=256 varlen kernel: attn_implementation=sdpa + use_remove_padding=False
#     (same constraint hit by gemma-4-E4B).
#   - vLLM 0.22.1's FA4 "cute" kernel (flash_fwd_sm100) is broken on B200, so the
#     colocated rollout pins flash_attn_version=2.
#
# Run the vLLM judge server (:8100) and the RL-mode gen server (:8005) FIRST.
set -xeuo pipefail

# Judge + proposer endpoint = the LOCAL vLLM server (NOT trapi). CHAT_PROVIDER=vllm
# so the gen server and the reward judges both emit vLLM-style payloads (max_tokens
# + chat_template_kwargs enable_thinking), not Azure max_completion_tokens.
export CHAT_PROVIDER="${CHAT_PROVIDER:-vllm}"
export API_BASE="${API_BASE:-http://localhost:8100/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare_si}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_qwen35_9b_selfimprove}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8005}"
# Embedding endpoint for the reward's embed_sim surrogate (same one the gen
# server's retriever uses); lives on mib, reachable from both B200 nodes.
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"

REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
REPO_ROOT="${REPO_ROOT:-/scratch/sheng/self_evolving/verl}"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
    data.shuffle=False \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    ++data.val_max_samples="${VAL_MAX_SAMPLES:--1}" \
    data.image_key=images \
    data.truncation=left \
    +data.self_evolving.gen_server_url="$GEN_SERVER_URL" \
    +data.self_evolving.dataset_length=100000 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$API_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$API_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$MODEL_NAME" \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key="$EMBED_API_KEY" \
    +reward.custom_reward_function.reward_kwargs.embed_model="$EMBED_MODEL" \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="$GEN_SERVER_URL" \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH:-2048}" \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH:-Qwen/Qwen3.5-9B}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio="${KL_COV_RATIO:-0.001}" \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef="${PPO_KL_COEF:-1.0}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    +actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM:-0.6}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-32768}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    critic.enable=False \
    trainer.n_gpus_per_node="${N_GPUS:-2}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.total_training_steps="${TOTAL_STEPS:-500}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR:-/scratch/sheng/self_evolving/logs_selfimprove/val_generations/$EXPERIMENT_NAME}" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "+ray_init.address=local" \
    "$@"
