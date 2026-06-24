#!/usr/bin/env bash
# Gemma-4-31B-it SELF-IMPROVE GRPO, SPLIT across two nodes (the layout that avoids the
# co-location NCCL crash): the gen-server backend + reward judge is a SEPARATE key-gated
# gemma-4-31B vLLM on server2 (point.dd.works:18187); this trainer owns all 4 GPUs on
# server4. Resource config is IDENTICAL to the proven run_gemma4_31b_rl.sh (4x B200,
# colocated rollout, param_offload). Only the judge/gen endpoints differ (vllm->server2).
# Gemma head_dim=256 -> sdpa + use_remove_padding=False; fixed micro-batch=1 (dynamic-bsz
# OOMs the 31B sdpa forward); rollout mem 0.3; B200 FA2. Multimodal val on mimiciv_rare/test.jsonl.
set -xeuo pipefail
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CHAT_PROVIDER=vllm
export API_BASE="${API_BASE:-http://point.dd.works:18187/v1}"
export API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.gen_host_key 2>/dev/null || echo EMPTY)}"
export MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_gemma4_31b_selfimprove_stock_${RUN_TS}}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8005}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
REPO_ROOT="${REPO_ROOT:-/scratch/sheng/self_evolving/verl}"
LOG_DIR=/scratch/sheng/self_evolving/logs_si_gemma31b_stock; mkdir -p "$LOG_DIR"
VALIDATION_DATA_DIR="$LOG_DIR/val_generations/$EXPERIMENT_NAME"
cd "$REPO_ROOT"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.max_prompt_length=8192 \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-3072}" \
    data.shuffle=False \
    data.val_batch_size=32 \
    ++data.val_max_samples="${VAL_MAX_SAMPLES:-64}" \
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
    +reward.reward_kwargs.max_resp_len=3072 \
    actor_rollout_ref.model.path=google/gemma-4-31B-it \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO:-1}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.32 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.001 \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM:-0.3}" \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "+ray_init.address=local" \
    "$@" 2>&1 | tee "$LOG_DIR/train_${EXPERIMENT_NAME}.log"
