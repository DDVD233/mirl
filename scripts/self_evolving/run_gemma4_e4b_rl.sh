#!/usr/bin/env bash
# Self-evolving GRPO RL for google/gemma-4-E4B-it on B200 (server1, 4x B200).
# RL counterpart of run_gemma4_e4b_sft.sh (SFT perf was poor -> switch to RL).
# Teacher infra via TRAPI (point.dd.works:18890):
#   - generation server  : gpt-5.5_2026-04-24   (supplies prompts; ~batch/step)
#   - reward judge        : gpt-5.1-chat_2025-11-13 (lighter, handles the high
#                           per-response judge volume)
# Body mirrors run_qwen35_9b_selfimprove.sh (GRPO + KL-Cov + repetition penalty),
# which already carries the head_dim>256 constraint Gemma shares:
#   actor attn_implementation=sdpa + use_remove_padding=False ; rollout TP=1,
#   flash_attn_version=2 (vLLM 0.22.1 FA4 cute kernel is broken on B200).
set -xeuo pipefail

export CHAT_PROVIDER=trapi
export API_BASE="${API_BASE:-http://point.dd.works:18890/v1}"
export API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key 2>/dev/null || true)}"
export MODEL_NAME="${MODEL_NAME:-gpt-5.1-chat_2025-11-13}"   # reward judge
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_gemma4_e4b_rl_gpt51judge_${RUN_TS}}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8004}"
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
cd "$REPO_ROOT"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    data.shuffle=False \
    data.val_batch_size="${VAL_BATCH_SIZE:-32}" \
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
    +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH:-4096}" \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH:-google/gemma-4-E4B-it}" \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM:-0.6}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-16384}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    critic.enable=False \
    trainer.n_gpus_per_node="${N_GPUS:-4}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.total_training_steps="${TOTAL_STEPS:-500}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR:-/scratch/sheng/self_evolving/logs/val_generations/$EXPERIMENT_NAME}" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "+ray_init.address=local" \
    "$@" 2>&1 | tee "/root/gemma_rl_${EXPERIMENT_NAME}.log"
