#!/usr/bin/env bash
# Self-evolving SFT-distillation smoke for google/gemma-4-E4B-it on B200.
#
# Mirrors run_qwen35_9b_sft.sh but:
#   - student model = google/gemma-4-E4B-it (gated; needs HF_TOKEN with access)
#   - forces vLLM flash_attn_version=2: vLLM 0.22.1rc1's FA4 "cute" kernel
#     (flash_fwd_sm100) is broken on Blackwell/sm100 and crashes the rollout
#     engine at init regardless of attention backend. FA2 sidesteps it.
#   - actor attn_implementation=sdpa + use_remove_padding=False: Gemma's head_dim
#     exceeds FlashAttention-2's 256 limit (RuntimeError in flash_attn_varlen),
#     and verl has no gemma remove-padding monkey-patch, so use the padded SDPA
#     path for the training/actor forward.
#   - wandb disabled + console logger (no W&B key on the cluster pods)
#   - batch sizes / freqs mirror run_qwen35_9b_sft.sh; runs effectively forever
#     (huge total_training_steps/epochs) with full-dataset eval (val_max_samples=-1)
#
# Requires the SFT-mode gen server up first (launch_gen_server_sft.sh, :8004).
# Invoke:  HF_TOKEN=<tok> API_KEY=<trapi> bash scripts/self_evolving/run_gemma4_e4b_sft.sh
set -euo pipefail

export CHAT_PROVIDER=trapi
export API_BASE="${API_BASE:-http://point.dd.works:18890/v1}"
export API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key 2>/dev/null || true)}"
export MODEL_NAME="${MODEL_NAME:-gpt-5.1-chat_2025-11-13}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_gemma4_e4b_sft}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8004}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
REPO_ROOT="${REPO_ROOT:-/scratch/sheng/self_evolving/verl}"
cd "$REPO_ROOT"

REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-8192}"

"$PYTHON_BIN" -m verl.trainer.main_sft_evolving \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_sft_dataset.py \
    data.custom_cls.name=SelfEvolvingSFTDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-32}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    ++data.val_max_samples="${VAL_MAX_SAMPLES:--1}" \
    data.max_length="$SFT_MAX_LENGTH" \
    data.pad_mode=right \
    data.truncation=left \
    data.shuffle=False \
    data.image_key=images \
    +data.self_evolving.gen_server_url="$GEN_SERVER_URL" \
    +data.self_evolving.dataset_length=100000 \
    +data.feedback.enable=true \
    +data.feedback.batch_size="${FEEDBACK_BATCH_SIZE:-32}" \
    +data.feedback.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    +data.feedback.custom_cls.name=SelfEvolvingDataset \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$API_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$API_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$MODEL_NAME" \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key="$EMBED_API_KEY" \
    +reward.custom_reward_function.reward_kwargs.embed_model="$EMBED_MODEL" \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="$GEN_SERVER_URL" \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH:-google/gemma-4-E4B-it}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    +actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM:-0.6}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-16384}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.enable=False \
    trainer.n_gpus_per_node="${N_GPUS:-4}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${TOTAL_EPOCHS:-1000000}" \
    trainer.total_training_steps="${TOTAL_STEPS:-1000000000}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR:-$DATA_DIR/../logs/val_generations/$EXPERIMENT_NAME}" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console"]' \
    "+ray_init.address=local" \
    "$@"
