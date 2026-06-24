#!/usr/bin/env bash
# CLIMB self-evolving *SFT-distillation* (multimodal). Same shape as
# run_qwen36_27b_selfimprove_sft.sh (verl.trainer.main_sft_evolving, sft_loss on
# the teacher's verified <think>…</think>\boxed{answer} traces fetched from the
# --sft_mode gen server) but on the CLIMB dataset with per-modality eval.
#
# Companion services (launch first):
#   1. CLIMB media file server on mib + reverse tunnel to this node (localhost:18080).
#   2. Teacher vLLM (e.g. Qwen3.6-27B on server5) — gen server + reward judge call it.
#   3. SFT-mode gen server (climb args + --sft_mode), e.g.:
#        CLIMB_FILE_TOKEN=… API_KEY=… bash /root/climb_gen_sft_launch.sh
#
# CLIMB media: climb:// handles resolved + every <video> flattened to image
# frames in SelfEvolvingSFTDataset (train) and RLHFDataset (val) — image-only
# format end to end. Per-modality accuracy + class-macro F1 come from the
# inherited RayPPOTrainer._validate climb hook (val set = val_mini.jsonl).
set -xeuo pipefail
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"

export CHAT_PROVIDER="${CHAT_PROVIDER:-vllm}"
export API_BASE="${API_BASE:-http://localhost:8100/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"          # teacher / reward judge
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$MODEL_NAME}"   # the SFT student
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/climb_datasets}"
export CLIMB_DIR="${CLIMB_DIR:-$DATA_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-climb_qwen36_27b_self_evolving_sft_${RUN_TS}}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8005}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"

# CLIMB media file server (remote node, via reverse tunnel). Token from env.
export CLIMB_FILE_BASE="${CLIMB_FILE_BASE:-http://localhost:18080}"
export CLIMB_FILE_TOKEN_ENV="${CLIMB_FILE_TOKEN_ENV:-CLIMB_FILE_TOKEN}"
CLIMB_MEDIA_CACHE="${CLIMB_MEDIA_CACHE:-$DATA_DIR/media_cache}"
CLIMB_MAX_PIXELS="${CLIMB_MAX_PIXELS:-1048576}"

REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VALIDATION_DATA_DIR="/scratch/sheng/self_evolving/logs_sft_climb/val_generations/$EXPERIMENT_NAME"
cd "$REPO_ROOT"

SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-12288}"

"$PYTHON_BIN" -m verl.trainer.main_sft_evolving \
    data.train_files="$CLIMB_DIR/train_seeds.jsonl" \
    data.val_files="$CLIMB_DIR/val_mini.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_sft_dataset.py \
    data.custom_cls.name=SelfEvolvingSFTDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    ++data.val_max_samples="${VAL_MAX_SAMPLES:-512}" \
    data.max_length="$SFT_MAX_LENGTH" \
    data.pad_mode=right \
    data.truncation=left \
    data.shuffle=False \
    data.image_key=images \
    data.video_key=videos \
    +data.self_evolving.gen_server_url="$GEN_SERVER_URL" \
    +data.self_evolving.dataset_length=100000 \
    +data.climb.file_base="$CLIMB_FILE_BASE" \
    +data.climb.file_token_env="$CLIMB_FILE_TOKEN_ENV" \
    +data.climb.cache_dir="$CLIMB_MEDIA_CACHE" \
    +data.climb.max_pixels="$CLIMB_MAX_PIXELS" \
    +data.climb.video_frames="${CLIMB_VIDEO_FRAMES:-6}" \
    +data.climb.video_max_pixels="${CLIMB_VIDEO_MAX_PIXELS:-200704}" \
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
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    +actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-4}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    critic.enable=False \
    trainer.n_gpus_per_node="${N_GPUS:-8}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_STEPS:-500}" \
    trainer.test_freq="${TEST_FREQ:-20}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.save_freq="${SAVE_FREQ:-20}" \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "+ray_init.address=local" \
    "$@" 2>&1 | tee "/root/sft_train_${EXPERIMENT_NAME}.log"
