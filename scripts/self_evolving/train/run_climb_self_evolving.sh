#!/usr/bin/env bash
# Self-evolving MULTIMODAL training on CLIMB + text seeds, with per-modality
# CLIMB evaluation (accuracy + class-macro F1, macro-averaged across modalities).
#
# Companion services (launch first, separate panes):
#   1. CLIMB media file server on the LOCAL node (mib), where the files live:
#        export CLIMB_FILE_TOKEN=sk-...           # same value as the teacher key
#        bash scripts/self_evolving/serve/start_climb_file_server.sh
#   2. Generation server (multimodal-enabled) near the chat/teacher endpoint:
#        CLIMB_FILE_TOKEN=sk-... GEN_MM_TARGET=0.5 \
#        CLIMB_SEEDS_PATH=$CLIMB_DIR/train_seeds.jsonl \
#        CLIMB_FILE_BASE=http://mib.media.mit.edu:18080 \
#        MILVUS_COLLECTION=medical_knowledge_v2 \
#        bash scripts/self_evolving/serve/start_generation_server.sh
#
# Build the data products once:
#   python scripts/self_evolving/preprocess_climb.py --split valid --out_dir $CLIMB_DIR
#   python scripts/self_evolving/preprocess_climb.py --split train --out_dir $CLIMB_DIR
#
# The reward function is the SAME dispatcher (self_evolving.compute_score): it
# routes climb_* data sources to the deterministic CLIMB scorer and text rows to
# the medical judge. The CLIMB media token is read from CLIMB_FILE_TOKEN (env),
# never hardcoded.

set -xeuo pipefail

export API_BASE="${API_BASE:-http://localhost:8005/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export CHAT_PROVIDER="${CHAT_PROVIDER:-vllm}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
# The reward judge / question teacher can be a STRONGER model than the solver
# (defaults to the solver model for backward compat). Decoupled so the trainer
# can keep a small solver while a big teacher runs on a separate node.
export JUDGE_MODEL="${JUDGE_MODEL:-$MODEL_NAME}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/climb_datasets}"
export CLIMB_DIR="${CLIMB_DIR:-$DATA_DIR}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-climb_qwen36_27b_self_evolving}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8004}"

# CLIMB media file server (LOCAL node). Token comes from the env, not the CLI.
export CLIMB_FILE_BASE="${CLIMB_FILE_BASE:-http://mib.media.mit.edu:18080}"
export CLIMB_FILE_TOKEN_ENV="${CLIMB_FILE_TOKEN_ENV:-CLIMB_FILE_TOKEN}"
CLIMB_MEDIA_CACHE="${CLIMB_MEDIA_CACHE:-$DATA_DIR/media_cache}"
CLIMB_MAX_PIXELS="${CLIMB_MAX_PIXELS:-1048576}"

# Trainer GPUs (the gen-server's vLLM teacher takes the other half of the node).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
N_GPUS="${N_GPUS:-4}"
TP_SIZE="${TP_SIZE:-4}"

REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"

if [ -d /home/dvdai/miniconda3/envs/cu130/lib ]; then
    export LD_LIBRARY_PATH="/home/dvdai/miniconda3/envs/cu130/lib:${LD_LIBRARY_PATH:-}"
fi
export RAY_ADDRESS="local"

PYTHON_BIN="${PYTHON_BIN:-/home/dvdai/miniconda3/envs/cu130/bin/python}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$DATA_DIR/val_generations/$EXPERIMENT_NAME}"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$CLIMB_DIR/train_seeds.jsonl" \
    data.val_files="$CLIMB_DIR/val_mini.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.shuffle=False \
    data.val_batch_size=64 \
    data.image_key=images \
    data.video_key=videos \
    data.truncation=left \
    data.filter_overlong_prompts=False \
    +data.self_evolving.gen_server_url="$GEN_SERVER_URL" \
    +data.self_evolving.dataset_length=100000 \
    +data.climb.file_base="$CLIMB_FILE_BASE" \
    +data.climb.file_token_env="$CLIMB_FILE_TOKEN_ENV" \
    +data.climb.cache_dir="$CLIMB_MEDIA_CACHE" \
    +data.climb.max_pixels="$CLIMB_MAX_PIXELS" \
    +data.climb.video_frames="${CLIMB_VIDEO_FRAMES:-6}" \
    +data.climb.video_max_pixels="${CLIMB_VIDEO_MAX_PIXELS:-200704}" \
    +data.climb.force_multimodal=True \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$API_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$API_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$JUDGE_MODEL" \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key="$EMBED_API_KEY" \
    +reward.custom_reward_function.reward_kwargs.embed_model="$EMBED_MODEL" \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="$GEN_SERVER_URL" \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=4096 \
    actor_rollout_ref.model.path="$MODEL_NAME" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BSZ:-2}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.32 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.001 \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-True}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-True}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BSZ:-4}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BSZ:-4}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-True}" \
    critic.enable=False \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_STEPS:-500}" \
    trainer.test_freq="${TEST_FREQ:-20}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.save_freq=20 \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "+ray_init.address=local" \
    "$@"
