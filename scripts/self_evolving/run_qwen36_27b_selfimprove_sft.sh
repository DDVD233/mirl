#!/usr/bin/env bash
# Qwen3.6-27B SELF-IMPROVEMENT *SFT-distillation* on server3 (8x B200). The SAME
# model is the gen-server backend + proposer/validator + reward judge + the
# TEACHER that writes the reasoning traces (local vLLM :8100 on GPU0-3) AND the
# SFT student being trained on those traces (GPU4-7). Everything — the SFT
# trace, the judge, the difficulty feedback — is the model itself.
#
# Train step = supervised CE (sft_loss) on the teacher's verified
# <think>…</think>\boxed{answer} traces fetched from the SFT-mode gen server;
# vLLM is woken only for validation + generated-question feedback (-> /report).
#
# Memory knobs mirror the proven run_qwen36_27b_selfimprove.sh GRPO config
# (FSDP2 param/optimizer offload + grad checkpointing + dynamic-bsz + TP=4
# rollout @ 0.55). SFT's train step is lighter than GRPO (no rollout/advantage/
# critic/ref on the gradient path), so this fits inside the same envelope.
#
# Run the vLLM :8100 server FIRST, then serve_gen_qwen36_27b_sft.sh (SFT mode),
# then this.
set -xeuo pipefail
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
export CHAT_PROVIDER=vllm
export API_BASE="${API_BASE:-http://localhost:8100/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_qwen36_27b_selfimprove_sft_${RUN_TS}}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8005}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export WANDB_MODE="${WANDB_MODE:-online}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
REPO_ROOT="${REPO_ROOT:-/scratch/sheng/self_evolving/verl}"
VALIDATION_DATA_DIR="/scratch/sheng/self_evolving/logs_sft_qwen36/val_generations/$EXPERIMENT_NAME"
cd "$REPO_ROOT"

# prompt + teacher trace budget (mimiciv_rare prompts carry a long
# demographics/labs preamble, so keep headroom over max_prompt_length).
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-12288}"

"$PYTHON_BIN" -m verl.trainer.main_sft_evolving \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_sft_dataset.py \
    data.custom_cls.name=SelfEvolvingSFTDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    ++data.val_max_samples="${VAL_MAX_SAMPLES:-64}" \
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
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH:-Qwen/Qwen3.6-27B}" \
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
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    critic.enable=False \
    trainer.n_gpus_per_node="${N_GPUS:-4}" \
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
    "$@" 2>&1 | tee "/root/sft_train_qwen36_${EXPERIMENT_NAME}.log"
