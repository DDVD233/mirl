#!/usr/bin/env bash
# SFT distillation: Qwen3.6-27B trained with supervised CE loss (sft_loss) on
# DISTILLATION TRACES — train input -> teacher <think>reasoning</think>\boxed{gt}.
# The supervised target is a verified reasoning trace (NOT just the bare boxed
# answer): built offline by scripts/self_evolving/make_distill_traces.py, which
# has the base Qwen3.6-27B solve each case (given the GT confidentially), keeps
# only traces that box the EXACT ground truth and never leak that the answer was
# given, at ~1000 tokens, with up to 3 diverse traces per question (the file is
# pre-shuffled). Consumed by StaticTraceSFTDataset. This is the SFT stage-1 init
# for the two-stage recipe (SFT -> RL via run_qwen36_27b_selfimprove_from_sft.sh).
#
# It uses the SAME validation set (test.jsonl) and is scored
# every TEST_FREQ steps by the SAME composite reward functions
# (verl/utils/reward_score/self_evolving.py::compute_score, judge = Qwen3.6-27B
# @ TEACHER_BASE) — the reward functions are EVALUATION-ONLY here; the training
# signal is pure cross-entropy. Validation generation is greedy (n=1, temp=0),
# matching the GRPO baseline's eval.
#
# Trainer on SERVER 1 (4 GPUs). Judge (same Qwen3.6-27B) on SERVER 5,
# http://point.dd.works:18184. No gen server needed.
set -xeuo pipefail

REPO=${REPO:-/root/mirl_evolve}
DATA_DIR=/scratch/sheng/self_evolving/mimiciv_rare
KEY=$(cat /scratch/sheng/self_evolving/.climb_teacher_key)
EXP="${EXP:-mimiciv_rare_qwen36_27b_sft_distill}"
TEACHER_BASE="${TEACHER_BASE:-http://point.dd.works:18184/v1}"
EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$EXP}"

export CHAT_PROVIDER=vllm
export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}"
cd "$REPO"

# Full SFT sequence length (long mimiciv prompts + short boxed answer).
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-12288}"

/usr/local/bin/python -m verl.trainer.main_sft_evolving \
    data.train_files="${DISTILL_FILE:-$DATA_DIR/distill_sft_train.jsonl}" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/static_trace_sft_dataset.py \
    data.custom_cls.name=StaticTraceSFTDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    ++data.val_max_samples="${VAL_MAX_SAMPLES:--1}" \
    data.max_length="$SFT_MAX_LENGTH" \
    data.pad_mode=right \
    data.truncation=left \
    data.shuffle=True \
    data.image_key=images \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=True \
    data.dataloader_num_workers=8 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$TEACHER_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name=Qwen/Qwen3.6-27B \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key=EMPTY \
    +reward.custom_reward_function.reward_kwargs.embed_model=Qwen/Qwen3-VL-Embedding-2B \
    +reward.custom_reward_function.reward_kwargs.evolve_enable=False \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=4096 \
    actor_rollout_ref.model.path=Qwen/Qwen3.6-27B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-False}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPT_OFFLOAD:-False}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    +actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs="${TOTAL_EPOCHS:-3}" \
    trainer.total_training_steps="${TOTAL_STEPS:-500}" \
    trainer.test_freq="${TEST_FREQ:-5}" \
    trainer.save_freq="${SAVE_FREQ:-20}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}" \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    +trainer.validation_data_dir="/scratch/sheng/self_evolving/logs_sft_distill/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "$@"
