#!/usr/bin/env bash
# Baseline run: plain GRPO on the real mimiciv_rare TRAIN set — NO generation
# component and NO reward evolution. This is the control for
# run_qwen36_27b_evolve_reward.sh / run_qwen36_27b_selfimprove_kl.sh.
#
# Differences from the self-evolving runs (everything else is byte-identical):
#   * default RLHFDataset (no SelfEvolvingDataset / gen-server client) — the model
#     trains directly on $DATA_DIR/train.jsonl, the same real dataset family used
#     for validation ($DATA_DIR/test.jsonl).
#   * reward evolution OFF (evolve_enable=False, reward.reward_evolution.enable=False):
#     the per-sample TRAINING reward uses the SAME composite judge path that
#     validation uses, so train and val are scored identically.
#   * judge / teacher is the SAME model (Qwen3.6-27B @ $TEACHER_BASE).
#
# Train set is finite (5719 rows ⇒ ~89 steps/epoch at bs=64), so total_epochs is
# raised to 6 and the 500-step cap (total_training_steps) decides when to stop —
# matching the 500-step budget of the self-evolving runs.
#
# Trainer on SERVER 1 (4 GPUs). Teacher/judge (same Qwen3.6-27B) on SERVER 5,
# http://point.dd.works:18184. No gen server needed.
set -xeuo pipefail

REPO=${REPO:-/root/mirl_evolve}
DATA_DIR=/scratch/sheng/self_evolving/mimiciv_rare
KEY=$(cat /scratch/sheng/self_evolving/.climb_teacher_key)
EXP="${EXP:-mimiciv_rare_qwen36_27b_baseline}"
# Scaling-figure baseline knobs: start from the SFT checkpoint (MODEL_PATH) and
# match the j75o3rrt main line's use_kl_loss=False; both env-overridable.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-27B}"
USE_KL_LOSS="${USE_KL_LOSS:-True}"
# Student quirks for non-27B actors (Qwen3.5-9B: USE_REMOVE_PADDING=False
# ATTN_SDPA=1 ROLLOUT_TP=2 — head_dim=256 breaks the FA varlen kernel).
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"
EXTRA_ARGS=()
if [ -n "${ATTN_SDPA:-}" ]; then
  EXTRA_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=sdpa")
fi
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

/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.shuffle=True \
    data.val_batch_size=64 \
    ++data.val_max_samples=-1 \
    data.image_key=images \
    data.truncation=left \
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
    reward.reward_evolution.enable=False \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=4096 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.001 \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1.0 \
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002 \
    actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0 \
    actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.32 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    +actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=6 \
    trainer.total_training_steps=500 \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    +trainer.validation_data_dir="/scratch/sheng/self_evolving/logs_baseline/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "${EXTRA_ARGS[@]}" \
    "$@"
