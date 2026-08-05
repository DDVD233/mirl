#!/usr/bin/env bash
# FAST train-on-val RL probe (dvd 2026-08-04) on the preemptible 4x B200 box.
#
# Purpose: answer "can this RL stack move the benchmark at all?" in hours, not
# days, by shrinking every slow axis while keeping the machinery identical to
# the AICR diagnostics:
#   - Qwen3.5-9B instead of 27B (3x faster; TP=2 rollout)
#   - train_files == val_files == the 525 real HealthBench-Pro tasks (train-on-test)
#   - ALL LLM roles = gpt-chat-latest via TRAPI (train judge, val judge, fallback)
#   - lr 1e-6, Dr.GRPO (no std-norm), symmetric clip, no shaping terms
#   - single-turn: no gen server, no retrieval, no evolution
#   - val subsampled to 200 for speed (val_max_samples), test_freq 5
#   - VAL DECODING MATCHES TRAINING (temp 1.0, no truncation) by default. Greedy
#     val while training samples at temp 1.0 optimizes E[reward | temp 1.0] but
#     measures the argmax path — the two diverged badly before (repetition loops
#     appeared ONLY at greedy). Set VAL_DO_SAMPLE=False VAL_TEMP=0 to recover the
#     benchmark-comparable greedy protocol.
#
# 9B quirks (from run_qwen36_27b_selfimprove_from_sft.sh): head_dim=256 breaks
# the FlashAttention varlen kernel -> use_remove_padding=False + sdpa.
set -xeuo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_healthbench}
cd "$REPO"

EXP="${EXP:-hb9b_trainval_fast}"
TRAPI_BASE=http://point.dd.works:18890/v1
TRAPI_KEY=$(cat $S/.trapi_key)
JUDGE=gpt-chat-latest_2026-05-28

export CHAT_PROVIDER=trapi
export HF_HOME=$S/hf_cache
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:?export WANDB_API_KEY first}"
# wandb run-id policy: RESUME (same id, resume="allow") only when there is a
# checkpoint to actually resume from; otherwise mint a NEW id. Per the wandb
# docs, resume="allow" on an existing id "resumes from last step" — so a
# from-scratch restart under the old id gets every metric REJECTED
# ("Tried to log to step 8 that is less than the current step 16").
_CKPT_DIR="$S/checkpoints/hb9b/$EXP"
_RUNID_FILE="$S/logs_hb9b/.wandb_runid.$EXP"
if [ -f "$_CKPT_DIR/latest_checkpointed_iteration.txt" ] && [ -s "$_RUNID_FILE" ]; then
    export WANDB_RUN_ID="$(cat "$_RUNID_FILE")"          # genuine resume
else
    export WANDB_RUN_ID="${EXP}_$(date +%m%d_%H%M)"      # fresh start -> fresh curve
    mkdir -p "$S/logs_hb9b"; printf '%s' "$WANDB_RUN_ID" > "$_RUNID_FILE"
fi
export WANDB_RESUME=allow
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
# no reward shaping: raw rubric fraction, floor 0, no think/rep penalties
export HB_SCORE_MIN="${HB_SCORE_MIN:-0.0}"
export HB_THINK_PENALTY_PER_1K="${HB_THINK_PENALTY_PER_1K:-0.0}"
export HB_REP_PENALTY_MAX="${HB_REP_PENALTY_MAX:-0.0}"
export VERL_THINK_BUDGET_TOKENS="${VERL_THINK_BUDGET_TOKENS:-0}"
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-8}"

VAL=$S/healthbench_pro_val.parquet
LOGDIR=$S/logs_hb9b; mkdir -p $LOGDIR

/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="$VAL" \
    data.val_files="$VAL" \
    data.train_batch_size=32 \
    data.max_prompt_length=6144 \
    data.max_response_length="${MAX_RESP_LEN:-4096}" \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.shuffle=True \
    ++data.val_max_samples="${VAL_MAX:-200}" \
    data.truncation=left \
    data.return_raw_chat=True \
    data.dataloader_num_workers=8 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.rubric_mode=True \
    +reward.custom_reward_function.reward_kwargs.api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.provider=trapi \
    +reward.custom_reward_function.reward_kwargs.fallback_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.fallback_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.fallback_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.fallback_provider=trapi \
    +reward.custom_reward_function.reward_kwargs.val_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.val_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.val_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.val_provider=trapi \
    reward.reward_manager.name=dapo \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-9B \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN:-12288}" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    +trainer.filter_zero_variance_groups=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_UTIL:-0.45}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_LEN:-10240}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-True}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:--1}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN:-12288}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN:-12288}" \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${STEPS:-60}" \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$S/checkpoints/hb9b/$EXP" \
    +trainer.rollout_data_dir="$LOGDIR/rollouts/$EXP" \
    +trainer.validation_data_dir="$LOGDIR/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "$@"
