#!/usr/bin/env bash
# Greedy, benchmark-protocol evaluation of a finished 9B run's checkpoint.
#
# The training runs validate at temperature 1.0, n=1 — deliberately, so validation
# measures the same distribution the rollouts optimize. That is the right choice for
# reading a TRAINING CURVE and the wrong one for quoting a NUMBER: HealthBench-Pro is
# a greedy benchmark, and sampled decoding carries ~0.017 sd per eval on the full 525.
# This re-scores a checkpoint under the real protocol (temp 0, do_sample=False, all
# 525 tasks) so the result is comparable to published figures.
#
#   RUN_DIR=/scratch/sheng/self_evolving/checkpoints/hb9b/hb9b_gen_control \
#   EXP=hb9b_gen_control_greedy bash scripts/self_evolving/train/eval_9b_greedy.sh
#
# No training happens: trainer.val_only=True runs one validation pass and exits.
set -xeuo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_healthbench}
cd "$REPO"

# RUN_DIR is the training run's checkpoint directory (containing global_step_N/).
# We RESUME it rather than pointing model.path at a checkpoint: verl writes the
# actor as FSDP shards (model_world_size_4_rank_*.pt) and its huggingface/ subdir
# holds only config + tokenizer, no weights — so there is nothing for model.path to
# load. resume_mode=auto picks up the latest step, and val_only runs one validation
# pass and exits without training.
# RUN_DIR=NONE evaluates the UNTRAINED base model. That baseline is not optional:
# a trained greedy score is uninterpretable without it, and greedy cannot be assumed
# to shift like sampled — the control scored 0.352 greedy vs 0.388 sampled at the
# same checkpoint, i.e. greedy was LOWER, so borrowing the sampled baseline would
# understate or overstate the gain by an unknown amount.
RUN_DIR="${RUN_DIR:?set RUN_DIR to the run checkpoint dir, or NONE for the base model}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
RESUME_ARGS=()
if [ "$RUN_DIR" = "NONE" ]; then
    echo "evaluating UNTRAINED base model: $BASE_MODEL"
    RESUME_ARGS=(trainer.resume_mode=disable
                 trainer.default_local_dir="$S/checkpoints/hb9b/_greedy_scratch")
else
    [ -f "$RUN_DIR/latest_checkpointed_iteration.txt" ] || { echo "FATAL: no checkpoint under $RUN_DIR" >&2; exit 1; }
    echo "resuming step $(cat "$RUN_DIR/latest_checkpointed_iteration.txt") from $RUN_DIR"
    RESUME_ARGS=(trainer.resume_mode=auto trainer.default_local_dir="$RUN_DIR")
fi
EXP="${EXP:-hb9b_greedy_eval}"
LOGDIR=$S/logs_hb9b; mkdir -p "$LOGDIR"
VAL=$S/healthbench_pro_val.parquet

TRAPI_BASE=http://point.dd.works:18890/v1
TRAPI_KEY=$(cat $S/.trapi_key)
JUDGE=gpt-chat-latest_2026-05-28

export CHAT_PROVIDER=trapi
export HF_HOME=$S/hf_cache
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:?export WANDB_API_KEY first}"
export WANDB_RUN_ID="${EXP}_$(date +%m%d_%H%M)"
export HB_SCORE_MIN=0.0 HB_THINK_PENALTY_PER_1K=0.0 HB_REP_PENALTY_MAX=0.0
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-10}"

/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$VAL" \
    data.val_files="$VAL" \
    data.train_batch_size=32 \
    data.max_prompt_length=6144 \
    data.max_response_length="${MAX_RESP_LEN:-8192}" \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    ++data.val_max_samples=-1 \
    data.truncation=left \
    data.return_raw_chat=True \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.rubric_mode=True \
    +reward.custom_reward_function.reward_kwargs.api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.provider=trapi \
    +reward.custom_reward_function.reward_kwargs.val_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.val_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.val_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.val_provider=trapi \
    reward.reward_manager.name=dapo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=14336 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len=14336 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=14336 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=14336 \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.total_training_steps=1 \
    +trainer.validation_data_dir="$LOGDIR/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "${RESUME_ARGS[@]}" \
    "$@"
