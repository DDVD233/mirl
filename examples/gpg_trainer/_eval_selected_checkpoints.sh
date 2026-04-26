#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================
# CONFIGURATION - Edit these variables
# ============================================
PROJECT_NAME="rl_omni_heldout"

# Exact checkpoint/split sequence to evaluate when no CLI args are passed.
# Format: "<checkpoint_folder>|<global_step_N|N>|<val|test>|<adv_estimator>[|wandb_name]"
# Examples:
#   "/scratch/keane/human_behaviour/22apr_gpg|global_step_350|val|gpg"
#   "/scratch/keane/human_behaviour/dapo_enhanced_harpo|400|test|tarpo"
# If wandb_name is omitted, the checkpoint folder basename is used.
EVAL_PLAN=(
  "/scratch/keane/human_behaviour/22apr_gpg|global_step_400|test|gpg"
  "/scratch/keane/human_behaviour/v2_emagrpo_engaging_baseline|global_step_300|val|emagrpo"
  "/scratch/keane/human_behaviour/v2_emagrpo_engaging_baseline|global_step_350|val|emagrpo"
  "/scratch/keane/human_behaviour/v2_emagrpo_engaging_baseline|global_step_400|val|emagrpo"
  "/scratch/keane/human_behaviour/v2_emagrpo_engaging_baseline|global_step_450|val|emagrpo"
  "/scratch/keane/human_behaviour/v2_emagrpo_engaging_baseline|global_step_500|test|emagrpo"
)

TRAIN_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_train_cleaned_2.jsonl"
VAL_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_val_cleaned.jsonl"
TEST_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_test_cleaned.jsonl"

VAL_BATCH_SIZE=64
TEST_BATCH_SIZE=64

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS_PER_NODE=2
GPU_MEMORY_UTILIZATION=0.6
DATALOADER_NUM_WORKERS=4
# ============================================

if [ "$#" -gt 0 ]; then
  EVAL_PLAN=("$@")
fi

export CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/keaneong/human-behavior/verl:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1

split_file() {
  case "$1" in
    val) echo "$VAL_FILE" ;;
    test) echo "$TEST_FILE" ;;
    *)
      echo "Error: unknown split '$1' (expected 'val' or 'test')" >&2
      return 1
      ;;
  esac
}

split_batch_size() {
  case "$1" in
    val) echo "$VAL_BATCH_SIZE" ;;
    test) echo "$TEST_BATCH_SIZE" ;;
    *)
      echo "Error: unknown split '$1' (expected 'val' or 'test')" >&2
      return 1
      ;;
  esac
}

normalize_step() {
  local raw_step="$1"
  raw_step="${raw_step#global_step_}"

  if [[ ! "$raw_step" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid checkpoint step '$1' (expected global_step_N or N)" >&2
    return 1
  fi

  echo "$raw_step"
}

parse_eval_entry() {
  local entry="$1"
  local extra=""

  IFS='|' read -r checkpoint_folder raw_step split adv_estimator wandb_name extra <<< "$entry"

  if [ -n "$extra" ] || [ -z "$checkpoint_folder" ] || [ -z "$raw_step" ] || [ -z "$split" ] || [ -z "$adv_estimator" ]; then
    echo "Error: invalid eval entry '$entry'" >&2
    echo "Expected: <checkpoint_folder>|<global_step_N|N>|<val|test>|<adv_estimator>[|wandb_name]" >&2
    return 1
  fi

  if [ -z "$wandb_name" ]; then
    wandb_name="$(basename "$checkpoint_folder")"
  fi
}

if [ "${#EVAL_PLAN[@]}" -eq 0 ]; then
  echo "Error: EVAL_PLAN is empty and no CLI checkpoint entries were provided"
  exit 1
fi

for entry in "${EVAL_PLAN[@]}"; do
  parse_eval_entry "$entry"

  step_number="$(normalize_step "$raw_step")"
  checkpoint_name="global_step_${step_number}"
  checkpoint_dir="$checkpoint_folder/$checkpoint_name"
  val_file="$(split_file "$split")"
  val_batch_size="$(split_batch_size "$split")"
  validation_dir="$checkpoint_folder/${checkpoint_name}_${split}"
  experiment_name="${wandb_name}_${checkpoint_name}_${split}"

  if [ ! -d "$checkpoint_folder" ]; then
    echo "Error: checkpoint folder '$checkpoint_folder' does not exist"
    exit 1
  fi

  if [ ! -d "$checkpoint_dir" ]; then
    echo "Error: checkpoint directory '$checkpoint_dir' does not exist"
    exit 1
  fi

  echo "=================================================="
  echo "Processing checkpoint: $checkpoint_name"
  echo "Step: $step_number"
  echo "Split: $split"
  echo "Advantage estimator: $adv_estimator"
  echo "Checkpoint folder: $checkpoint_folder"
  echo "Data file: $val_file"
  echo "Resume from: $checkpoint_dir"
  echo "Validation dir: $validation_dir"
  echo "Experiment name: $experiment_name"
  echo "=================================================="

  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$adv_estimator" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$val_file" \
    data.train_batch_size=256 \
    data.val_batch_size="$val_batch_size" \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.image_key=images \
    data.video_key=videos \
    data.prompt_key=problem \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    data.modalities=\'audio,videos\' \
    data.train_modality_batching.enabled=True \
    data.train_modality_batching.drop_last=True \
    data.val_modality_batching.enabled=True \
    data.val_modality_batching.drop_last=False \
    data.format_prompt=/home/keaneong/human-behavior/verl/examples/format_prompt/default.jinja \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Omni-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_model_len=6192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=3096 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=/home/keaneong/human-behavior/verl/examples/reward_function/human_behaviour_tarpo.py \
    custom_reward_function.name=human_behaviour_compute_score_batch \
    reward_model.reward_manager=batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.val_before_train=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$checkpoint_dir" \
    trainer.val_only=True \
    trainer.validation_data_dir="$validation_dir" \
    trainer.test_freq=99999 \
    trainer.total_epochs=5 \
    trainer.advantage_save_dir="$checkpoint_folder/advantages" \
    trainer.advantage_plot_freq=15 \
    trainer.default_local_dir="$checkpoint_folder"
done
