#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================
# CONFIGURATION - Edit these variables
# ============================================
PROJECT_NAME="rl_omni_heldout"
EXP_NAME="trial"      # used as wandb_name

CKPT_FOLDER="/scratch/keane/human_behaviour/trial"
CKPT_STEP="0"
SPLIT="test"                                  # val | test
ADV_ESTIMATOR="tarpo"
N_GPUS=2

TRAIN_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_train_cleaned_2.jsonl"
VAL_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_val_cleaned.jsonl"
TEST_FILE="/scratch/keane/human_behaviour/human_behaviour_data/final_v8_test_cleaned.jsonl"

VAL_BATCH_SIZE=16
TEST_BATCH_SIZE=64

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPU_MEMORY_UTILIZATION=0.6
DATALOADER_NUM_WORKERS=4
# ============================================

# Derived paths
CHECKPOINT_DIR="$CKPT_FOLDER/$CKPT_STEP"
VALIDATION_DIR="$CKPT_FOLDER/${CKPT_STEP}_${SPLIT}"
EXPERIMENT_NAME="${EXP_NAME}_${CKPT_STEP}_${SPLIT}"

if [ "$SPLIT" = "val" ]; then
  DATA_FILE="$VAL_FILE"
  BATCH_SIZE="$VAL_BATCH_SIZE"
elif [ "$SPLIT" = "test" ]; then
  DATA_FILE="$TEST_FILE"
  BATCH_SIZE="$TEST_BATCH_SIZE"
else
  echo "Error: unknown split '$SPLIT' (expected 'val' or 'test')" >&2
  exit 1
fi

# if [ ! -d "$CKPT_FOLDER" ]; then
#   echo "Error: checkpoint folder '$CKPT_FOLDER' does not exist"
#   exit 1
# fi

# if [ ! -d "$CHECKPOINT_DIR" ]; then
#   echo "Error: checkpoint directory '$CHECKPOINT_DIR' does not exist"
#   exit 1
# fi

export CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/keaneong/human-behavior/verl:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1

echo "=================================================="
echo "Checkpoint:       $CHECKPOINT_DIR"
echo "Split:            $SPLIT"
echo "Adv estimator:    $ADV_ESTIMATOR"
echo "Data file:        $DATA_FILE"
echo "Validation dir:   $VALIDATION_DIR"
echo "Experiment name:  $EXPERIMENT_NAME"
echo "N GPUs:           $N_GPUS"
echo "=================================================="

ray stop --force 2>/dev/null || true
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$DATA_FILE" \
  data.train_batch_size=256 \
  data.val_batch_size="$BATCH_SIZE" \
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
  actor_rollout_ref.model.path=keentomato/harpo_hier_step400 \
  actor_rollout_ref.model.trust_remote_code=True \
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
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=25 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.validation_data_dir="$VALIDATION_DIR" \
  trainer.test_freq=99999 \
  trainer.total_epochs=5 \
  trainer.advantage_save_dir="$CKPT_FOLDER/advantages" \
  trainer.advantage_plot_freq=15 \
  trainer.default_local_dir="$CKPT_FOLDER"


  # trainer.resume_mode=resume_path \
  # trainer.resume_from_path="$CHECKPOINT_DIR" \