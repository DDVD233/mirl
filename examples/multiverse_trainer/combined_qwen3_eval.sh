export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_TMPDIR=/scratch/alecz/ray_tmp
set -x

# Evaluate a trained checkpoint on the combined val set.
#
# Usage:
#   bash combined_qwen3_eval.sh <experiment_name>
#       Auto-finds the latest checkpoint in checkpoints/multiverse/<experiment_name>/
#
#   bash combined_qwen3_eval.sh <experiment_name> <model_path>
#       Uses the given model path directly (e.g. a HuggingFace model ID)
#
# How eval works:
#   1. Sets trainer.val_only=True — skips training, runs one validation pass
#   2. Loads the model, generates responses for the val set using vLLM rollout
#   3. Scores each response via default_compute_score (dispatches by data_source)
#   4. Logs metrics to wandb and saves outputs to outputs/eval_<experiment_name>/
#   5. The reward dict keys (score, acc, f1, etc.) are all logged as eval metrics

EXPERIMENT_NAME=${1:?Usage: $0 <experiment_name> [model_path]}
PROJECT_NAME='multiverse'
CKPT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

if [ -n "$2" ]; then
    MODEL_PATH="$2"
else
    LATEST_STEP=$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")
    MODEL_PATH="${CKPT_DIR}/global_step_${LATEST_STEP}/actor/huggingface"
fi

echo "Evaluating model at: ${MODEL_PATH}"

VAL_FILE=/home/alecz/scratch/alecz/multiverse-test/mirl_tactile/data/combined_val.json
OUTPUT_DIR="outputs/eval_${EXPERIMENT_NAME}"

mkdir -p "$OUTPUT_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$VAL_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.image_key=images \
    data.video_key=videos \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=4096 \
    +ray_init.num_cpus=16 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="eval_${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_only=True \
    trainer.validation_data_dir="$OUTPUT_DIR" \
    trainer.total_epochs=15 $@
