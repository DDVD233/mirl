#!/usr/bin/env bash
# Qwen3.6-27B student + Qwen3.5-397B-A17B-FP8 teacher via external vLLM URL.
# Same reward function as the DAPO/self-evolving runs; adds on-policy
# distillation loss (k1, policy-gradient form, matches the Thinking Machines
# recipe). The teacher lives on a separate machine — no extra GPUs in the
# student Ray cluster.
#
# Teacher must be a vLLM server started with `--max-logprobs 64` (or more)
# and reachable at $TEACHER_URL.

set -xeuo pipefail

export API_BASE="${API_BASE:-http://node2500:8002/v1}"
export API_KEY="${API_KEY:-EMPTY}"
export CHAT_PROVIDER="${CHAT_PROVIDER:-vllm}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export DATA_DIR="${DATA_DIR:-/home/dvdai/scratch/dvdai/self_evolving_datasets/mimiciv_rare}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_qwen36_27b_opd_external}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export TEACHER_URL="${TEACHER_URL:-http://node2500:8005/v1}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
export DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-k1}"
export DISTILL_TOPK="${DISTILL_TOPK:-1}"
export DISTILL_USE_PG="${DISTILL_USE_PG:-True}"
export DISTILL_USE_TASK_REWARDS="${DISTILL_USE_TASK_REWARDS:-True}"
export DISTILL_COEF="${DISTILL_COEF:-1.0}"

if [ -d /home/dvdai/miniconda3/envs/cu130/lib ]; then
    export LD_LIBRARY_PATH="/home/dvdai/miniconda3/envs/cu130/lib:${LD_LIBRARY_PATH:-}"
fi

export RAY_ADDRESS="local"

PYTHON_BIN="${PYTHON_BIN:-/home/dvdai/miniconda3/envs/cu130/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/dvdai/verl}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-/home/dvdai/scratch/dvdai/self_evolving_datasets/logs/val_generations/$EXPERIMENT_NAME}"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.shuffle=True \
    data.val_batch_size=64 \
    data.image_key=images \
    data.truncation=left \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$API_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$API_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$MODEL_NAME" \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key="$EMBED_API_KEY" \
    +reward.custom_reward_function.reward_kwargs.embed_model="$EMBED_MODEL" \
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
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=False \
    distillation.enabled=True \
    distillation.n_gpus_per_node=0 \
    distillation.nnodes=0 \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.external_url="$TEACHER_URL" \
    distillation.teacher_models.teacher_model.external_api_key="$TEACHER_API_KEY" \
    distillation.teacher_models.teacher_model.inference.temperature=1.0 \
    distillation.distillation_loss.loss_mode="$DISTILL_LOSS_MODE" \
    distillation.distillation_loss.topk="$DISTILL_TOPK" \
    distillation.distillation_loss.use_policy_gradient="$DISTILL_USE_PG" \
    distillation.distillation_loss.use_task_rewards="$DISTILL_USE_TASK_REWARDS" \
    distillation.distillation_loss.distillation_loss_coef="$DISTILL_COEF" \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    +trainer.max_actor_ckpt_to_keep=2 \
    +trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    'trainer.logger=["console","wandb"]' \
    "+ray_init.address=local" \
    "$@"
