#!/usr/bin/env bash
# GRPO RL stage: Qwen3-VL-8B (or SFT checkpoint) on ChildPlay ADOS item scoring.
# FSDP2 actor + vLLM rollout; reward = childplay_ados (exact score match + format).
# Env knobs: NUM_GPUS, DATA_DIR, MODEL_PATH, TRAIN_FILE, VAL_FILE, TRAIN_BS, GRPO_N,
#            ROLLOUT_TP, GPU_UTIL, RL_EXP, TOTAL_EPOCHS, LOGGER, VAL_BEFORE
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-4}
DATA_DIR=${DATA_DIR:-/scratch/dvdai/childplay_dataset}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
PROJECT_NAME=${PROJECT_NAME:-childplay_ados}
RL_EXP=${RL_EXP:-qwen3vl8b_grpo}
LOGGER=${LOGGER:-'["console","wandb"]'}
TRAIN_BS=${TRAIN_BS:-32}
RUN_DIR=${RUN_DIR:-/scratch/dvdai/childplay_ados}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-10240}
MAX_RESP_LEN=${MAX_RESP_LEN:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/${TRAIN_FILE:-childplay_ados_train.jsonl}" \
    data.val_files="${DATA_DIR}/${VAL_FILE:-childplay_ados_val_mini.jsonl}" \
    data.train_batch_size="${TRAIN_BS}" \
    data.val_batch_size=${VAL_BS:-128} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=${FSDP_STRATEGY:-fsdp2} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD:-False} \
    actor_rollout_ref.actor.optim.lr=${RL_LR:-1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BS}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OFFLOAD:-False} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.6} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${WEIGHT_BUCKET_MB:-4096} \
    actor_rollout_ref.rollout.n=${GRPO_N:-8} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=256 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESP_LEN} \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${RL_EXP}" \
    trainer.default_local_dir="${RUN_DIR}/checkpoints/${PROJECT_NAME}/${RL_EXP}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.val_before_train=${VAL_BEFORE:-True} \
    trainer.total_epochs=${TOTAL_EPOCHS:-3} "$@"
