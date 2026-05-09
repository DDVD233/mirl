#!/usr/bin/env bash
set -x

# RL (GRPO) training after SFT with reasoning traces, for all 4 splits (skip participant).
# The only difference from direct RL is model.path points to the SFT checkpoint.

ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

GEN_TP=${GEN_TP:-4}
CP=${CP:-2}
TP=${TP:-2}
PP=${PP:-1}

DATA_DIR="$HOME/scratch/raofu/3DHaptic"

for SPLIT_NAME in date glove question task; do
    echo "============================================"
    echo "Starting RL (post-SFT reasoning) for split: ${SPLIT_NAME}"
    echo "============================================"

    SFT_MODEL_PATH="./outputs/qwen3vl_sft_split_${SPLIT_NAME}_official"

    if [ ! -d "$SFT_MODEL_PATH" ]; then
        echo "WARNING: SFT model not found at ${SFT_MODEL_PATH}, skipping split '${SPLIT_NAME}'."
        continue
    fi

    train_path="${DATA_DIR}/annotation_verl_split_${SPLIT_NAME}_train.json"
    test_path="${DATA_DIR}/annotation_verl_split_${SPLIT_NAME}_test_mini.json"

    python3 -m verl.trainer.main_ppo --config-path=config \
        --config-name='ppo_megatron_trainer.yaml'\
        algorithm.adv_estimator=grpo \
        data.train_files="$train_path" \
        data.val_files="$test_path" \
        data.train_batch_size=64 \
        data.val_batch_size=32 \
        data.max_prompt_length=8192 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        actor_rollout_ref.model.path="$SFT_MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
        actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.megatron.use_mbridge=True \
        actor_rollout_ref.actor.megatron.param_offload=True \
        actor_rollout_ref.actor.megatron.optimizer_offload=True \
        actor_rollout_ref.actor.megatron.grad_offload=True \
        actor_rollout_ref.ref.megatron.param_offload=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
        +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=2048 \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='tactile' \
        trainer.experiment_name="qwen3vl_dapo_reasoning_split_${SPLIT_NAME}_official" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=5 \
        trainer.val_before_train=True \
        trainer.total_epochs=15 ${@:2}

    echo "============================================"
    echo "RL training for split '${SPLIT_NAME}' done."
    echo "============================================"
done

echo ""
echo "All splits done!"
