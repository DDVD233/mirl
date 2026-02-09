#!/usr/bin/env bash
set -x

# K-Fold Cross-Validation Training Script
# Usage: ./_training_kfold.sh [fold_name]
# Example: ./_training_kfold.sh adam
# Or run all folds: ./_training_kfold.sh all

FOLD=${1:-adam}  # Default to adam if no argument

# Subject boundaries
declare -A FOLDS=(
    ["adam"]="fold_adam"
    ["amanda"]="fold_amanda"
    ["davida"]="fold_davida"
    ["deborah"]="fold_deborah"
    ["luke"]="fold_luke"
)

# Data directory
DATA_DIR="/home/jadali85/orcd/scratch/autism/ados_eval/global_text"

# Pin to GPUs 0,1
export CUDA_VISIBLE_DEVICES=0,1
unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/jadali85/orcd/scratch/updated2/mirl:$PYTHONPATH"
export NCCL_ASYNC_ERROR_HANDLING=1
module load cuda/12.4.0

run_fold() {
    local fold_name=$1
    local fold_prefix=${FOLDS[$fold_name]}
    
    echo "========================================="
    echo "Running K-Fold CV: Holding out $fold_name"
    echo "Train file: ${DATA_DIR}/${fold_prefix}_train.jsonl"
    echo "Val file: ${DATA_DIR}/${fold_prefix}_val.jsonl"
    echo "========================================="
    
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=${DATA_DIR}/${fold_prefix}_train.jsonl \
        data.val_files=${DATA_DIR}/${fold_prefix}_val.jsonl \
        data.train_batch_size=128 \
        data.val_batch_size=64 \
        data.max_prompt_length=8000 \
        actor_rollout_ref.rollout.prompt_length=8000 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=False \
        data.truncation='right' \
        data.image_key=images \
        data.audio_key=audios \
        data.video_key=videos \
        data.prompt_key=problem \
        data.dataloader_num_workers=8 \
        data.modalities=\'text\' \
        data.train_modality_batching.enabled=False \
        data.train_modality_batching.drop_last=True \
        data.seed=2 \
        data.val_modality_batching.enabled=False \
        data.val_modality_batching.drop_last=False \
        data.format_prompt=/home/jadali85/orcd/scratch/updated2/mirl/examples/format_prompt/default.jinja \
        actor_rollout_ref.model.path=Qwen/Qwen3-8B \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.max_model_len=6144 \
        actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=/home/jadali85/orcd/scratch/updated2/mirl/examples/reward_function/multilabel_fixed.py \
        custom_reward_function.name=human_behaviour_compute_score_batch \
        reward_model.reward_manager=batch \
        reward_model.launch_reward_fn_async=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="jada_CP_VLmodel_kfold" \
        trainer.experiment_name="fold_${fold_name}" \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=5 \
        trainer.val_before_train=True \
        trainer.val_only=False \
        trainer.validation_data_dir=/home/jadali85/orcd/scratch/autism/ados_eval/models/ \
        trainer.test_freq=1 \
        trainer.total_epochs=3 \
        trainer.default_local_dir=/home/jadali85/orcd/scratch/autism/ados_eval/models/fold_${fold_name} \
        hydra.job.name="fold_${fold_name}" \
        hydra.run.dir="/home/jadali85/orcd/scratch/autism/ados_eval/global_text/results_GtM_Qwen3text_${fold_name}" \
        ${@:2}
    
    echo "Completed fold: $fold_name"
    echo ""
}

# Run based on argument
if [ "$FOLD" == "all" ]; then
    echo "Running all 5 folds sequentially..."
    for fold_name in adam amanda davida deborah luke; do
        run_fold $fold_name
    done
    
    echo "========================================="
    echo "All folds completed!"
    echo "========================================="
    echo "Results saved in:"
    echo "  /home/jadali85/orcd/scratch/autism/ados_eval/models/fold_*"
    echo ""
    echo "To compute average metrics across folds, run:"
    echo "  python compute_kfold_metrics.py"
else
    # Run single fold
    if [[ -v FOLDS[$FOLD] ]]; then
        run_fold $FOLD
    else
        echo "Error: Invalid fold name '$FOLD'"
        echo "Valid options: adam, amanda, davida, deborah, luke, all"
        exit 1
    fi
fi
