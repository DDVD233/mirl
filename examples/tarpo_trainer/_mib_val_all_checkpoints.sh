#!/usr/bin/env bash
set -x

# ============================================
# CONFIGURATION - Edit these variables
# ============================================
CHECKPOINT_FOLDER="/scratch/keane/hb_atlas_models/gpg_mib_baseline"
ALGO_NAME="gpg"
MIN_CHECKPOINT_STEP=0  # Only evaluate checkpoints >= this step number (0 = evaluate all)
# ============================================

if [ ! -d "$CHECKPOINT_FOLDER" ]; then
    echo "Error: Checkpoint folder '$CHECKPOINT_FOLDER' does not exist"
    echo "Please edit the CHECKPOINT_FOLDER variable at the top of this script"
    exit 1
fi

# Pin to GPUs 0,1
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/keaneong/human-behavior/verl:$PYTHONPATH"
export NCCL_ASYNC_ERROR_HANDLING=1

# train modality batching = do one modality at a time;

# --train_file "/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_exclude_heldout_train.jsonl" \
# --val_file "/scratch/keane/human_behaviour/human_behaviour_data/w_feats_v6_exclude_heldout_val.jsonl" \

# the printed out total steps will essentially be the all the steps within the full number of epochs (i.e. 1/1210, where epoch is 5), 1210 is num of steps for 5 epochs
# take 1210/5 = 242 as steps per epoch
# hence we should eval every 242 but save every 121

# list of edits done:
# free_cache_engine:changed from true to false for speed at the expense of memory (failed)
# actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \ set to 2 from 1 
# actor_rollout_ref.model.use_remove_padding=True \ set to true from false 
# max prompt length and max response length set to 2048 from 4096
# dataloader num workers set to 8
# gpu memory set to 0.7 from 0.6

# originally prompt length, response length, max model len is 2048, 2048, 8192

# Find all checkpoint directories matching global_step_*
for checkpoint_dir in "$CHECKPOINT_FOLDER"/global_step_*; do
    # Check if directory exists (handles case where no matches found)
    if [ ! -d "$checkpoint_dir" ]; then
        continue
    fi

    # Extract checkpoint name (e.g., "global_step_100")
    checkpoint_name=$(basename "$checkpoint_dir")

    # Extract step number from checkpoint name
    step_number=$(echo "$checkpoint_name" | sed 's/global_step_//')

    # Skip if step number is below threshold
    if [ -n "$step_number" ] && [ "$step_number" -lt "$MIN_CHECKPOINT_STEP" ]; then
        echo "Skipping $checkpoint_name (step $step_number < $MIN_CHECKPOINT_STEP)"
        continue
    fi

    # Create validation data directory name (e.g., "global_step_100_val")
    validation_dir="$checkpoint_dir/${checkpoint_name}_val"

    # Create experiment name (e.g., "gpg_global_step_100")
    experiment_name="${ALGO_NAME}_${checkpoint_name}"

    echo "=================================================="
    echo "Processing checkpoint: $checkpoint_name"
    echo "Resume from: $checkpoint_dir"
    echo "Validation dir: $validation_dir"
    echo "Experiment name: $experiment_name"
    echo "=================================================="

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=tarpo \
        data.train_files=/scratch/keane/human_behaviour_data/final_v8_train_cleaned_2.jsonl \
        data.val_files=/scratch/keane/human_behaviour_data/final_v8_val_cleaned.jsonl \
        data.train_batch_size=256 \
        data.val_batch_size=64 \
        data.max_prompt_length=4096 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=False \
        data.truncation='right' \
        data.image_key=images \
        data.video_key=videos \
        data.prompt_key=problem \
        data.dataloader_num_workers=8 \
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
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.max_model_len=6192 \
        actor_rollout_ref.rollout.max_num_batched_tokens=6192 \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=/home/keaneong/human-behavior/verl/examples/reward_function/human_behaviour_tarpo.py \
        custom_reward_function.name=human_behaviour_compute_score_batch \
        reward_model.reward_manager=batch \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='rl_baselines' \
        trainer.experiment_name="$experiment_name" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.val_before_train=True \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$checkpoint_dir" \
        trainer.val_only=True \
        trainer.validation_data_dir="$validation_dir" \
        trainer.test_freq=1 \
        trainer.total_epochs=1 $@ \
        trainer.default_local_dir="$CHECKPOINT_FOLDER"

done

echo "=================================================="
echo "All checkpoints processed!"
echo "=================================================="