#!/usr/bin/env bash
set -x

# Pin to GPUs 0,1
export CUDA_VISIBLE_DEVICES=0,1
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
    #  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8 \

# max response_length doesn't really do anything I beleive, its in fact summed by model length minus prompt length.
# model length should be greater than that of the prompt length
# NOTE: seems like the prompt length may be set as conservative while resuming 

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=tarpo \
        data.train_files=/scratch/keane/human_behaviour/human_behaviour_data/final_v8_train_cleaned.jsonl \
        data.val_files=/scratch/keane/human_behaviour/human_behaviour_data/final_v8_val_cleaned.jsonl \
        data.train_batch_size=128 \
        data.val_batch_size=64 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=False \
        data.truncation='right' \
        data.image_key=images \
        data.video_key=videos \
        data.prompt_key=problem \
        data.dataloader_num_workers=4 \
        data.modalities=\'audio,videos\' \
        data.train_modality_batching.enabled=True \
        data.train_modality_batching.drop_last=True \
        data.val_modality_batching.enabled=True \
        data.val_modality_batching.drop_last=False \
        data.format_prompt=/home/keaneong/human-behavior/verl/examples/format_prompt/default.jinja \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-Omni-7B \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.max_model_len=10240 \
        actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
        actor_rollout_ref.rollout.max_num_seqs=2 \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=/home/keaneong/human-behavior/verl/examples/reward_function/human_behaviour_tarpo.py \
        custom_reward_function.name=human_behaviour_compute_score_batch \
        reward_model.reward_manager=batch \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='rl_omni_heldout' \
        trainer.experiment_name='tarpo_full' \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.val_before_train=False \
        trainer.val_only=False \
        trainer.validation_data_dir=/scratch/keane/human_behaviour/tarpo_full \
        trainer.test_freq=100 \
        trainer.total_epochs=10 $@ \
        trainer.default_local_dir=/scratch/keane/human_behaviour/tarpo_full