set -x
ENGINE=${1:-vllm}
[ $# -gt 0 ] && shift

# Strip leaked conda-build / spack compiler env so nvcc doesn't pick up a
# foreign sysroot at JIT time (sgl-kernel JIT-compiles e.g.
# resolve_future_token_ids when n>1, even with enforce_eager=True).
unset NVCC_PREPEND_FLAGS CC CXX CUDAHOSTCXX CUDACXX \
      CPPFLAGS CFLAGS CXXFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS \
      LDFLAGS CONDA_BUILD_SYSROOT

export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_P2P_DISABLE=1

ROLLOUT_TP=${ROLLOUT_TP:-2}
SP_SIZE=${SP_SIZE:-1}

train_path="$HOME/scratch/raofu/3DHaptic/annotation_verl_split_date_train.json"
test_path="$HOME/scratch/raofu/3DHaptic/annotation_verl_split_date_test_mini.json"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.model.path=google/gemma-3-4b-it \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$SP_SIZE \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=$SP_SIZE \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma3DecoderLayer] \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=2048 \
    trainer.critic_warmup=0 \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='tactile' \
    trainer.experiment_name='gemma3_4b_dapo_split_date_fsdp' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.total_epochs=15 $@
