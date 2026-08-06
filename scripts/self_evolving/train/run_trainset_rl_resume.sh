#!/usr/bin/env bash
# Resume the train-set RL control (the fair baseline for the self-evolving run).
#
# Original run: wandb 3xf0orfr / mimiciv_rare_qwen36_27b_trainset_rl_from_sft.
# It crashed at global_step 187 of a planned 500, still improving, so we cannot say
# whether it plateaus below the self-evolving curve or catches up. This resumes it.
#
# Everything is byte-identical to the original except the actor init: the local
# checkpoints were deleted in the July storage sweep, so we restart the policy from
# the archived HF copy of global_step_180. Optimizer state is not recovered - the run
# restarts its step counter at 0, and the scaling plot has to offset this leg's
# compute by what the first 187 steps already spent (2.94e8 rollout tokens).
#
# Same as the original: prompts come from the REAL curated train set (no gen server,
# no proposer), judge is the base Qwen3.6-27B served separately, reward evolution off.
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl}
DATA_DIR=/scratch/sheng/self_evolving/mimiciv_rare
KEY=sk-xMByFeWLKB87wZ
EXP="${EXP:-mimiciv_rare_qwen36_27b_trainset_rl_from_sft_resume}"
HF_CKPT_REPO="${HF_CKPT_REPO:-ddvd233/mimiciv_rare_qwen36_27b_trainset_rl_from_sft}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/scratch/sheng/self_evolving/checkpoints/restored/trainset_rl_step180}"
TEACHER_BASE="${TEACHER_BASE:-http://point.dd.works:18184/v1}"
EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
# 320 more steps takes the run to an effective 500, the original target.
TOTAL_STEPS="${TOTAL_STEPS:-320}"

export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export CHAT_PROVIDER=vllm
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

if [ ! -d "$ACTOR_MODEL_PATH" ]; then
  echo "[resume] restoring policy weights from $HF_CKPT_REPO"
  mkdir -p "$ACTOR_MODEL_PATH"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_CKPT_REPO', local_dir='$ACTOR_MODEL_PATH', local_dir_use_symlinks=False)
"
fi

cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.shuffle=False \
    data.val_batch_size=64 \
    ++data.val_max_samples=-1 \
    data.image_key=images \
    data.truncation=left \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=True \
    data.dataloader_num_workers=8 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$TEACHER_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name=Qwen/Qwen3.6-27B \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key=EMPTY \
    +reward.custom_reward_function.reward_kwargs.embed_model=Qwen/Qwen3-VL-Embedding-2B \
    +reward.custom_reward_function.reward_kwargs.evolve_enable=False \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=4096 \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.001 \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1.0 \
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002 \
    actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0 \
    actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.32 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    +actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=6 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$EXP \
    +trainer.validation_data_dir=/scratch/sheng/self_evolving/logs_baseline/val_generations/$EXP \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    trainer.logger='["console","wandb"]' \
    +ray_init.address=local
