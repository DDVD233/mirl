#!/usr/bin/env bash
# Curated-data RL, 9B SFT initialization, frozen 9B self-judge on the same node.
set -euo pipefail
S=${S:-/scratch/sheng/self_evolving}
REPO=${REPO:-$S/verl_stage1_9b_control}
EXP=${EXP:-mimiciv_rare_qwen35_9b_trainset_selfjudge}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-$S/checkpoints/restored/qwen35_9b_sft_distill_step90/actor/huggingface}
JUDGE_BASE=${JUDGE_BASE:-http://127.0.0.1:8188/v1}
LOGDIR=$S/logs_stage1_9b_control
DATA=$LOGDIR/data_images_fixed
CKPT=$S/checkpoints/self_evolving_medical/$EXP
source "$S/venvs/stage1_9b_control/bin/activate"
test "$(git -C "$REPO" rev-parse --short=8 HEAD)" = 01c43aa6
test -f "$ACTOR_MODEL_PATH/config.json"
python "$REPO/scripts/self_evolving/train/prepare_stage1_control_data.py"
mkdir -p "$LOGDIR" "$CKPT"
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CHAT_PROVIDER=vllm CUDA_VISIBLE_DEVICES=0,1 RAY_ADDRESS=local
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600 VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export REWARD_JUDGE_CONCURRENCY=16
cd "$REPO"
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="$DATA/train.jsonl" data.val_files="$DATA/test.jsonl" \
  data.train_batch_size=64 data.val_batch_size=64 ++data.val_max_samples=-1 \
  data.max_prompt_length=8192 data.max_response_length=4096 \
  data.shuffle=False data.truncation=left data.filter_overlong_prompts=False \
  data.image_key=images data.return_raw_chat=True data.return_multi_modal_inputs=True \
  data.dataloader_num_workers=4 \
  reward.custom_reward_function.path="$REPO/verl/utils/reward_score/self_evolving.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.api_base="$JUDGE_BASE" \
  +reward.custom_reward_function.reward_kwargs.api_key=EMPTY \
  +reward.custom_reward_function.reward_kwargs.model_name=Qwen/Qwen3.5-9B \
  +reward.custom_reward_function.reward_kwargs.embed_api_base=http://mib.media.mit.edu:18001/v1 \
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
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=2e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.32 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 +actor_rollout_ref.rollout.repetition_penalty=1.1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
  critic.enable=False trainer.n_gpus_per_node=2 trainer.nnodes=1 \
  trainer.total_epochs=1000 trainer.total_training_steps="${TOTAL_STEPS:-1000}" \
  trainer.test_freq=20 trainer.save_freq=20 trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  +trainer.max_actor_ckpt_to_keep=2 trainer.resume_mode=auto \
  trainer.default_local_dir="$CKPT" \
  +trainer.validation_data_dir="$LOGDIR/val_generations/$EXP" \
  +trainer.rollout_data_dir="$LOGDIR/rollouts/$EXP" \
  trainer.project_name=self_evolving_medical trainer.experiment_name="$EXP" \
  'trainer.logger=["console","wandb"]' \
  +ray_init.address=local +ray_init.object_store_memory=21474836480 "$@"
