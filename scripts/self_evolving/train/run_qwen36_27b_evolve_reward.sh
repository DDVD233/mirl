#!/usr/bin/env bash
# Self-evolving REWARD run: same Qwen3.6-27B self-improve setup as
# run_qwen36_27b_selfimprove_kl.sh, but with the optional reward-evolution switch ON.
#
# When ON the per-sample TRAINING reward drops the composite (embedding / multi-judge /
# bleu) and uses only:  renorm( w_judge * judge_reward(evolvable prompt)
#                              + w_func  * function_reward(evolvable python) ).
# At the end of EACH training step the judge (gpu5 teacher) rewrites the judging prompt
# and writes/improves an executable function_reward; artifacts are versioned under
# $EVOLVE_DIR/step_NNN with an atomic current/ pointer. Validation always uses the old
# composite path, so val accuracy stays comparable. With the switch OFF this script
# behaves byte-identically to run_qwen36_27b_selfimprove_kl.sh.
#
# Trainer on SERVER 4 (4 GPUs). Teacher (same Qwen3.6-27B) on SERVER 5,
# http://point.dd.works:18184, used as the gen-server LLM + reward self-judge + reward evolver.
# Gen server on server 4 :8006 (start_gen_server_mimic_s1.sh BEFORE this).
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl_evolve}
DATA_DIR=/scratch/sheng/self_evolving/mimiciv_rare
KEY=$(cat /scratch/sheng/self_evolving/.climb_teacher_key)
EXP="${EXP:-mimiciv_rare_qwen36_27b_evolve_reward}"
TEACHER_BASE="${TEACHER_BASE:-http://point.dd.works:18184/v1}"
GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8006}"
EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"

# Reward-evolution switch + weights (single source of truth, fed to BOTH the trainer-side
# evolution step and the reward-side compute_score).
EVOLVE_ENABLE="${EVOLVE_ENABLE:-True}"
EVOLVE_W_JUDGE="${EVOLVE_W_JUDGE:-0.7}"
EVOLVE_W_FUNC="${EVOLVE_W_FUNC:-0.3}"
EVOLVE_EVERY_N="${EVOLVE_EVERY_N:-1}"
EVOLVE_N_EXAMPLES="${EVOLVE_N_EXAMPLES:-6}"
# Keep the artifact dir under the experiment's checkpoint dir; absolute so the trainer and
# the colocated reward workers agree on the path.
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$EXP}"
EVOLVE_DIR="${EVOLVE_DIR:-$DEFAULT_LOCAL_DIR/reward_evolution}"

export CHAT_PROVIDER=vllm
export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}"
cd "$REPO"

/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$DATA_DIR/train.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
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
    +data.self_evolving.gen_server_url="$GEN_SERVER_URL" \
    +data.self_evolving.dataset_length=100000 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$TEACHER_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name=Qwen/Qwen3.6-27B \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_API_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key=EMPTY \
    +reward.custom_reward_function.reward_kwargs.embed_model=Qwen/Qwen3-VL-Embedding-2B \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="$GEN_SERVER_URL" \
    +reward.custom_reward_function.reward_kwargs.evolve_enable="$EVOLVE_ENABLE" \
    +reward.custom_reward_function.reward_kwargs.evolve_dir="$EVOLVE_DIR" \
    +reward.custom_reward_function.reward_kwargs.evolve_w_judge="$EVOLVE_W_JUDGE" \
    +reward.custom_reward_function.reward_kwargs.evolve_w_func="$EVOLVE_W_FUNC" \
    reward.reward_evolution.enable="$EVOLVE_ENABLE" \
    reward.reward_evolution.evolve_dir="$EVOLVE_DIR" \
    reward.reward_evolution.every_n_steps="$EVOLVE_EVERY_N" \
    reward.reward_evolution.num_examples="$EVOLVE_N_EXAMPLES" \
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
    actor_rollout_ref.actor.use_kl_loss=True \
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
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    +trainer.validation_data_dir="/scratch/sheng/self_evolving/logs_evolve_reward/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "$@"
