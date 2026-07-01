#!/usr/bin/env bash
# HealthBench-Professional RUBRIC run: Qwen3.6-27B self-improve where, instead of
# a universal evolving reward over diagnosis questions, the gen server co-generates
# an open-ended clinician TASK + a HealthBench-Pro RUBRIC per item, and the TRAINING
# reward is purely that rubric, graded item-by-item by SELF (server 5).
#
#   reward = clip( sum(met points) / sum(positive points), 0, 1 )    (rubric_mode)
#
# The 8-component composite and the judge/function reward-evolution are OFF.
# Instead, at the end of each step the trainer samples 20 rollouts and POSTs them to
# the gen server's /evolve endpoint, which rewrites the (file-backed) generation
# prompts to target the model's capability gaps.
#
# VALIDATION runs in-loop at test_freq against the official HealthBench Professional
# benchmark (data.val_files), graded with the gpt-chat-latest_2026-05-28 TRAPI judge
# (the rubric scorer switches judge on _is_validation), length-adjusted to match the
# benchmark. Build the val parquet first:
#   python scripts/self_evolving/eval/preprocess_healthbench_professional.py \
#       --out /scratch/sheng/self_evolving/healthbench_pro_val.parquet
#
# Trainer on SERVER 4 (4 GPUs). Self teacher/judge (Qwen3.6-27B) on SERVER 5,
# http://point.dd.works:18184. Gen server (rubric mode) on server 4 :8006 — start
# start_gen_server_rubric_s4.sh BEFORE this.
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl_healthbench}
KEY=$(cat /scratch/sheng/self_evolving/.climb_teacher_key)
EXP="${EXP:-healthbench_rubric_qwen36_27b}"
TEACHER_BASE="${TEACHER_BASE:-http://point.dd.works:18184/v1}"
GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8006}"

# Validation judge: gpt-chat-latest via the TRAPI proxy (the standard HealthBench
# judge). Overridable; the rubric scorer only uses these on validation batches.
VAL_JUDGE_BASE="${VAL_JUDGE_BASE:-http://point.dd.works:18890/v1}"
VAL_JUDGE_MODEL="${VAL_JUDGE_MODEL:-gpt-chat-latest_2026-05-28}"
VAL_JUDGE_KEY="${VAL_JUDGE_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key 2>/dev/null || echo EMPTY)}"

# Official HealthBench Professional val parquet (built by preprocess_healthbench_professional.py).
VAL_FILES="${VAL_FILES:-/scratch/sheng/self_evolving/healthbench_pro_val.parquet}"
# train_files is a PLACEHOLDER only: SelfEvolvingDataset fetches every training
# sample from the gen server and ignores this file's content, but verl still needs
# a readable verl-shape file to construct the parent dataset. Reuse the val parquet.
TRAIN_PLACEHOLDER="${TRAIN_PLACEHOLDER:-$VAL_FILES}"

# Generation-prompt evolution (end-of-step): sample 20 rollouts -> gen server /evolve.
EVOLVE_GENERATION="${EVOLVE_GENERATION:-True}"
EVOLVE_EVERY_N="${EVOLVE_EVERY_N:-1}"
EVOLVE_N_EXAMPLES="${EVOLVE_N_EXAMPLES:-20}"

DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-/scratch/sheng/self_evolving/checkpoints/healthbench_rubric/$EXP}"

export CHAT_PROVIDER=vllm
export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export GEN_SERVER_URL="$GEN_SERVER_URL"
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}"
# Per-item rubric grading debug trace (0.0-1.0 sampling probability).
export HB_DEBUG_PRINT_PROB="${HB_DEBUG_PRINT_PROB:-0.02}"
cd "$REPO"

/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$TRAIN_PLACEHOLDER" \
    data.val_files="$VAL_FILES" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length="${MAX_RESP_LEN:-8192}" \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.shuffle=False \
    data.val_batch_size=64 \
    ++data.val_max_samples=-1 \
    data.truncation=left \
    data.return_raw_chat=True \
    data.dataloader_num_workers=8 \
    +data.self_evolving.gen_server_url="$GEN_SERVER_URL" \
    +data.self_evolving.dataset_length=100000 \
    +data.self_evolving.evolve_generation="$EVOLVE_GENERATION" \
    +data.self_evolving.evolve_every_n_steps="$EVOLVE_EVERY_N" \
    +data.self_evolving.evolve_num_examples="$EVOLVE_N_EXAMPLES" \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.rubric_mode=True \
    +reward.custom_reward_function.reward_kwargs.api_base="$TEACHER_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name=Qwen/Qwen3.6-27B \
    +reward.custom_reward_function.reward_kwargs.provider=vllm \
    +reward.custom_reward_function.reward_kwargs.val_api_base="$VAL_JUDGE_BASE" \
    +reward.custom_reward_function.reward_kwargs.val_api_key="$VAL_JUDGE_KEY" \
    +reward.custom_reward_function.reward_kwargs.val_model_name="$VAL_JUDGE_MODEL" \
    +reward.custom_reward_function.reward_kwargs.val_provider=trapi \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="$GEN_SERVER_URL" \
    reward.reward_manager.name=dapo \
    actor_rollout_ref.model.path=Qwen/Qwen3.6-27B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-2e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
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
    +trainer.validation_data_dir="/scratch/sheng/self_evolving/logs_healthbench_rubric/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "$@"
