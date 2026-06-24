#!/usr/bin/env bash
# Evaluate ONE verl checkpoint with the EXACT training-validation pipeline:
# verl main_ppo in val_only mode, loading the checkpoint via resume_from_path,
# generating with the same rollout val_kwargs (greedy n=1) and scoring with the
# same self_evolving.compute_score reward + judge. The val data_source is the
# ICD-10 category, so verl logs per-category metrics (val-core/mimic_rare/<cat>/
# acc/mean@1) and dumps per-sample generations to VAL_DUMP for aggregation.
#
# This reproduces wandb val numbers (unlike the hand-built eval_sota requests).
#
# Usage:
#   CKPT=.../global_step_240 BASE=google/gemma-4-31B-it EXP=valonly_gemma31b_step240 \
#   TP=4 bash run_val_only.sh
set -xeuo pipefail

REPO=/scratch/sheng/self_evolving/verl
DATA_DIR=/scratch/sheng/self_evolving/mimiciv_rare
KEY=$(cat /scratch/sheng/self_evolving/.trapi_key)

CKPT="${CKPT:?set CKPT=.../global_step_N}"
BASE="${BASE:?set BASE=<hf model>}"
EXP="${EXP:?set EXP=<name>}"
TP="${TP:-4}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.3-chat_2026-03-03}"
JUDGE_BASE="${JUDGE_BASE:-http://point.dd.works:18890/v1}"
EMBED_BASE="${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}"
VAL_DUMP="${VAL_DUMP:-/scratch/sheng/self_evolving/eval_valonly/dump/$EXP}"
GPUS="${GPUS:-4}"

export CHAT_PROVIDER=trapi
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
mkdir -p "$VAL_DUMP"
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"

PYTHONPATH="$REPO" "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/test.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.train_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.shuffle=False \
    data.val_batch_size=200 \
    data.image_key=images \
    data.truncation=right \
    data.filter_overlong_prompts=False \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.api_base="$JUDGE_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$JUDGE_MODEL" \
    +reward.custom_reward_function.reward_kwargs.embed_api_base="$EMBED_BASE" \
    +reward.custom_reward_function.reward_kwargs.embed_api_key=EMPTY \
    +reward.custom_reward_function.reward_kwargs.embed_model=Qwen/Qwen3-VL-Embedding-2B \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward.reward_kwargs.overlong_buffer_cfg.len=256 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=2048 \
    actor_rollout_ref.model.path="$BASE" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    critic.enable=False \
    trainer.n_gpus_per_node="$GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CKPT" \
    +trainer.validation_data_dir="$VAL_DUMP" \
    trainer.project_name=self_evolving_valonly \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console"]' \
    "$@"
