#!/usr/bin/env bash
# Evaluate a HuggingFace MERGED model (ddvd233/mimiciv_rare_<run>) on the FULL
# MIMIC-IV rare test set (2452 cases) with the EXACT verl validation pipeline
# (main_ppo val_only), greedy n=1, scored by self_evolving.compute_score:
# headline `acc` == lenient LLM-judge disease match (gpt-5.3), plus exact_acc,
# judge_acc_strict, embed_sim, etc. The val data_source is the ICD-10 category,
# so verl logs per-category metrics (val-core/mimic_rare/<cat>/<metric>/mean@1).
# No training, no FSDP resume — model.path IS the full merged model.
#
# Usage:
#   MODEL=ddvd233/mimiciv_rare_qwen36_27b_selfimprove EXP=hfeval_qwen36_27b_selfimprove \
#   TP=4 bash run_val_only_hf.sh
set -xeuo pipefail

REPO=/scratch/sheng/self_evolving/verl
DATA_DIR=/scratch/sheng/self_evolving/mimiciv_rare
KEY=$(cat /scratch/sheng/self_evolving/.trapi_key)

MODEL="${MODEL:?set MODEL=ddvd233/<repo>}"
EXP="${EXP:?set EXP=<name>}"
TP="${TP:-4}"
GPUS="${GPUS:-4}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.3-chat_2026-03-03}"
JUDGE_BASE="${JUDGE_BASE:-http://point.dd.works:18890/v1}"
EMBED_BASE="${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}"
VAL_DUMP="${VAL_DUMP:-/scratch/sheng/self_evolving/eval_hf/dump/$EXP}"

# trapi judge => reward uses max_completion_tokens + reasoning_effort:none (gpt-5.x).
export CHAT_PROVIDER=trapi
# The TRAPI proxy enforces a GLOBAL ~2000 req/60s cap; cap in-flight judge calls so
# the full-batch judge burst doesn't trip it (403 -> judge returns 0 -> acc reads 0).
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-16}"
# Eval reports the lenient-judge acc only; skip strict/answer-quality/reasoning
# judge calls (4x fewer calls -> stays under the rate limit + ~4x faster).
export REWARD_EVAL_LENIENT_ONLY="${REWARD_EVAL_LENIENT_ONLY:-1}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export RAY_ADDRESS=local
mkdir -p "$VAL_DUMP"
cd "$REPO"

PYTHONPATH="$REPO" /usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/test.jsonl" \
    data.val_files="$DATA_DIR/test.jsonl" \
    data.train_batch_size=64 \
    data.max_prompt_length=12288 \
    data.max_response_length=16384 \
    data.shuffle=False \
    data.val_batch_size=256 \
    data.image_key=images \
    data.truncation=left \
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
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=16384 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    critic.enable=False \
    trainer.n_gpus_per_node="$GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    +trainer.validation_data_dir="$VAL_DUMP" \
    trainer.project_name=self_evolving_hfeval \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console"]' \
    "$@"
