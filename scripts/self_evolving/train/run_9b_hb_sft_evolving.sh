#!/usr/bin/env bash
# 9B SFT-via-RL-pipeline probe (dvd 2026-08-04), server 3 (4x B200, port 2336).
#
# QUESTION: without any RL, does supervised training on GPT-written GOLD answers
# for self-generated (task, rubric) pairs improve HealthBench-Pro?
#
# Pipeline (no SFT trainer; the RL stack with an SFT loss):
#   gen server --rubric_mode --sft_mode  ->  for each generated task it now also
#     writes a GOLD answer conditioned on the rubric (attach_rubric_gold_trace),
#     self-grades it against that rubric, and keeps it only if it scores
#     >= HB_GOLD_MIN_SCORE. The kept trace is <think>...</think> + final answer.
#   verl.trainer.main_sft_evolving  ->  supervised CE on those traces, while
#     VALIDATION runs the normal RL validation path on the real 525-task
#     HealthBench-Pro parquet with the gpt-chat-latest rubric judge.
# Every LLM role (task gen, rubric gen, gold answer, self-grade, val judge) is
# gpt-chat-latest via TRAPI. No local teacher, no Qwen judge.
set -xeuo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_healthbench}
cd "$REPO"

EXP="${EXP:-hb9b_sft_evolving}"
TRAPI_BASE=http://point.dd.works:18890/v1
TRAPI_KEY=$(cat $S/.trapi_key)
JUDGE=gpt-chat-latest_2026-05-28
GEN_PORT="${GEN_PORT:-8021}"
VAL=$S/healthbench_pro_val.parquet
LOGDIR=$S/logs_hb9b; mkdir -p "$LOGDIR"

export CHAT_PROVIDER=trapi
export HF_HOME=$S/hf_cache
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:?export WANDB_API_KEY first}"
# wandb run-id policy: RESUME (same id, resume="allow") only when there is a
# checkpoint to actually resume from; otherwise mint a NEW id. Per the wandb
# docs, resume="allow" on an existing id "resumes from last step" — so a
# from-scratch restart under the old id gets every metric REJECTED
# ("Tried to log to step 8 that is less than the current step 16").
_CKPT_DIR="$S/checkpoints/hb9b/$EXP"
_RUNID_FILE="$S/logs_hb9b/.wandb_runid.$EXP"
if [ -f "$_CKPT_DIR/latest_checkpointed_iteration.txt" ] && [ -s "$_RUNID_FILE" ]; then
    export WANDB_RUN_ID="$(cat "$_RUNID_FILE")"          # genuine resume
else
    export WANDB_RUN_ID="${EXP}_$(date +%m%d_%H%M)"      # fresh start -> fresh curve
    mkdir -p "$S/logs_hb9b"; printf '%s' "$WANDB_RUN_ID" > "$_RUNID_FILE"
fi
export WANDB_RESUME=allow
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600 VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export HB_GOLD_MIN_SCORE="${HB_GOLD_MIN_SCORE:-0.8}"   # gold must earn 80% of its own rubric
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-8}"

# --- 1. gen server: rubric mode + SFT mode (gold answers), all-GPT ------------
if curl -sf "localhost:$GEN_PORT/healthz" >/dev/null 2>&1; then
    echo "FATAL: port $GEN_PORT already serving (orphan?)" >&2; exit 1
fi
PROMPT_DIR="$LOGDIR/$EXP/prompts"
if [ ! -d "$PROMPT_DIR" ]; then
    mkdir -p "$LOGDIR/$EXP"
    cp -r "$S/logs_healthbench_rubric/v17_prompt_seed/prompts" "$PROMPT_DIR" 2>/dev/null || \
    cp -r "$S/logs_healthbench_rubric/v16_prompt_seed/prompts" "$PROMPT_DIR"
fi
python scripts/self_evolving/generation_server.py \
    --rubric_mode --sft_mode \
    --prompt_dir "$PROMPT_DIR" \
    --api_base "$TRAPI_BASE" --api_key "$TRAPI_KEY" --model_name "$JUDGE" \
    --embed_api_base http://mib.media.mit.edu:18001/v1 \
    --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --milvus_uri http://mib.media.mit.edu:19531 --milvus_token root:Milvus \
    --milvus_collection medical_knowledge_v2 --milvus_top_k 8 \
    --n_queries 6 --questions_per_query 1 --workers 8 \
    --teacher_retries 2 --teacher_max_tokens 4096 \
    --log_dir "$LOGDIR/$EXP" --host 0.0.0.0 --port "$GEN_PORT" \
    > "$LOGDIR/gen_server_${EXP}.log" 2>&1 &
GEN_PID=$!
start=$SECONDS
until curl -sf "localhost:$GEN_PORT/healthz" >/dev/null; do
    kill -0 $GEN_PID 2>/dev/null || { echo "FATAL: gen server died"; tail -20 "$LOGDIR/gen_server_${EXP}.log"; exit 1; }
    (( SECONDS - start > 900 )) && { echo "FATAL: gen server unhealthy"; exit 1; }
    sleep 5
done
echo "gen server (rubric+sft) healthy on :$GEN_PORT"

# --- 2. SFT-loss trainer, RL-path validation on HealthBench-Pro ---------------
python -m verl.trainer.main_sft_evolving \
    data.train_files="$VAL" \
    data.val_files="$VAL" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_sft_dataset.py \
    data.custom_cls.name=SelfEvolvingSFTDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.val_batch_size=64 \
    data.max_prompt_length=6144 \
    data.max_response_length=4096 \
    ++data.val_max_samples="${VAL_MAX:-200}" \
    data.max_length="${SFT_MAX_LENGTH:-10240}" \
    data.pad_mode=right \
    data.truncation=left \
    data.shuffle=False \
    +data.self_evolving.gen_server_url="http://localhost:$GEN_PORT" \
    +data.self_evolving.dataset_length=100000 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.rubric_mode=True \
    +reward.custom_reward_function.reward_kwargs.api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.provider=trapi \
    +reward.custom_reward_function.reward_kwargs.val_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.val_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.val_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.val_provider=trapi \
    +reward.custom_reward_function.reward_kwargs.fallback_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.fallback_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.fallback_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.fallback_provider=trapi \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="http://localhost:$GEN_PORT" \
    reward.reward_manager.name=dapo \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH:-Qwen/Qwen3.5-9B}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-True}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP:-1.0}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${STEPS:-100}" \
    trainer.test_freq="${TEST_FREQ:-5}" \
    trainer.val_before_train=True \
    trainer.save_freq=25 \
    +trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$S/checkpoints/hb9b/$EXP" \
    +trainer.validation_data_dir="$LOGDIR/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "$@"
rc=$?
kill $GEN_PID 2>/dev/null
exit $rc
