#!/usr/bin/env bash
# 9B HealthBench RL on GENERATED tasks, with retrieval as a single on/off switch.
#
# WHY THIS IS ONE SCRIPT AND NOT TWO. The first retrieval result was 70% artifact
# because the retrieval arm also changed the response budget. Both arms now come from
# this file and differ ONLY in RETRIEVAL=1|0; every budget, batch size, judge and
# decoding parameter is shared by construction, so nothing can silently diverge.
#
# WHY GENERATED TASKS. The earlier runs trained on the validation set itself
# (train_files == val_files), so their absolute numbers were memorisation, not
# generalisation, and no benchmark claim could be made. Here the gen server
# co-generates HealthBench-style tasks + rubrics (SelfEvolvingDataset pulls them at
# every step) and validation stays the REAL, untouched healthbench_pro_val.parquet.
# Val is therefore genuinely held out.
#
#   RETRIEVAL=1 bash scripts/self_evolving/train/run_9b_hb_gen.sh   # retrieval arm
#   RETRIEVAL=0 bash scripts/self_evolving/train/run_9b_hb_gen.sh   # control arm
#
# Generation-prompt evolution is OFF (EVOLVE_GENERATION=False): this run measures the
# retrieval effect on generated data, and the v13-v16 diagnosis showed the evolver
# collapsing task diversity, which would be a second moving part.
#
# VALIDATION USES ALL 525 TASKS (val_max_samples=-1). The fast-iteration script
# subsampled to 200 for speed, which was fine when val was a progress gauge trained
# on anyway — it is not fine now: this is the held-out benchmark number, and
# subsampling both makes it non-comparable and widens the per-eval noise band by
# sqrt(525/200) ~ 1.6x, which is most of the +/-0.05 uncertainty that kept forcing
# caveats onto the earlier deltas. Full val costs ~2.6x more per eval; that is the
# right trade for the number the whole experiment exists to produce.
#
# 9B quirks: head_dim=256 breaks the FlashAttention varlen kernel -> use_remove_padding
# =False + sdpa (same as run_9b_trainval_fast.sh).
set -xeuo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_healthbench}
cd "$REPO"

RETRIEVAL="${RETRIEVAL:?set RETRIEVAL=1 (retrieval arm) or RETRIEVAL=0 (control arm)}"
EXP="${EXP:-hb9b_gen_$([ "$RETRIEVAL" = 1 ] && echo retrieval || echo control)}"
LOGDIR=$S/logs_hb9b; mkdir -p "$LOGDIR"

TRAPI_BASE=http://point.dd.works:18890/v1
TRAPI_KEY=$(cat $S/.trapi_key)
JUDGE=gpt-chat-latest_2026-05-28
GEN_PORT="${GEN_PORT:-8041}"
SUMM_PORT="${SUMM_PORT:-8199}"
SUMM_BASE="${SUMM_BASE:-http://localhost:$SUMM_PORT/v1}"
SUMM_MODEL="${SUMM_MODEL:-Qwen/Qwen3.5-9B}"
EMBED_BASE="${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}"
MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
VAL=$S/healthbench_pro_val.parquet

export CHAT_PROVIDER=trapi
export HF_HOME=$S/hf_cache
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:?export WANDB_API_KEY first}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

# wandb id policy: resume only when a checkpoint actually exists, else a fresh id.
# resume=allow on an existing id resumes FROM ITS LAST STEP, so a from-scratch
# restart under the old id has every metric silently rejected.
_CKPT_DIR="$S/checkpoints/hb9b/$EXP"
_RUNID_FILE="$LOGDIR/.wandb_runid.$EXP"
if [ -f "$_CKPT_DIR/latest_checkpointed_iteration.txt" ] && [ -s "$_RUNID_FILE" ]; then
    export WANDB_RUN_ID="$(cat "$_RUNID_FILE")"
else
    export WANDB_RUN_ID="${EXP}_$(date +%m%d_%H%M)"
    printf '%s' "$WANDB_RUN_ID" > "$_RUNID_FILE"
fi
export WANDB_RESUME=allow

# ---- shaping: none, so the rubric fraction is the signal (matches prior runs) ----
export HB_THINK_PENALTY_PER_1K=0.0
export HB_REP_PENALTY_MAX=0.0
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-10}"

# ---- token budget: IDENTICAL in both arms (this is the confound that bit us) ----
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
ROLLOUT_MAX_LEN="${ROLLOUT_MAX_LEN:-14336}"      # 6144 prompt + 8192 response
# HARD RULE: >= max_prompt + max_response, else rearrange_micro_batches asserts on
# the first long rollout (the assert is on the longest ACTUAL sequence).
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-14336}"
LOGPROB_MAX_TOKEN_LEN="${LOGPROB_MAX_TOKEN_LEN:-14336}"

cleanup() { kill ${GEN_PID:-} ${SUMM_PID:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

curl -sf -m 10 "$EMBED_BASE/models" >/dev/null \
    || { echo "FATAL: embed server unreachable at $EMBED_BASE" >&2; exit 1; }

# ---- retrieval-only wiring -------------------------------------------------------
AGENT_ARGS=()
if [ "$RETRIEVAL" = 1 ]; then
    export HB_SCORE_MIN="${HB_SCORE_MIN:--0.25}"   # headroom so the +/-w bonus is not
                                                   # clipped one-sidedly at the floor
    export HB_RETRIEVAL_WEIGHT="${HB_RETRIEVAL_WEIGHT:-0.20}"
    export HB_RETRIEVAL_GROUP_BASELINE=1
    export HB_RETRIEVAL_NOSEARCH_COVERAGE="${HB_RETRIEVAL_NOSEARCH_COVERAGE:-0.35}"
    export RETRIEVAL_URL="http://localhost:$GEN_PORT/retrieve"
    export VERL_MAX_SEARCHES="${VERL_MAX_SEARCHES:-2}"
    export VERL_SEARCH_THINK_BUDGET="${VERL_SEARCH_THINK_BUDGET:-1024}"
    export VERL_THINK_BUDGET_TOKENS="${VERL_THINK_BUDGET_TOKENS:-3072}"
    export VERL_ANSWER_RESERVE_TOKENS="${VERL_ANSWER_RESERVE_TOKENS:-3500}"
    export VERL_MIN_ANSWER_TOKENS="${VERL_MIN_ANSWER_TOKENS:-1536}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.35}"         # summarizer shares a GPU
    AGENT_ARGS=(
        actor_rollout_ref.rollout.multi_turn.enable=True
        actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESP:-6000}"
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
        actor_rollout_ref.rollout.multi_turn.tool_config_path=scripts/self_evolving/train/config/medical_retrieval_tool.yaml
        actor_rollout_ref.rollout.agent.default_agent_loop=retrieval_tool_agent
        +reward.custom_reward_function.reward_kwargs.cov_api_base="$TRAPI_BASE"
        +reward.custom_reward_function.reward_kwargs.cov_api_key="$TRAPI_KEY"
        +reward.custom_reward_function.reward_kwargs.cov_model_name="$JUDGE"
        +reward.custom_reward_function.reward_kwargs.cov_provider=trapi
    )
    # Summarizer (frozen self-model), pinned to one GPU.
    if ! curl -sf -m 5 "$SUMM_BASE/models" >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="${SUMM_GPU:-3}" MODEL="$SUMM_MODEL" PORT="$SUMM_PORT" \
        TP=1 MEM="${SUMM_MEM:-0.14}" MAXSEQS=64 \
            bash scripts/self_evolving/serve/serve_summarizer.sh \
            > "$LOGDIR/summarizer_${EXP}.log" 2>&1 &
        SUMM_PID=$!
        start=$SECONDS
        until curl -sf -m 5 "$SUMM_BASE/models" >/dev/null; do
            kill -0 "$SUMM_PID" 2>/dev/null || { echo "FATAL: summarizer died" >&2; tail -40 "$LOGDIR/summarizer_${EXP}.log"; exit 1; }
            (( SECONDS - start > 1800 )) && { echo "FATAL: summarizer unhealthy" >&2; exit 1; }
            sleep 5
        done
    fi
    echo "summarizer healthy at $SUMM_BASE"
else
    export HB_SCORE_MIN="${HB_SCORE_MIN:-0.0}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"
fi

# ---- gen server: co-generates the TRAINING tasks + rubrics (both arms) ------------
if curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null 2>&1; then
    echo "FATAL: port $GEN_PORT already serving (orphan?)" >&2; exit 1
fi
PROMPT_DIR="$LOGDIR/$EXP/prompts"; mkdir -p "$PROMPT_DIR"
SUMM_FLAGS=()
[ "$RETRIEVAL" = 1 ] && SUMM_FLAGS=(--summarizer_api_base "$SUMM_BASE"
                                    --summarizer_model "$SUMM_MODEL"
                                    --summarizer_provider vllm)
/usr/local/bin/python scripts/self_evolving/generation_server.py \
    --rubric_mode --prompt_dir "$PROMPT_DIR" \
    --api_base "$TRAPI_BASE" --api_key "$TRAPI_KEY" --model_name "$JUDGE" \
    --embed_api_base "$EMBED_BASE" --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --milvus_uri "$MILVUS_URI" --milvus_token root:Milvus \
    --milvus_collection medical_knowledge_v2 --milvus_top_k 8 \
    --retrieve_top_k "${RETRIEVE_TOP_K:-5}" --retrieve_total "${RETRIEVE_TOTAL:-16}" \
    "${SUMM_FLAGS[@]}" \
    --n_queries "${N_QUERIES:-6}" --questions_per_query 1 \
    --accuracy_window 64 --max_pool_size 200 \
    --workers "${GEN_WORKERS:-8}" --log_dir "$LOGDIR/$EXP" \
    --host 0.0.0.0 --port "$GEN_PORT" \
    > "$LOGDIR/gen_server_${EXP}.log" 2>&1 &
GEN_PID=$!
start=$SECONDS
until curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null; do
    kill -0 "$GEN_PID" 2>/dev/null || { echo "FATAL: gen server died" >&2; tail -40 "$LOGDIR/gen_server_${EXP}.log"; exit 1; }
    (( SECONDS - start > 900 )) && { echo "FATAL: gen server unhealthy" >&2; exit 1; }
    sleep 5
done
echo "gen server healthy on :$GEN_PORT"

# Prove retrieval works BEFORE spending a step; a silent fallback to raw passages
# would change the whole run's context distribution and look like a modelling result.
if [ "$RETRIEVAL" = 1 ]; then
    curl -sf -m 180 -X POST "localhost:$GEN_PORT/retrieve" -H 'content-type: application/json' \
        -d '{"question":"75yo, eGFR 38, on metformin - safe to continue?",
             "queries":["metformin contraindication eGFR threshold",
                        "metformin lactic acidosis risk renal impairment"]}' \
      | /usr/local/bin/python -c '
import json, sys
d = json.load(sys.stdin)
assert d["n_merged"] >= 4, d
assert d["summarized"] is True, "summarizer NOT active: %s" % (d.get("fallback_reason"),)
print("retrieve smoke OK: %sq -> %s passages -> %s chars"
      % (d["n_queries"], d["n_merged"], d["chars"]))
' || { echo "FATAL: /retrieve smoke failed" >&2; exit 1; }
fi

# ---- trainer ---------------------------------------------------------------------
# train_files is a PLACEHOLDER: SelfEvolvingDataset fetches every training sample
# from the gen server. val_files is the REAL benchmark set and is never trained on.
/usr/local/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="$VAL" \
    data.custom_cls.path=scripts/self_evolving/self_evolving_dataset.py \
    data.custom_cls.name=SelfEvolvingDataset \
    +data.self_evolving.gen_server_url="http://localhost:$GEN_PORT" \
    +data.self_evolving.evolve_generation=False \
    data.val_files="$VAL" \
    data.train_batch_size="${TRAIN_BS:-32}" \
    data.max_prompt_length=6144 \
    data.max_response_length="$MAX_RESP_LEN" \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.shuffle=True \
    ++data.val_max_samples="${VAL_MAX:--1}" \
    data.truncation=left \
    data.return_raw_chat=True \
    data.dataloader_num_workers=8 \
    reward.custom_reward_function.path=verl/utils/reward_score/self_evolving.py \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.rubric_mode=True \
    +reward.custom_reward_function.reward_kwargs.api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.provider=trapi \
    +reward.custom_reward_function.reward_kwargs.fallback_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.fallback_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.fallback_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.fallback_provider=trapi \
    +reward.custom_reward_function.reward_kwargs.val_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.val_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.val_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.val_provider=trapi \
    +reward.custom_reward_function.reward_kwargs.gen_server_url="http://localhost:$GEN_PORT" \
    reward.reward_manager.name=dapo \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH:-Qwen/Qwen3.5-9B}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN" \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    +trainer.filter_zero_variance_groups=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-True}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMP:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.use_trtllm_attention=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
    critic.enable=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${STEPS:-60}" \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.val_before_train=True \
    +trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$_CKPT_DIR" \
    +trainer.rollout_data_dir="$LOGDIR/rollouts/$EXP" \
    +trainer.validation_data_dir="$LOGDIR/val_generations/$EXP" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=["console","wandb"]' \
    +ray_init.address=local \
    "${AGENT_ARGS[@]}" \
    "$@"
rc=$?
kill "$GEN_PID" "${SUMM_PID:-}" 2>/dev/null || true
exit $rc
