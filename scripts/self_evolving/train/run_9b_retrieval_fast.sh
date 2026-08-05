#!/usr/bin/env bash
# 9B RETRIEVAL RL on the preemptible 4x B200 box. Wraps run_9b_trainval_fast.sh so
# the base recipe stays the single source of truth for everything non-retrieval.
#
# WHAT IS DIFFERENT FROM EVERY PRIOR RETRIEVAL RUN
# ------------------------------------------------
# 1. ONE CONTINUOUS TRAJECTORY. The old loop generated the graded answer from a
#    rebuilt tool-free prompt and reset response_mask, which DELETED the query
#    tokens from the trained sequence — retrieval quality had no gradient at all.
#    Now the query turns, the (masked) evidence, and the answer are one sequence,
#    so the policy learns WHAT TO RETRIEVE as well as how to answer.
# 2. QUERY PLANS, not one query. `search_medical_kb` takes 1-4 sub-queries covering
#    different facets. Measured on 93 rubric criteria whose supporting fact IS in the
#    KB: one query supplies it 10.8% of the time, two supply 35.9% (McNemar p<1e-4).
# 3. SUMMARIZED EVIDENCE. /retrieve merges the per-query results and a frozen
#    summarizer compresses them into a question-conditioned brief (~2.5k chars from
#    ~10k), which is what makes retrieving MORE affordable inside the response window.
# 4. A SECOND GRADED COMPONENT. `retrieval_coverage` scores whether the retrieved
#    evidence supplies what the rubric grades, independent of whether the answer used
#    it, and is folded GROUP-RELATIVELY in the trainer so it ranks query quality
#    without biasing the decision to search at all.
#
# METRIC COMPARABILITY: val `acc` keeps its exact definition (the retrieval term
# touches `score` only, never `length_adjusted`), so these numbers stay comparable to
# the NO-RETRIEVAL baseline. They are NOT comparable to the v7-v11 retrieval runs,
# because the graded text moves from the rebuilt answer to the in-trajectory answer.
#
#   WANDB_API_KEY=... bash scripts/self_evolving/train/run_9b_retrieval_fast.sh
#
# Prerequisites (checked below, fatal if missing): the embed server + Milvus that
# /retrieve needs, and — unless SUMM_BASE is set to something already running — a
# summarizer (scripts/self_evolving/serve/serve_summarizer.sh).
set -xeuo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_healthbench}
cd "$REPO"

export EXP="${EXP:-hb9b_retrieval_fast}"
LOGDIR=$S/logs_hb9b; mkdir -p "$LOGDIR"
GEN_PORT="${GEN_PORT:-8031}"
SUMM_PORT="${SUMM_PORT:-8199}"
SUMM_BASE="${SUMM_BASE:-http://localhost:$SUMM_PORT/v1}"
SUMM_MODEL="${SUMM_MODEL:-Qwen/Qwen3.5-9B}"
EMBED_BASE="${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}"
MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"

TRAPI_BASE=http://point.dd.works:18890/v1
TRAPI_KEY=$(cat $S/.trapi_key)
JUDGE=gpt-chat-latest_2026-05-28

# ---------------------------------------------------------------- token budget
# The response now carries the masked evidence too:
#   2 x search turn (think 1024 + ~200 tool call)  ~= 2450
# + 2 x evidence brief (<=2500 chars ~ 760 tok + scaffolding)  ~= 1600
# + close instruction                                          ~=  120
# + answer turn (think 3072 + >=1536 answer)                   >= 4600
#                                                              ~= 8800
export MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
export ROLLOUT_MAX_LEN="${ROLLOUT_MAX_LEN:-14336}"   # = 6144 prompt + 8192 response
# HARD RULE: ppo_max_token_len_per_gpu >= max_prompt + max_response. The
# rearrange_micro_batches assert fires on the longest ACTUAL sequence (the batch is
# nested by then), so the previous 12288 against a 14336 ceiling was a crash waiting
# for the first long rollout — which a continuous trajectory produces routinely.
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-14336}"
export LOGPROB_MAX_TOKEN_LEN="${LOGPROB_MAX_TOKEN_LEN:-14336}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.40}"        # v10 lesson: activations vs vLLM reservation

# ------------------------------------------------------------ agent-loop knobs
export RETRIEVAL_URL="http://localhost:$GEN_PORT/retrieve"
export VERL_MAX_SEARCHES="${VERL_MAX_SEARCHES:-2}"       # optional 0-2; a 3rd call errors
export VERL_SEARCH_THINK_BUDGET="${VERL_SEARCH_THINK_BUDGET:-1024}"
export VERL_THINK_BUDGET_TOKENS="${VERL_THINK_BUDGET_TOKENS:-3072}"
export VERL_ANSWER_RESERVE_TOKENS="${VERL_ANSWER_RESERVE_TOKENS:-3500}"
export VERL_MIN_ANSWER_TOKENS="${VERL_MIN_ANSWER_TOKENS:-1536}"

# --------------------------------------------------------------- reward knobs
export HB_RETRIEVAL_WEIGHT="${HB_RETRIEVAL_WEIGHT:-0.20}"
export HB_RETRIEVAL_GROUP_BASELINE="${HB_RETRIEVAL_GROUP_BASELINE:-1}"
# The group-relative bonus is +/-HB_RETRIEVAL_WEIGHT and is clipped into
# [HB_SCORE_MIN, 1]. The base recipe floors at 0.0, where a NEGATIVE delta on a
# low-scoring rollout (common: raw rubric fractions here are often near 0) would be
# swallowed by the floor while a positive one survives — a one-sided bonus that
# pushes the policy to search regardless of query quality. Give the floor at least
# `w` of headroom; watch reward/retrieval_bonus/clipped_frac to confirm it is small.
export HB_SCORE_MIN="${HB_SCORE_MIN:--0.25}"
# Only used for groups with <2 searching rollouts. Set near the observed
# reward/retrieval_coverage/mean once the first few steps have logged it.
export HB_RETRIEVAL_NOSEARCH_COVERAGE="${HB_RETRIEVAL_NOSEARCH_COVERAGE:-0.35}"
# +1 concurrency slot for the coverage judge (shares the rubric judge's semaphore).
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-10}"
# think_chars now counts the search turns' reasoning too, so the think penalty would
# become a per-search tax. It is already 0 in the base recipe; keep it that way.
export HB_THINK_PENALTY_PER_1K=0.0

# ------------------------------------------------------- services, gated, ordered
cleanup() { kill ${GEN_PID:-} ${SUMM_PID:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

curl -sf -m 10 "$EMBED_BASE/models" >/dev/null \
    || { echo "FATAL: embed server unreachable at $EMBED_BASE" >&2; exit 1; }

# 1) summarizer (skip if something is already serving it)
if ! curl -sf -m 5 "$SUMM_BASE/models" >/dev/null 2>&1; then
    # Pin to ONE device explicitly. Left to inherit, vLLM would grab GPU 0 and its
    # allocation would land on top of the trainer's rank 0; SUMM_GPU makes the
    # placement deterministic and keeps the fraction accounting checkable.
    CUDA_VISIBLE_DEVICES="${SUMM_GPU:-3}" \
    MODEL="$SUMM_MODEL" PORT="$SUMM_PORT" TP="${SUMM_TP:-1}" MEM="${SUMM_MEM:-0.08}" \
        bash scripts/self_evolving/serve/serve_summarizer.sh \
        > "$LOGDIR/summarizer_${EXP}.log" 2>&1 &
    SUMM_PID=$!
    start=$SECONDS
    until curl -sf -m 5 "$SUMM_BASE/models" >/dev/null; do
        kill -0 "$SUMM_PID" 2>/dev/null \
            || { echo "FATAL: summarizer died during startup" >&2; tail -40 "$LOGDIR/summarizer_${EXP}.log"; exit 1; }
        (( SECONDS - start > 1800 )) && { echo "FATAL: summarizer not healthy after 1800s" >&2; exit 1; }
        sleep 5
    done
fi
echo "summarizer healthy at $SUMM_BASE"

# 2) gen server (hosts /retrieve). Orphan pre-flight: an old server on this port
# passes the health gate and the run would silently retrieve from the wrong config.
if curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null 2>&1; then
    echo "FATAL: port $GEN_PORT already serving (orphan?)" >&2; exit 1
fi
/usr/local/bin/python scripts/self_evolving/generation_server.py \
    --rubric_mode \
    --api_base "$TRAPI_BASE" --api_key "$TRAPI_KEY" --model_name "$JUDGE" \
    --embed_api_base "$EMBED_BASE" --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --milvus_uri "$MILVUS_URI" --milvus_token root:Milvus \
    --milvus_collection medical_knowledge_v2 \
    --retrieve_top_k "${RETRIEVE_TOP_K:-5}" --retrieve_total "${RETRIEVE_TOTAL:-16}" \
    --summarizer_api_base "$SUMM_BASE" --summarizer_model "$SUMM_MODEL" \
    --summarizer_provider vllm \
    --workers 4 --log_dir "$LOGDIR/$EXP" --host 0.0.0.0 --port "$GEN_PORT" \
    > "$LOGDIR/gen_server_${EXP}.log" 2>&1 &
GEN_PID=$!
start=$SECONDS
until curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null; do
    kill -0 "$GEN_PID" 2>/dev/null \
        || { echo "FATAL: gen server died during startup" >&2; tail -40 "$LOGDIR/gen_server_${EXP}.log"; exit 1; }
    (( SECONDS - start > 900 )) && { echo "FATAL: gen server not healthy after 900s" >&2; exit 1; }
    sleep 5
done
echo "gen server healthy on :$GEN_PORT"

# 3) prove retrieval actually works BEFORE spending a training step on it. A silent
# fallback to raw passages (summarizer down) would change the whole run's context
# distribution and look like a modelling result.
curl -sf -m 180 -X POST "localhost:$GEN_PORT/retrieve" -H 'content-type: application/json' \
    -d '{"question":"75yo, eGFR 38, on metformin - safe to continue?",
         "queries":["metformin contraindication eGFR threshold",
                    "metformin lactic acidosis risk renal impairment"]}' \
  | /usr/local/bin/python -c '
import json, sys
d = json.load(sys.stdin)
assert d["n_merged"] >= 4, d
assert d["summarized"] is True, f"summarizer NOT active: {d.get(\"fallback_reason\")}"
print(f"retrieve smoke OK: {d[\"n_queries\"]}q -> {d[\"n_merged\"]} passages -> {d[\"chars\"]} chars")
' || { echo "FATAL: /retrieve smoke failed" >&2; exit 1; }

# ------------------------------------------------------------------- trainer
bash scripts/self_evolving/train/run_9b_trainval_fast.sh \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESP:-6000}" \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=scripts/self_evolving/train/config/medical_retrieval_tool.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=retrieval_tool_agent \
    +reward.custom_reward_function.reward_kwargs.cov_api_base="$TRAPI_BASE" \
    +reward.custom_reward_function.reward_kwargs.cov_api_key="$TRAPI_KEY" \
    +reward.custom_reward_function.reward_kwargs.cov_model_name="$JUDGE" \
    +reward.custom_reward_function.reward_kwargs.cov_provider=trapi \
    "$@"
rc=$?
kill "$GEN_PID" "${SUMM_PID:-}" 2>/dev/null || true
exit $rc
