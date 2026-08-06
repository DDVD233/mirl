#!/usr/bin/env bash
# 27B HealthBench RETRIEVAL run on AICR (dvd, 2026-08-05).
#
# Scales up what was validated at 9B on the MSR pods: a single CONTINUOUS
# trajectory in which the retrieval QUERY and the answer are both trained, query
# plans of 1-4 sub-queries, a summarized evidence brief, and a second graded
# component (retrieval_coverage) folded group-relatively in the trainer.
#
# The 9B evidence this rests on (train-on-val pair, 60 steps each, 13 matched evals):
# retrieval 0.227 -> 0.515 vs same-budget control 0.204 -> 0.407; mean delta +0.095,
# positive at 13/13 steps, and the gap WIDENS with training (+0.061 early ->
# +0.114 late), i.e. RL learning to USE evidence rather than merely being handed it.
#
# ALL LLM ROLES ARE gpt-chat-latest VIA TRAPI: train judge, val judge, fallback,
# the retrieval-coverage judge, AND the /retrieve summarizer. The summarizer is
# NOT the frozen self-model here (as it is on the MSR 9B runs) because a 27B
# summarizer would need ~54 GB and these nodes have 4 GPUs already carrying a 27B
# actor + rollout engine. Consequence to keep in mind: evidence briefs are written
# by a frontier model, so this measures "does 27B benefit from good retrieved
# evidence", not "can 27B summarize for itself".
#
# Training tasks are co-generated (SelfEvolvingDataset); validation is the real,
# untouched HealthBench-Pro set, all 525 tasks. Val is genuinely held out.
# Generation-prompt evolution OFF (single moving part).
#
# Launch through the chain so the allocation survives a workload crash:
#   cd /work/mit/ppliang_mit/dvdai/mirl
#   echo -e "$PWD\nsbatch --job-name=hb-retrieval --cpus-per-task=48 --mem=800G scripts/self_evolving/aicr/train_chain.sbatch" \
#       > /scratch/dvdai_mit/self_evolving/chain/SUBMITLINE.hb-retrieval
#   sbatch --job-name=hb-retrieval --cpus-per-task=48 --mem=800G scripts/self_evolving/aicr/train_chain.sbatch
set -uo pipefail

WORK=/work/mit/ppliang_mit/dvdai
S=/scratch/dvdai_mit/self_evolving
SIF=$S/sif/verl-selfevolving-cu130-vllm0.22.1.sif
MSR=/scratch/sheng/self_evolving          # container-side view of $S

export EXP="${EXP:-hb27b_retrieval}"
# dvd's OWN key, and it must be passed to the trainer COMMAND, not just exported:
# the dvdai_mit account is shared, and ~/.bashrc:42 sources ~/.wandb_env which
# exports a DIFFERENT (collaborator's) WANDB_API_KEY. The trainer runs under
# `bash -lc`, a LOGIN shell, so that profile is sourced AFTER this export and
# would silently win — the run then lands in the wrong wandb account.
WANDB_KEY_="$(cat "$S/.wandb_key")"
export WANDB_API_KEY="$WANDB_KEY_"
export WANDB_RESUME=allow
# Resume the same wandb run ONLY when a checkpoint exists; otherwise a fresh id.
# resume=allow on an existing id resumes FROM ITS LAST STEP, so a from-scratch
# restart under the old id has every metric silently rejected.
_state="$S/chain/WANDB_RUNID.$EXP"
_ckpt="$S/checkpoints/healthbench_rubric/$EXP"
if [[ -f "$_ckpt/latest_checkpointed_iteration.txt" && -s "$_state" ]]; then
    export WANDB_RUN_ID="$(cat "$_state")"
else
    export WANDB_RUN_ID="${EXP}_$(date +%m%d_%H%M)"
    mkdir -p "$S/chain"; printf '%s' "$WANDB_RUN_ID" > "$_state"
fi

BINDS=(--bind "$S:$MSR" --bind "$WORK/mirl:$MSR/verl_healthbench")
RUN_GPU=(apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF")
RUN_CPU=(apptainer exec --writable-tmpfs "${BINDS[@]}" "$SIF")

TRAPI_BASE="http://point.dd.works:18890/v1"
TRAPI_KEY_="$(cat "$S/.trapi_key")"
JUDGE="gpt-chat-latest_2026-05-28"
GEN_PORT="${GEN_PORT:-8051}"

probe_trapi() {
    curl -s -m 20 -H "Authorization: Bearer $TRAPI_KEY_" -H "Content-Type: application/json" \
        -d "{\"model\":\"$JUDGE\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_completion_tokens\":8}" \
        "$TRAPI_BASE/chat/completions" | grep -q '"choices"'
}
tstart=$SECONDS
until probe_trapi; do
    (( SECONDS - tstart > 3600 )) && { echo "FATAL: TRAPI $JUDGE unreachable for 1h" >&2; exit 1; }
    sleep 30
done
echo "TRAPI $JUDGE healthy"

# The retrieval stack lives on mib and is reachable directly from AICR compute
# (verified 2026-08-05: embed :18001, Milvus :19531, TRAPI :18890 all OK), so no
# tunnel is needed beyond the chain's own.
EMBED_BASE="${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}"
MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
curl -sf -m 15 "$EMBED_BASE/models" >/dev/null \
    || { echo "FATAL: embed server unreachable at $EMBED_BASE" >&2; exit 1; }

INIT_HOST=$S/checkpoints/self_evolving_medical/gpt56_sft_qwen36_27b/global_step_28/hf_merged
[[ -f "$INIT_HOST/config.json" ]] || { echo "FATAL: 27B SFT init missing at $INIT_HOST" >&2; exit 1; }
INIT=$MSR/checkpoints/self_evolving_medical/gpt56_sft_qwen36_27b/global_step_28/hf_merged

# ---- judges: gpt-chat-latest on every side --------------------------------------
export CHAT_PROVIDER=trapi
export TRAIN_JUDGE_BASE="$TRAPI_BASE" TRAIN_JUDGE_MODEL="$JUDGE"
export TRAIN_JUDGE_PROVIDER=trapi     TRAIN_JUDGE_KEY="$TRAPI_KEY_"
export VAL_JUDGE_BASE="$TRAPI_BASE"   VAL_JUDGE_MODEL="$JUDGE"
export VAL_JUDGE_PROVIDER=trapi       VAL_JUDGE_KEY="$TRAPI_KEY_"
export FALLBACK_JUDGE_BASE="$TRAPI_BASE" FALLBACK_JUDGE_MODEL="$JUDGE"
export FALLBACK_JUDGE_PROVIDER=trapi     FALLBACK_JUDGE_KEY="$TRAPI_KEY_"
export EVOLVE_GENERATION=False
export ENTROPY_COEFF=0.0
export HB_REP_PENALTY_MAX=0
export HB_THINK_PENALTY_PER_1K=0.0
export LR="${LR:-1e-6}"
export VAL_TEMP="${VAL_TEMP:-1.0}" VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"

# ---- retrieval reward ------------------------------------------------------------
# HB_SCORE_MIN needs >= HB_RETRIEVAL_WEIGHT of headroom below 0: the group-relative
# bonus is +/-w and is clipped into [HB_SCORE_MIN, 1], so a floor at 0 swallows the
# NEGATIVE deltas while keeping the positive ones -- a one-sided bonus that pushes
# the policy to search regardless of query quality.
export HB_SCORE_MIN="${HB_SCORE_MIN:--0.25}"
export HB_RETRIEVAL_WEIGHT="${HB_RETRIEVAL_WEIGHT:-0.20}"
export HB_RETRIEVAL_GROUP_BASELINE=1
export HB_RETRIEVAL_NOSEARCH_COVERAGE="${HB_RETRIEVAL_NOSEARCH_COVERAGE:-0.35}"
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-10}"

# ---- agent loop ------------------------------------------------------------------
export RETRIEVAL_URL="http://localhost:$GEN_PORT/retrieve"
export VERL_MAX_SEARCHES="${VERL_MAX_SEARCHES:-2}"
export VERL_SEARCH_THINK_BUDGET="${VERL_SEARCH_THINK_BUDGET:-1024}"
export VERL_THINK_BUDGET_TOKENS="${VERL_THINK_BUDGET_TOKENS:-3072}"
export VERL_ANSWER_RESERVE_TOKENS="${VERL_ANSWER_RESERVE_TOKENS:-3500}"
export VERL_MIN_ANSWER_TOKENS="${VERL_MIN_ANSWER_TOKENS:-1536}"

# ---- token budget ----------------------------------------------------------------
# The response now carries the loss-masked evidence as well as both turns.
# HARD RULE: ppo_max_token_len_per_gpu >= max_prompt + max_response, else
# rearrange_micro_batches asserts on the longest ACTUAL sequence. 8192+8192=16384.
export MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-16384}"
export LOGPROB_MAX_TOKEN_LEN="${LOGPROB_MAX_TOKEN_LEN:-16384}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.40}"

# ---- gen server: co-generates tasks+rubrics AND serves /retrieve ------------------
cleanup() { kill ${GEN_PID:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null 2>&1; then
    echo "FATAL: port $GEN_PORT already serving (orphan?)" >&2; exit 1
fi
LOGDIR=$S/logs_healthbench_rubric/$EXP; mkdir -p "$LOGDIR/prompts"
# TRAPI rejects reasoning_effort=none on gpt-chat-latest, and generation_server
# defaults that env var to "none" -- so every summarizer call 400s and /retrieve
# silently serves RAW passages for the whole run. This MUST be set before the
# server starts (attempt 1 exported it afterwards and the smoke gate caught it),
# and it is set INSIDE the command string because the server runs through
# `apptainer exec` + `bash -lc`, either of which can drop an outer export.
export TRAPI_NO_THINK_EFFORT=""
"${RUN_CPU[@]}" bash -lc "export TRAPI_NO_THINK_EFFORT='' && cd $MSR/verl_healthbench && python scripts/self_evolving/generation_server.py \
    --rubric_mode --prompt_dir $MSR/logs_healthbench_rubric/$EXP/prompts \
    --api_base '$TRAPI_BASE' --api_key '$TRAPI_KEY_' --model_name '$JUDGE' \
    --embed_api_base '$EMBED_BASE' --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --milvus_uri '$MILVUS_URI' --milvus_token root:Milvus \
    --milvus_collection medical_knowledge_v2 --milvus_top_k 8 \
    --retrieve_top_k ${RETRIEVE_TOP_K:-5} --retrieve_total ${RETRIEVE_TOTAL:-16} \
    --summarizer_api_base '$TRAPI_BASE' --summarizer_api_key '$TRAPI_KEY_' \
    --summarizer_model '$JUDGE' --summarizer_provider trapi \
    --n_queries ${N_QUERIES:-6} --questions_per_query 1 \
    --accuracy_window 64 --max_pool_size 200 --workers ${GEN_WORKERS:-8} \
    --log_dir $MSR/logs_healthbench_rubric/$EXP --host 0.0.0.0 --port $GEN_PORT" \
    > "$LOGDIR/gen_server.log" 2>&1 &
GEN_PID=$!
start=$SECONDS
until curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null; do
    kill -0 "$GEN_PID" 2>/dev/null || { echo "FATAL: gen server died" >&2; tail -40 "$LOGDIR/gen_server.log"; exit 1; }
    (( SECONDS - start > 1200 )) && { echo "FATAL: gen server unhealthy" >&2; exit 1; }
    sleep 5
done
echo "gen server healthy on :$GEN_PORT"

# Prove retrieval works before spending a step.
curl -sf -m 240 -X POST "localhost:$GEN_PORT/retrieve" -H 'content-type: application/json' \
    -d '{"question":"75yo, eGFR 38, on metformin - safe to continue?",
         "queries":["metformin contraindication eGFR threshold",
                    "metformin lactic acidosis risk renal impairment"]}' \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
assert d["n_merged"] >= 4, d
assert d["summarized"] is True, "summarizer NOT active: %s" % (d.get("fallback_reason"),)
print("retrieve smoke OK: %sq -> %s passages -> %s chars"
      % (d["n_queries"], d["n_merged"], d["chars"]))
' || { echo "FATAL: /retrieve smoke failed" >&2; exit 1; }

# ---- trainer ---------------------------------------------------------------------
export ACTOR_MODEL_PATH="$INIT"
export GEN_SERVER_URL="http://localhost:$GEN_PORT"
LOGF=$MSR/logs_healthbench_rubric/${EXP}_train.log
# WANDB_* set on the COMMAND so they survive the login shell's profile (see above).
"${RUN_GPU[@]}" bash -lc "cd $MSR/verl_healthbench && \
    WANDB_API_KEY='$WANDB_KEY_' WANDB_RUN_ID='$WANDB_RUN_ID' WANDB_RESUME=allow \
    bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric.sh \
      actor_rollout_ref.rollout.multi_turn.enable=True \
      actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
      actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESP:-6000} \
      actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
      actor_rollout_ref.rollout.multi_turn.tool_config_path=scripts/self_evolving/train/config/medical_retrieval_tool.yaml \
      actor_rollout_ref.rollout.agent.default_agent_loop=retrieval_tool_agent \
      +reward.custom_reward_function.reward_kwargs.cov_api_base='$TRAPI_BASE' \
      +reward.custom_reward_function.reward_kwargs.cov_api_key='$TRAPI_KEY_' \
      +reward.custom_reward_function.reward_kwargs.cov_model_name='$JUDGE' \
      +reward.custom_reward_function.reward_kwargs.cov_provider=trapi \
      2>&1 | tee $LOGF"
rc=${PIPESTATUS[0]}
kill "$GEN_PID" 2>/dev/null || true
exit $rc
