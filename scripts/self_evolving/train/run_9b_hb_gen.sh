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
# TWO FURTHER SWITCHES, each isolating one question against the SAME baseline
# (RETRIEVAL=0 EVOLVE=0 SELF_JUDGE=0 — the arm that produced the completed
# held-out result of 0.121 -> 0.352). Each new arm moves exactly one factor, so
# its delta is attributable:
#
#   EVOLVE=1      generation-prompt evolution ON. The gen server's /evolve loop
#                 rewrites the proposer and task+rubric-generator guidance from a
#                 structured error analysis of each step's rollouts.
#                 EVOLVE_EVERY defaults to 5 to MATCH test_freq. Evolution runs
#                 just before validation within a step, so on this cadence each
#                 guidance version is measured by exactly one held-out evaluation
#                 and the val deltas in the evolver's history are attributable to
#                 a single rewrite. Evolving every step instead would stack five
#                 rewrites between two val points and make the outcome signal --
#                 the only ungameable one the loop has -- uninterpretable.
#   SELF_JUDGE=1  the TRAINING reward is graded by a frozen local 9B (the model
#                 judging itself) instead of gpt-chat-latest. VALIDATION always
#                 stays on gpt-chat-latest, so held-out numbers remain comparable
#                 across every arm — this measures how much of the gain came from
#                 the judge's quality rather than from the self-evolving loop.
#
# Do not set both at once: the point of each is a single-factor delta.
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
EVOLVE="${EVOLVE:-0}"
SELF_JUDGE="${SELF_JUDGE:-0}"
REWARD_EVOLVE="${REWARD_EVOLVE:-0}"
if [ "$REWARD_EVOLVE" = 1 ] && [ "$RETRIEVAL" != 1 ]; then
    echo "FATAL: REWARD_EVOLVE=1 needs RETRIEVAL=1 — the coverage judge it evolves only" \
         "runs on searching rollouts, so with retrieval off it would rewrite a prompt" \
         "that is never called" >&2
    exit 1
fi
# This guard protects ONE experimental design: comparing an arm against the gpt-judge
# gen-RL baseline, where switching on evolution AND the judge together moves two factors
# at once and neither delta is attributable.
#
# It does NOT apply to a judge SWAP at fixed variant. "v3 with the local 9B judge" versus
# "v3 with gpt-chat-latest" differs in exactly one factor -- the judge -- because the
# variant is its own control. That is a legitimate and different question, so the
# combination is allowed when the caller declares that design explicitly rather than
# tripping into it.
if [ "$EVOLVE" = 1 ] && [ "$SELF_JUDGE" = 1 ] && [ "${ALLOW_EVOLVE_SELF_JUDGE:-0}" != 1 ]; then
    echo "FATAL: EVOLVE and SELF_JUDGE together confound both deltas against the gpt-judge" >&2
    echo "       baseline. If you are running a judge swap at fixed variant (v_N+selfjudge" >&2
    echo "       vs v_N+gptjudge, single factor), set ALLOW_EVOLVE_SELF_JUDGE=1." >&2
    exit 1
fi

# ---- adversarial specification refinement ----
# The reward for each generated task is a rubric another model wrote, and it can be
# satisfied without doing the clinical work. These switches test and repair it.
#
#   PROBE=1   Before a task is served, a FROZEN rubric-farmer writes the laziest
#             answer that ticks every box, and a rubric-BLIND honest answer is
#             written from the bare task. Both are graded. If the farmer wins, the
#             rubric cannot tell good medicine from empty words.
#   PATCH=1   A farmable rubric is REPAIRED rather than discarded: mint a negative
#             criterion from the farmed/honest contrast, keep it only if grading
#             proves it fires on the farmed answer and not the honest one, then
#             re-probe. HB_REFINE_ROUNDS bounds the loop. Rejection is the last
#             resort -- discarding specs starves the pool and turns the probe into a
#             difficulty filter.
#   SPEC_GAP=1  During training, a rubric-blind referee ranks each GRPO group's 8
#             rollouts. H = the fraction of decisive pairs where the rubric's order
#             contradicts the referee's. This changes NOTHING the actor trains on --
#             it detects the exploits the frozen farmer missed, because the real
#             policy is a better adversary and gets better as it trains.
#   SPEC_GAP_SHIP=1  Those confirmed exploits POST to /patch_spec, repairing the
#             rubric for future rollouts of that task.
#   HACK_MEMO=1  The exploit MODES accumulate into a memo in the generator's prompt,
#             so future rubrics do not have the same hole. This is the only switch
#             that improves generation itself; without it you patch forever.
SPEC_GAP="${SPEC_GAP:-0}"
SPEC_GAP_SHIP="${SPEC_GAP_SHIP:-0}"
PROBE="${PROBE:-0}"
PATCH="${PATCH:-0}"
HACK_MEMO="${HACK_MEMO:-0}"
if [ "$SPEC_GAP_SHIP" = 1 ] && [ "$EVOLVE" != 1 ]; then
    # The exploit buffer drains inside _maybe_evolve_generation, which returns at its
    # first guard when evolve_generation is False. Without EVOLVE=1 the buffer would
    # fill and nothing would ever ship -- the same class of silent no-op as the
    # /evolve_retrieval 500 that read healthy for a whole 60-step run.
    echo "FATAL: SPEC_GAP_SHIP=1 needs EVOLVE=1 (exploits drain on the evolve round)" >&2
    exit 1
fi
if [ "$SPEC_GAP_SHIP" = 1 ] && [ "$SPEC_GAP" != 1 ]; then
    echo "FATAL: SPEC_GAP_SHIP=1 needs SPEC_GAP=1 (nothing measures the exploits)" >&2
    exit 1
fi
if [ "$PATCH" = 1 ] && [ "$PROBE" != 1 ] && [ "$SPEC_GAP_SHIP" != 1 ]; then
    echo "FATAL: PATCH=1 needs PROBE=1 (refine before serving) or SPEC_GAP_SHIP=1" \
         "(repair on-policy exploits); on its own it has nothing to repair from" >&2
    exit 1
fi

_ARM=$([ "$RETRIEVAL" = 1 ] && echo retrieval || echo control)
[ "$EVOLVE" = 1 ] && _ARM="${_ARM}_evolve"
[ "$SELF_JUDGE" = 1 ] && _ARM="${_ARM}_selfjudge"
[ "$SPEC_GAP" = 1 ] && _ARM="${_ARM}_sg"
[ "$PROBE" = 1 ] && _ARM="${_ARM}_probe"
[ "$PATCH" = 1 ] && _ARM="${_ARM}_patch"
EXP="${EXP:-hb9b_gen_$_ARM}"
LOGDIR=$S/logs_hb9b; mkdir -p "$LOGDIR"

TRAPI_BASE=http://point.dd.works:18890/v1
TRAPI_KEY=$(cat $S/.trapi_key)
JUDGE=gpt-chat-latest_2026-05-28
GEN_PORT="${GEN_PORT:-8041}"
SUMM_PORT="${SUMM_PORT:-8199}"

SUMM_BASE="${SUMM_BASE:-http://localhost:$SUMM_PORT/v1}"
SUMM_MODEL="${SUMM_MODEL:-Qwen/Qwen3.5-9B}"
# A SECOND HOST serving that same model. Set it when SUMM_BASE is a dedicated inference
# box, so a hiccup there is absorbed by the small shared server instead of silently
# turning /retrieve into a raw-passage feed for the rest of the step.
SUMM_FALLBACK_BASE="${SUMM_FALLBACK_BASE:-}"
# WEB EVIDENCE. The embedded corpus effectively ends in 2019, so guideline editions and trials
# the rubrics name from the last five years are unreachable from Milvus at any ranking quality.
# With this on, /retrieve queries web search alongside Milvus and the GENERATOR grounds minted
# tasks the same way, both through one cache service so a fetch for either warms the other.
# ON BY DEFAULT (dvd, 2026-08-12). The embedded corpus ends in 2019, so every arm that does
# not query the web is capped on anything the rubrics name from the last five years. Set
# WEB_EVIDENCE=0 explicitly for a no-web control.
WEB_EVIDENCE="${WEB_EVIDENCE:-1}"
EVIDENCE_CACHE_URL="${EVIDENCE_CACHE_URL:-http://localhost:8055}"
EVIDENCE_CACHE_DB="${EVIDENCE_CACHE_DB:-/root/evidence_cache.sqlite}"
# WEB SEARCH TOOL (solver-facing; dvd 2026-08-12). Independent of WEB_EVIDENCE:
# that one feeds /retrieve through a GPT lookup the solver never sees, while this
# gives the SOLVER its own `web_search` tool whose results come VERBATIM from the
# Serper API -- no model composes the evidence, so the search behaviour and the
# reading of raw results both train. Serper is paid: every query goes through the
# serper cache service (kb/serper_cache_server.py), started below.
WEB_SEARCH_TOOL="${WEB_SEARCH_TOOL:-0}"
WEB_SEARCH_URL="${WEB_SEARCH_URL:-http://localhost:8056/search}"
SEARCH_CACHE_DB="${SEARCH_CACHE_DB:-/root/search_cache.sqlite}"
# Secrets (SERPER_API_KEY) live in the gitignored scripts/self_evolving/.env, or in
# key files under /scratch on the pods (serper_cache_server --key_file fallback).
if [ -f "$(dirname "$0")/../.env" ]; then set -a; . "$(dirname "$0")/../.env"; set +a; fi
# Does the frozen 9B live on a TRAINING GPU, or on another box? It decides how much
# memory the rollout engine may take, so it must be known before any util default is
# picked. The retrieval arm's 0.35 was never about retrieval -- it was the concession to
# a summarizer sharing GPU 3, and applying it to an arm whose summarizer is remote
# quietly runs the rollout engine on two thirds of the KV cache it could have had.
case "$SUMM_BASE" in
    *localhost*|*127.0.0.1*) SUMM_LOCAL=1 ;;
    *)                       SUMM_LOCAL=0 ;;
esac
# The self-judge IS the frozen-9B server: same model, same frozen weights, one
# process serving both roles.
SJUDGE_BASE="$SUMM_BASE"
SJUDGE_MODEL="$SUMM_MODEL"

# Judge routing, AFTER the endpoints above exist. VAL is pinned to gpt-chat-latest
# in EVERY arm so the held-out number stays one comparable series; only the
# TRAINING judge moves.
TRAIN_JUDGE_BASE="$TRAPI_BASE"; TRAIN_JUDGE_KEY="$TRAPI_KEY"
TRAIN_JUDGE_MODEL="$JUDGE";     TRAIN_JUDGE_PROVIDER=trapi
# The fallback fires only when the primary judge call raises. It deliberately
# points at the SAME endpoint as the primary rather than at TRAPI: routing the
# self-judge arm's failures to gpt-chat-latest would silently grade part of the
# batch with the judge this arm exists to do without. Watch reward/judge_fail/mean
# instead -- an outage should be visible, not quietly repaired.
FALLBACK_JUDGE_BASE="$TRAPI_BASE"; FALLBACK_JUDGE_KEY="$TRAPI_KEY"
FALLBACK_JUDGE_MODEL="$JUDGE";     FALLBACK_JUDGE_PROVIDER=trapi
if [ "$SELF_JUDGE" = 1 ]; then
    TRAIN_JUDGE_BASE="$SJUDGE_BASE"; TRAIN_JUDGE_KEY=EMPTY
    TRAIN_JUDGE_MODEL="$SJUDGE_MODEL"; TRAIN_JUDGE_PROVIDER=vllm
    FALLBACK_JUDGE_BASE="$SJUDGE_BASE"; FALLBACK_JUDGE_KEY=EMPTY
    FALLBACK_JUDGE_MODEL="$SJUDGE_MODEL"; FALLBACK_JUDGE_PROVIDER=vllm
    # provider=vllm sets chat_template_kwargs.enable_thinking=False and temp 0 in
    # _call_api, so the 9B answers the grader template directly. Without that it
    # would spend all 512 max_tokens reasoning and never emit a verdict, which
    # parses as "not met" and would look like a terrible model rather than a
    # misconfigured judge.
    REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-48}"
fi
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
# Train on the SAME length-adjusted score validation reports. Set explicitly rather
# than relying on the default: with train and report disagreeing, nothing in the
# training reward opposes verbosity, and the predecessor run grew from 3357 to 9165
# answer chars while 82% of its raw rubric gain was eaten by the length adjustment
# it was never shown. HB_TRAIN_LENGTH_ADJ=0 opts back out.
export HB_TRAIN_LENGTH_ADJ="${HB_TRAIN_LENGTH_ADJ:-1}"
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-10}"

# Judge voting. Off by default historically, and that is what bounds the
# specification-gap measurement: a pair only counts as decisive if the rubric's
# margin exceeds the grader's own noise, and at single vote the score-level sd is
# ~0.11, which erases most pairs. Adaptive spends the extra votes only on the
# criteria whose flips dominate that noise (long ones, and "at least one of"
# disjunctions), so it buys the resolution at ~15-20% more judge calls instead of
# 40%. Measure the realized sd offline and set SPEC_GAP_MARGIN from it.
if [ "$SPEC_GAP" = 1 ]; then
    export HB_JUDGE_VOTES="${HB_JUDGE_VOTES:-3}"
    export HB_JUDGE_VOTES_ADAPTIVE="${HB_JUDGE_VOTES_ADAPTIVE:-1}"
fi

# ---- gen-server side of hack-then-patch (read by generation_server.py) ----
# Exported here so the arm switches above are the single source of truth; the gen
# server inherits this environment.
export HB_PROBE="$PROBE"
export HB_PROBE_MODE="${HB_PROBE_MODE:-log}"     # log = measure without rejecting
export HB_PROBE_RATE="${HB_PROBE_RATE:-0.34}"
export HB_PATCH="$PATCH"
export HB_REFINE_ROUNDS="${HB_REFINE_ROUNDS:-1}"
# Refinement makes a specification ~2.5x more expensive to produce (2 extra
# generations + per-criterion grading for the probe, plus a mint and its validation
# per repair round), and a worker is SERIAL within one spec. Same worker count would
# cut spec throughput by the same factor and the trainer would end up blocking on
# /sample. The work is all HTTP wait, so the fix is more of them; the TRAPI budget is
# not the constraint here (generation is ~0.4 req/s against a ~14 req/s ceiling, the
# rubric judge dominates), latency x concurrency is.
if [ "$PROBE" = 1 ]; then
    GEN_WORKERS="${GEN_WORKERS:-20}"
    export HB_PROBE_CONCURRENCY="${HB_PROBE_CONCURRENCY:-16}"
else
    # 16, not 8, even with no probe to pay for. The dataloader prefetches roughly
    # (dataloader_workers x prefetch_factor x batch) ~= 500 samples before the first
    # step, and 8 workers at ~6s per generation supply ~29/min, so that burst drains
    # the pool for ~15 minutes and /sample falls back to re-serving stale history.
    # Measured on the evolve-only arm: 6 groups served from history before it caught
    # up. Steady state needs only ~3 specs/min; this is entirely about the burst.
    GEN_WORKERS="${GEN_WORKERS:-16}"
fi
export HB_HACK_MEMO="$HACK_MEMO"

# ---- token budget: IDENTICAL in both arms (this is the confound that bit us) ----
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
ROLLOUT_MAX_LEN="${ROLLOUT_MAX_LEN:-14336}"      # 6144 prompt + 8192 response
# HARD RULE: >= max_prompt + max_response, else rearrange_micro_batches asserts on
# the first long rollout (the assert is on the longest ACTUAL sequence).
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-14336}"
LOGPROB_MAX_TOKEN_LEN="${LOGPROB_MAX_TOKEN_LEN:-14336}"

cleanup() { kill ${GEN_PID:-} ${SUMM_PID:-} ${SJUDGE_PID:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

curl -sf -m 10 "$EMBED_BASE/models" >/dev/null \
    || { echo "FATAL: embed server unreachable at $EMBED_BASE" >&2; exit 1; }

# ---- retrieval-only wiring -------------------------------------------------------
AGENT_ARGS=()
if [ "$RETRIEVAL" = 1 ]; then
    # No floor at all (dvd 2026-08-11). The -0.25 here previously existed only to keep
    # the +/-w retrieval bonus from being clipped one-sidedly; with no floor that concern
    # disappears entirely and every rollout keeps its true ordering.
    export HB_SCORE_MIN="${HB_SCORE_MIN:-none}"
    export HB_RETRIEVAL_WEIGHT="${HB_RETRIEVAL_WEIGHT:-0.20}"
    export HB_RETRIEVAL_GROUP_BASELINE=1
    export HB_RETRIEVAL_NOSEARCH_COVERAGE="${HB_RETRIEVAL_NOSEARCH_COVERAGE:-0.35}"
    export RETRIEVAL_URL="http://localhost:$GEN_PORT/retrieve"
    export VERL_MAX_SEARCHES="${VERL_MAX_SEARCHES:-2}"
    export VERL_SEARCH_THINK_BUDGET="${VERL_SEARCH_THINK_BUDGET:-1024}"
    export VERL_THINK_BUDGET_TOKENS="${VERL_THINK_BUDGET_TOKENS:-3072}"
    export VERL_ANSWER_RESERVE_TOKENS="${VERL_ANSWER_RESERVE_TOKENS:-3500}"
    export VERL_MIN_ANSWER_TOKENS="${VERL_MIN_ANSWER_TOKENS:-1536}"
    # The web arm swaps in the two-tool config; every other line of the rollout
    # setup is shared, so the tool set is the single factor between the arms.
    TOOL_CONFIG=scripts/self_evolving/train/config/medical_retrieval_tool.yaml
    if [ "$WEB_SEARCH_TOOL" = 1 ]; then
        TOOL_CONFIG=scripts/self_evolving/train/config/medical_retrieval_web_tool.yaml
        export WEB_SEARCH_URL
    fi
    AGENT_ARGS=(
        actor_rollout_ref.rollout.multi_turn.enable=True
        actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESP:-6000}"
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
        actor_rollout_ref.rollout.agent.default_agent_loop=retrieval_tool_agent
        # The coverage judge follows the TRAIN judge: in the self-judge arm the
        # model must grade its own retrieval too, or the arm would still be
        # partly gpt-graded and would not answer the question it exists for.
        +reward.custom_reward_function.reward_kwargs.cov_api_base="$TRAIN_JUDGE_BASE"
        +reward.custom_reward_function.reward_kwargs.cov_api_key="$TRAIN_JUDGE_KEY"
        +reward.custom_reward_function.reward_kwargs.cov_model_name="$TRAIN_JUDGE_MODEL"
        +reward.custom_reward_function.reward_kwargs.cov_provider="$TRAIN_JUDGE_PROVIDER"
    )
    echo "summarizer will be served by the frozen-9B server at $SUMM_BASE"
else
    # A floor of 0.0 threw away the gradient exactly where the loss lives. Measured on
    # hb9b_specgap_full_rewrite at step 165: 26.7% of val tasks score BELOW zero (mean
    # -0.276), and a tripped trap flips a task from +0.57 to -0.53 -- yet every one of
    # those rollouts was clipped to the same 0.0 as a bland, safe, content-free answer,
    # so GRPO had no ordering to learn from in that whole region. critic/score/min was
    # 0.0 at every logged step, i.e. the floor bound continuously, not occasionally.
    #
    # This is the v4 post-mortem documented at healthbench_pro.HB_SCORE_MIN ("slightly
    # bad and catastrophic become indistinguishable... whole GRPO groups go
    # zero-variance"), and the 27B rubric recipe already defaults to -0.5. The 9B arms
    # were the ones still on 0.0. Validation is unaffected: the reported signed metrics
    # are computed unclipped.
    # Disabled entirely rather than floored at -0.5: a floor still collapses everything
    # below it, and the worst observed val task reaches -2.06.
    export HB_SCORE_MIN="${HB_SCORE_MIN:-none}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"
fi

# ---- ONE frozen 9B serving every non-actor role ----------------------------------
# The retrieval summarizer and the self-judge are the SAME model at the SAME frozen
# weights, so they are one vLLM server, not two: a second copy would burn ~18 GB of
# weights and a second scheduler to do a job the first one is already sized for.
#
# Frozen, never the actor's live weights. A judge that tracks the policy makes the
# reward non-stationary (the score drifts for reasons unrelated to the answers), and
# a summarizer that tracks it breaks the invariant that offline SFT traces and online
# rollouts see byte-identical evidence briefs.
#
# Load, at batch 32 x n 8: ~256 summarizer calls per step (search turns), plus in the
# self-judge arm ~560 rubric calls (one PER CRITERION) and ~256 coverage calls. Hence
# the larger slice and max-num-seqs when it is judging as well as summarizing.
FROZEN_NEEDED=0
[ "$RETRIEVAL" = 1 ] && FROZEN_NEEDED=1
[ "$SELF_JUDGE" = 1 ] && FROZEN_NEEDED=1
if [ "$FROZEN_NEEDED" = 1 ]; then
    # FROZEN_* size a LOCAL server and are inert when SUMM_BASE is remote.
    if [ "$SELF_JUDGE" = 1 ]; then
        FROZEN_MEM="${FROZEN_MEM:-0.30}"; FROZEN_SEQS="${FROZEN_SEQS:-256}"
    else
        FROZEN_MEM="${FROZEN_MEM:-0.16}"; FROZEN_SEQS="${FROZEN_SEQS:-96}"
    fi
    # The rollout engine's share, decided ONCE and only by whether something else is
    # going to sit on these GPUs. Layered ":-" defaults used to settle this in three
    # places, and they disagreed: the self-judge line read 0.30 but a retrieval arm had
    # already pinned 0.35 above it, so the value the code appeared to choose was not the
    # value it used.
    if [ "$SUMM_LOCAL" = 1 ]; then
        [ "$SELF_JUDGE" = 1 ] && VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.30}" \
                              || VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.35}"
    else
        VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"
    fi
    if ! curl -sf -m 5 "$SUMM_BASE/models" >/dev/null 2>&1; then
        # A REMOTE SUMM_BASE that is not answering must stop the launch, not be quietly
        # replaced by a local copy. The fallback path spawns a summarizer on FROZEN_GPU
        # (default 3) and drops the rollout engine's memory share to match -- so an arm
        # configured for a dedicated inference box would come up on 3 training GPUs
        # instead of 4, with a summarizer sized for one shared GPU, and nothing but a
        # line 300 logs deep would say so. Silent capacity loss is worse than exit 1.
        case "$SUMM_BASE" in
            *localhost*|*127.0.0.1*) : ;;   # local by design: spawning it here is correct
            *) echo "FATAL: SUMM_BASE=$SUMM_BASE is remote but not answering." >&2
               echo "       Refusing to silently spawn a local summarizer on a training GPU." >&2
               echo "       Start the dedicated server (serve/serve_frozen9b_dp.sh) or set" >&2
               echo "       SUMM_BASE=http://localhost:\$SUMM_PORT/v1 to serve it here." >&2
               exit 1 ;;
        esac
        CUDA_VISIBLE_DEVICES="${FROZEN_GPU:-3}" MODEL="$SUMM_MODEL" PORT="$SUMM_PORT" \
        TP=1 MEM="$FROZEN_MEM" MAXSEQS="$FROZEN_SEQS" MAXLEN=16384 \
            bash scripts/self_evolving/serve/serve_summarizer.sh \
            > "$LOGDIR/frozen9b_${EXP}.log" 2>&1 &
        SUMM_PID=$!
        start=$SECONDS
        until curl -sf -m 5 "$SUMM_BASE/models" >/dev/null; do
            kill -0 "$SUMM_PID" 2>/dev/null || { echo "FATAL: frozen 9B died" >&2; tail -40 "$LOGDIR/frozen9b_${EXP}.log"; exit 1; }
            (( SECONDS - start > 1800 )) && { echo "FATAL: frozen 9B unhealthy" >&2; exit 1; }
            sleep 5
        done
    fi
    echo "frozen 9B healthy at $SUMM_BASE (mem=$FROZEN_MEM seqs=$FROZEN_SEQS)"
    if [ -n "$SUMM_FALLBACK_BASE" ]; then
        # Checked at launch, not on first use. A fallback that is only exercised when
        # the primary is already failing is a fallback nobody has ever tested, and the
        # moment it matters is the moment there is no attention to spare for it.
        curl -sf -m 10 "$SUMM_FALLBACK_BASE/models" >/dev/null \
            || { echo "FATAL: summarizer fallback $SUMM_FALLBACK_BASE unreachable" >&2; exit 1; }
        echo "summarizer fallback healthy at $SUMM_FALLBACK_BASE ($SUMM_MODEL)"
    fi
fi

# Prove the judge returns a GRADABLE verdict before spending a step. A server that
# 200s but replies with reasoning prose parses as "not met" on every criterion,
# which is indistinguishable from a model that cannot answer at all.
if [ "$SELF_JUDGE" = 1 ]; then
    curl -sf -m 120 "$SJUDGE_BASE/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"$SJUDGE_MODEL\",\"max_tokens\":64,\"temperature\":0,
             \"chat_template_kwargs\":{\"enable_thinking\":false},
             \"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly the word: true\"}]}" \
      | /usr/local/bin/python -c '
import json, sys
d = json.load(sys.stdin)
t = d["choices"][0]["message"]["content"] or ""
assert "true" in t.lower(), "self-judge did not answer directly: %r" % (t[:200],)
print("self-judge smoke OK:", t.strip()[:60])
' || { echo "FATAL: self-judge smoke failed" >&2; exit 1; }
fi

# ---- evidence cache service (only when web evidence is on) -----------------------
# Started here rather than by hand so a relaunched arm never trains with the cache missing:
# without it every web miss costs a live ~6.6s call and ~8.8k tokens, and nothing would say so
# except a hit-rate of zero. --restore seeds a fresh pod from the /scratch snapshot, which is
# what makes the accumulated fetches survive preemption.
if [ "$WEB_EVIDENCE" = 1 ]; then
    case "$EVIDENCE_CACHE_URL" in
      *localhost*|*127.0.0.1*)
        if ! curl -sf -m 5 "$EVIDENCE_CACHE_URL/healthz" >/dev/null 2>&1; then
            echo "starting evidence cache service at $EVIDENCE_CACHE_URL"
            EV_PORT="${EVIDENCE_CACHE_URL##*:}"; EV_PORT="${EV_PORT%%/*}"
            nohup /usr/local/bin/python scripts/self_evolving/kb/evidence_cache_server.py                 --port "$EV_PORT" --db "$EVIDENCE_CACHE_DB" --restore                 --api_base "$TRAPI_BASE" --model "$JUDGE"                 >> "$LOGDIR/evidence_cache.log" 2>&1 &
            start=$SECONDS
            until curl -sf -m 5 "$EVIDENCE_CACHE_URL/healthz" >/dev/null 2>&1; do
                (( SECONDS - start > 120 )) && { echo "FATAL: evidence cache did not come up" >&2
                                                 tail -20 "$LOGDIR/evidence_cache.log" >&2; exit 1; }
                sleep 3
            done
        fi
        ;;
    esac
    echo "evidence cache: $(curl -s -m 5 "$EVIDENCE_CACHE_URL/stats")"
fi

# ---- serper search cache (only when the solver has the web_search tool) ----------
# Same rationale as the evidence cache: started here so a relaunched arm never
# trains with the cache missing and re-buys queries already paid for. --restore
# seeds a fresh pod from the /scratch snapshot.
if [ "$WEB_SEARCH_TOOL" = 1 ]; then
    WS_BASE="${WEB_SEARCH_URL%/search}"
    case "$WEB_SEARCH_URL" in
      *localhost*|*127.0.0.1*)
        if ! curl -sf -m 5 "$WS_BASE/healthz" >/dev/null 2>&1; then
            echo "starting serper cache service at $WEB_SEARCH_URL"
            WS_PORT="${WS_BASE##*:}"
            nohup /usr/local/bin/python scripts/self_evolving/kb/serper_cache_server.py \
                --port "$WS_PORT" --db "$SEARCH_CACHE_DB" --restore \
                >> "$LOGDIR/serper_cache.log" 2>&1 &
            start=$SECONDS
            until curl -sf -m 5 "$WS_BASE/healthz" >/dev/null 2>&1; do
                (( SECONDS - start > 60 )) && { echo "FATAL: serper cache did not come up (missing SERPER_API_KEY?)" >&2
                                                tail -20 "$LOGDIR/serper_cache.log" >&2; exit 1; }
                sleep 3
            done
        fi
        ;;
    esac
    echo "serper cache: $(curl -s -m 5 "$WS_BASE/stats")"
fi

# ---- gen server: co-generates the TRAINING tasks + rubrics (both arms) ------------
if curl -sf -m 5 "localhost:$GEN_PORT/healthz" >/dev/null 2>&1; then
    echo "FATAL: port $GEN_PORT already serving (orphan?)" >&2; exit 1
fi
PROMPT_DIR="$LOGDIR/$EXP/prompts"; mkdir -p "$PROMPT_DIR"
# Retrieval-REWARD evolution: /evolve_retrieval rewrites the coverage judge here and
# the reward re-reads it on mtime change. Both sides must name the SAME file, or the
# rewrites land somewhere nothing reads and the run trains on the frozen v0 prompt
# while every log claims evolution is on.
COVERAGE_PROMPT_FILE="$PROMPT_DIR/coverage_prompt.json"
export HB_COVERAGE_PROMPT_FILE="$COVERAGE_PROMPT_FILE"
SUMM_FLAGS=()
if [ "$RETRIEVAL" = 1 ]; then
    SUMM_FLAGS=(--summarizer_api_base "$SUMM_BASE"
                --summarizer_model "$SUMM_MODEL"
                --summarizer_provider vllm)
    [ -n "$SUMM_FALLBACK_BASE" ] && SUMM_FLAGS+=(
        --summarizer_fallback_api_base "$SUMM_FALLBACK_BASE")
fi
[ "$WEB_EVIDENCE" = 1 ] && SUMM_FLAGS+=(--web_evidence
                                        --evidence_cache_url "$EVIDENCE_CACHE_URL")
/usr/local/bin/python scripts/self_evolving/generation_server.py \
    --rubric_mode --prompt_dir "$PROMPT_DIR" \
    --coverage_prompt_file "$COVERAGE_PROMPT_FILE" \
    --api_base "$TRAPI_BASE" --api_key "$TRAPI_KEY" --model_name "$JUDGE" \
    --embed_api_base "$EMBED_BASE" --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --milvus_uri "$MILVUS_URI" --milvus_token root:Milvus \
    --milvus_collection medical_knowledge_v2 --milvus_top_k 8 \
    --retrieve_top_k "${RETRIEVE_TOP_K:-5}" --retrieve_total "${RETRIEVE_TOTAL:-16}" \
    "${SUMM_FLAGS[@]}" \
    --n_queries "${N_QUERIES:-6}" --questions_per_query 1 \
    --accuracy_window 64 --max_pool_size 200 \
    --workers "$GEN_WORKERS" --log_dir "$LOGDIR/$EXP" \
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

# Prove /evolve_retrieval is REACHABLE before training. It first fires at step 5,
# and its failure is invisible by construction: the trainer's raise_for_status is
# swallowed by a blanket except so the round logs FAILED once and returns {}, and
# the COVERAGE_EVOLVE_FAILING marker is written inside the handler so it is never
# dropped when the handler itself cannot be entered. A monitor watching markers
# reads healthy forever while the treatment silently never runs. A shipped build
# did exactly this (HTTP 500, KeyError 'server', every round of a 60-step run).
# Empty `cases` exercises routing and state access without touching a model.
if [ "$REWARD_EVOLVE" = 1 ]; then
    _rc=$(curl -s -o /dev/null -w '%{http_code}' -m 60 -X POST \
          "localhost:$GEN_PORT/evolve_retrieval" -H 'content-type: application/json' \
          -d '{"step":0,"cases":[]}')
    [ "$_rc" = 200 ] || { echo "FATAL: /evolve_retrieval returned HTTP $_rc (want 200); reward evolution would silently never run" >&2
                          tail -30 "$LOGDIR/gen_server_${EXP}.log" >&2; exit 1; }
    echo "evolve_retrieval smoke OK (HTTP 200)"
fi

# Same reasoning for /patch_spec: it first fires at step 5, the trainer swallows its
# failure, and the PATCH_FAILING marker is written inside the handler. Empty `cases`
# proves routing + state access without touching a model.
if [ "$PATCH" = 1 ]; then
    _rc=$(curl -s -o /dev/null -w '%{http_code}' -m 60 -X POST \
          "localhost:$GEN_PORT/patch_spec" -H 'content-type: application/json' \
          -d '{"step":0,"cases":[]}')
    [ "$_rc" = 200 ] || { echo "FATAL: /patch_spec returned HTTP $_rc (want 200); spec repair would silently never run" >&2
                          tail -30 "$LOGDIR/gen_server_${EXP}.log" >&2; exit 1; }
    echo "patch_spec smoke OK (HTTP 200)"
fi

# WARM THE POOL before the trainer starts. /healthz answers as soon as the process is
# up, which says nothing about whether any specification has been generated yet --
# NOTE this is a floor, not a guarantee: the dataloader's initial prefetch pulls far
# more than this gate waits for, so the burst can still outrun generation. Worker
# count is what covers that; pool_starved in /stats is what proves it did.
# and with refinement each one costs a proposer call, a co-generation call, a probe
# (2 generations + per-criterion grading) and possibly a repair round. Without this
# gate the first training step blocks inside /sample for minutes and the run looks
# hung. Waiting here instead makes the cost visible and pays it once.
_want=$(( ${TRAIN_BS:-32} * 2 ))
_deadline=$(( SECONDS + ${GEN_WARMUP_S:-1800} ))
echo "warming the generation pool to $_want specs (workers=$GEN_WORKERS, probe=$PROBE)..."
while :; do
    _pool=$(curl -sf -m 10 "localhost:$GEN_PORT/stats" | python3 -c \
        'import json,sys; print(json.load(sys.stdin).get("pool_size", 0))' 2>/dev/null || echo 0)
    [ "${_pool:-0}" -ge "$_want" ] && { echo "pool warm: $_pool specs"; break; }
    if [ "$SECONDS" -ge "$_deadline" ]; then
        echo "FATAL: pool only reached $_pool/$_want in ${GEN_WARMUP_S:-1800}s." >&2
        echo "  Generation cannot keep up with training. Check $LOGDIR/gen_server_${EXP}.log" >&2
        echo "  for probe/mint failures, then raise GEN_WORKERS or lower HB_PROBE_RATE." >&2
        curl -sf -m 10 "localhost:$GEN_PORT/stats" >&2 || true
        exit 1
    fi
    sleep 20
done

# Prove the REFEREE endpoint answers before training. Its failure mode is the same
# shape: _run_spec_gap returns {} on any exception, so a bad endpoint degrades to
# measure-off and every step logs a referee failure that nothing is watching. One
# real ranking call on a hand-built group costs a couple of seconds and settles it.
# Also asserts the referee prefers the SHORT CORRECT answer over the long unsafe
# one, which is the prompt's whole premise -- a referee that reads as a length proxy
# would down-weight exactly the groups where the rubric correctly punished verbosity.
if [ "$SPEC_GAP" = 1 ]; then
    # Defaults to the VAL judge, which is gpt-chat-latest in every arm. Never
    # TRAIN_JUDGE_*: under SELF_JUDGE=1 that flips to the frozen local 9B, i.e. the
    # policy's own family, and a referee sharing the policy's blind spots cannot
    # detect a hack the two of them share.
    REFEREE_BASE="${REFEREE_BASE:-$TRAPI_BASE}" \
    REFEREE_KEY="${REFEREE_KEY:-$TRAPI_KEY}" \
    REFEREE_MODEL="${REFEREE_MODEL:-$JUDGE}" \
    REFEREE_PROVIDER="${REFEREE_PROVIDER:-trapi}" \
    /usr/local/bin/python scripts/self_evolving/analysis/referee_smoke.py || {
        echo "FATAL: referee smoke failed; the specification-gap measurement would silently never run" >&2
        exit 1; }
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
    +data.self_evolving.evolve_generation=$([ "$EVOLVE" = 1 ] && echo True || echo False) \
    +data.self_evolving.evolve_every_n_steps="${EVOLVE_EVERY:-5}" \
    +data.self_evolving.evolve_num_examples="${EVOLVE_N:-24}" \
    +data.self_evolving.evolve_retrieval_reward=$([ "$REWARD_EVOLVE" = 1 ] && echo True || echo False) \
    +data.self_evolving.evolve_retrieval_every_n_steps="${REWARD_EVOLVE_EVERY:-5}" \
    +data.self_evolving.spec_gap=$([ "$SPEC_GAP" = 1 ] && echo True || echo False) \
    +data.self_evolving.spec_gap_ship_exploits=$([ "$SPEC_GAP_SHIP" = 1 ] && echo True || echo False) \
    +data.self_evolving.spec_gap_margin="${SPEC_GAP_MARGIN:-0.05}" \
    +data.self_evolving.spec_gap_min_pairs="${SPEC_GAP_MIN_PAIRS:-3}" \
    +data.self_evolving.spec_gap_swap="${SPEC_GAP_SWAP:-True}" \
    +data.self_evolving.spec_gap_concurrency="${SPEC_GAP_CONCURRENCY:-16}" \
    +data.self_evolving.spec_gap_deadline_s="${SPEC_GAP_DEADLINE_S:-120}" \
    +data.self_evolving.spec_gap_exploit_margin="${SPEC_GAP_EXPLOIT_MARGIN:-0.15}" \
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
    +reward.custom_reward_function.reward_kwargs.api_base="$TRAIN_JUDGE_BASE" \
    +reward.custom_reward_function.reward_kwargs.api_key="$TRAIN_JUDGE_KEY" \
    +reward.custom_reward_function.reward_kwargs.model_name="$TRAIN_JUDGE_MODEL" \
    +reward.custom_reward_function.reward_kwargs.provider="$TRAIN_JUDGE_PROVIDER" \
    +reward.custom_reward_function.reward_kwargs.fallback_api_base="$FALLBACK_JUDGE_BASE" \
    +reward.custom_reward_function.reward_kwargs.fallback_api_key="$FALLBACK_JUDGE_KEY" \
    +reward.custom_reward_function.reward_kwargs.fallback_model_name="$FALLBACK_JUDGE_MODEL" \
    +reward.custom_reward_function.reward_kwargs.fallback_provider="$FALLBACK_JUDGE_PROVIDER" \
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
