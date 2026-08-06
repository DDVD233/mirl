#!/usr/bin/env bash
# AICR launcher for the LATEST healthbench rubric / curriculum-evolution RL:
# the v13 recipe (gpt56-SFT warm start, redesigned retrieval loop, rebuilt KB,
# self train-judge on server 5, gpt-chat-latest val judge) plus the v10
# frontier /evolve meta-optimizer, continuing the evolved prompt state.
#
# Meant to run as RUN_SCRIPT inside train_chain.sbatch on 4x B200:
#   RUN_SCRIPT=$WORK/mirl/scripts/self_evolving/aicr/run_healthbench_evolve.sh \
#     sbatch --job-name=hb-evolve -c 48 --mem=800G \
#     $WORK/mirl/scripts/self_evolving/aicr/train_chain.sbatch
#
# Path strategy: the MSR scripts hardcode /scratch/sheng/self_evolving/...;
# instead of patching them, bind our AICR dirs to those paths inside the SIF:
#   /scratch/dvdai_mit/self_evolving      -> /scratch/sheng/self_evolving
#   /work/mit/ppliang_mit/dvdai/mirl      -> /scratch/sheng/self_evolving/verl_healthbench
# so every hardcoded path (repo, hf_cache, keys, checkpoints, logs) resolves.
#
# External deps (all verified reachable from AICR compute nodes 2026-07-30):
#   server5 vLLM teacher/self-judge  http://point.dd.works:18184/v1
#   TRAPI proxy (val judge + evolve) http://point.dd.works:18890/v1
#   mib embedding server             http://mib.media.mit.edu:18001/v1
#   mib Milvus (medvecdb)            http://mib.media.mit.edu:19531
set -uo pipefail

WORK=/work/mit/ppliang_mit/dvdai
S=/scratch/dvdai_mit/self_evolving
SIF=$S/sif/verl-selfevolving-cu130-vllm0.22.1.sif
MSR=/scratch/sheng/self_evolving   # container-side mount point

# Diagnostic dispatch (dvd 2026-08-02): if chain/DIAG_MODE exists, run the
# diagnostic launcher instead of the evolve pipeline (content = mode:
# "trainval" or "seeded"). Remove the file to return to normal runs.
if [[ -f "$S/chain/DIAG_MODE" ]]; then
    exec bash "$WORK/mirl/scripts/self_evolving/aicr/run_hb_diag.sh" "$(cat "$S/chain/DIAG_MODE")"
fi

# v16 (2026-08-02) = fresh from the SFT init after the 3-agent v15 post-mortem:
#   - /evolve was dead all of v15 (undated TRAPI id 404'd silently) -> dated id
#     gpt-5.6-sol_2026-07-09 + loud failures (EVOLVE_FAILING marker).
#   - v15 trained on the ORPHANED v14 gen server (port 8006 collision) -> own
#     port + pre-flight check that nothing else already serves it.
#   - curriculum was frozen AND anti-val -> new v16 prompt seed with guidance
#     rewritten to the measured benchmark distribution (genre mix, prompt
#     surface form, rubric shape); evolve history starts fresh.
#   - style exemplar corpus restored to seed_corpora/.
#   - dominant failure was repetition-loop degeneration -> train-only
#     dup-8gram penalty in healthbench_pro.py (HB_REP_*).
export EXP="${EXP:-healthbench_rubric_qwen36_27b_v16_aicr}"
GEN_PORT=${GEN_PORT:-8016}
export GEN_SERVER_URL="http://localhost:$GEN_PORT"
export RETRIEVAL_URL="http://localhost:$GEN_PORT/retrieve"
export WANDB_API_KEY="$(cat "$S/.wandb_key")"   # dvd's key — the shared ~/.wandb_env belongs to weianxie
# Keep ONE wandb run across the 24h chain links: a fixed run id + resume=allow
# makes each restarted trainer append to the same run instead of minting a new
# one per day. verl resumes at the checkpoint step, so the x-axis stays clean.
export WANDB_RESUME=allow
export WANDB_RUN_ID="${WANDB_RUN_ID:-$EXP}"

BINDS=(--bind "$S:$MSR" --bind "$WORK/mirl:$MSR/verl_healthbench")
RUN_GPU=(apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF")
RUN_CPU=(apptainer exec --writable-tmpfs "${BINDS[@]}" "$SIF")

# --- teacher resolution: server5 is gone (2026-08-01); the Qwen teacher now
# runs as the qwen-teacher chain job on AICR and publishes its (moving)
# endpoint to chain/TEACHER_ENDPOINT. Wait for it before starting anything —
# gen server and judges silently degrade without it.
# All-GPT roles (dvd 2026-08-03): generation + fallback judge on TRAPI; the
# local Qwen teacher is no longer a healthbench dependency (p1-mimic only).
TRAPI_BASE="http://point.dd.works:18890/v1"
export CHAT_PROVIDER=trapi
export API_BASE="$TRAPI_BASE"
export API_KEY="$(cat "$S/.trapi_key")"
export MODEL_NAME="gpt-5.4_2026-03-05"
export FALLBACK_JUDGE_BASE="$TRAPI_BASE"
export FALLBACK_JUDGE_MODEL="gpt-5.4_2026-03-05"
export FALLBACK_JUDGE_PROVIDER="trapi"

# --- 2026-08-01 mid-run config change (per dvd, at ~step 60): val slid
# 0.368 -> 0.281 while train reward held ~0.63 (self-judge drift + entropy
# creep 1.12 -> 1.49). (1) entropy bonus OFF; (2) TRAIN judge -> external
# gpt-chat-latest via TRAPI (v10 role allocation) to break the self-grading
# loop. Fallback judge stays the local teacher.
export ENTROPY_COEFF=0.0
export TRAIN_JUDGE_BASE="http://point.dd.works:18890/v1"
export TRAIN_JUDGE_MODEL="gpt-chat-latest_2026-05-28"
export TRAIN_JUDGE_PROVIDER="trapi"
export TRAIN_JUDGE_KEY="$(cat "$S/.trapi_key")"

# Sanity: the SFT init must be fully transferred before we start.
INIT=$S/checkpoints/self_evolving_medical/gpt56_sft_qwen36_27b/global_step_28/hf_merged
if [[ ! -f "$INIT/config.json" ]]; then
    echo "FATAL: gpt56 SFT init not present at $INIT (transfer still running?)" >&2
    exit 1
fi

# Seed the evolvable prompt state from the v10/v13 curriculum (only on the
# first chain link; later links continue the evolved state in place).
LOG_DIR=$S/logs_healthbench_rubric/$EXP
PROMPT_DIR=$LOG_DIR/prompts
if [[ ! -d "$PROMPT_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    cp -r "$S/logs_healthbench_rubric/v16_prompt_seed/prompts" "$PROMPT_DIR"
    echo "seeded prompt state from v16_prompt_seed (val-aligned guidance, fresh evolve history)"
fi

# --- 1. generation server (rubric mode, :$GEN_PORT, CPU-only) ----------------
# Pre-flight: if SOMETHING already answers on our port, an orphaned server from
# an earlier run holds it — starting ours would silently bind-fail while the
# healthz gate passes against the orphan (exactly how v15 trained on v14's
# frozen prompts). Fail loudly instead.
if curl -sf "localhost:$GEN_PORT/healthz" > /dev/null 2>&1; then
    echo "FATAL: port $GEN_PORT already serves a gen server (orphan from an earlier run?) — kill it first" >&2
    exit 1
fi
env EXP="$EXP" EVOLVE_MODEL="${EVOLVE_MODEL:-gpt-5.6-sol_2026-07-09}" GEN_SERVER_PORT="$GEN_PORT" \
    "${RUN_CPU[@]}" bash "$MSR/verl_healthbench/scripts/self_evolving/serve/start_gen_server_rubric_s4.sh" \
    > "$S/logs/gen_server_${EXP}_${SLURM_JOB_ID:-manual}.log" 2>&1 &
GEN_PID=$!

start=$SECONDS
until curl -sf "localhost:$GEN_PORT/healthz" > /dev/null; do
    kill -0 "$GEN_PID" 2>/dev/null || { echo "FATAL: gen server died during startup, see its log" >&2; exit 1; }
    (( SECONDS - start > 600 )) && { echo "FATAL: gen server not healthy after 600s" >&2; exit 1; }
    sleep 5
done
echo "gen server healthy on :$GEN_PORT (pid $GEN_PID)"

# --- 2. trainer (4x B200: FSDP2 actor + colocated async vLLM rollout) --------
# run_v11_gpt56_sft_val.sh auto-resolves the SFT init through the bind and
# carries every v13 fix (top_p/top_k truncation, OOM micro-batch, rollout dumps).
# trainer.resume_mode=auto makes the 24h chain restarts seamless.
env VAL_ONLY=0 EXP="$EXP" LOG="$MSR/logs_healthbench_rubric/${EXP}_train.log" \
    "${RUN_GPU[@]}" bash "$MSR/verl_healthbench/scripts/self_evolving/train/run_v11_gpt56_sft_val.sh"
rc=$?

echo "trainer exited rc=$rc"
kill "$GEN_PID" 2>/dev/null
exit $rc
