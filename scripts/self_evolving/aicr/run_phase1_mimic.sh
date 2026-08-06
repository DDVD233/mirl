#!/usr/bin/env bash
# AICR launcher for the PHASE-1 validation run: self-evolving DATA generation
# with a FIXED reward (no reward/rubric evolution), mimic-rare only.
#
#   - code: /work/.../mirl-phase1 @ de576393 (the pre-rubric commit that created
#     the two-stage SFT->RL pipeline, same era as the original 27B gen-RL run),
#     bound to /root/mirl_evolve = that launcher's default REPO.
#   - recipe: scripts/self_evolving/train/run_qwen36_27b_selfimprove_from_sft.sh
#     (self-evolving questions from the gen server, EVOLVE_ENABLE=False, fixed
#     composite reward, actor from the merged distill-SFT init).
#   - judge (train reward + val): gpt-chat-latest_2026-05-28 via the TRAPI proxy
#     (overridden with hydra ++ args). The gen-server proposer/generator/
#     validator uses the SAME GPT deployment (CHAT_PROVIDER=trapi) — no local
#     Qwen anywhere in this run.
#   - init: ddvd233/mimiciv_rare_qwen36_27b_sft_distill (HF backup of the
#     deleted mimiciv_rare_qwen36_27b_sft_distill/global_step_90 merge).
#   - KL fully off (per dvd 2026-08-01, default for future runs): kl_ctrl 0,
#     use_kl_loss already False, and loss_mode kl_cov -> vanilla clip (kl_cov
#     IS a KL penalty; vanilla keeps stability via standard clipping instead).
#   - Val integrity (dvd 2026-08-03): the held-out test.jsonl is used ONLY as
#     data.val_files. The gen server is started with TEST_SEEDS_PATH="" and
#     GEN_TEST_TARGET=0 so no test item can seed a generated training question
#     (verl already forces RLHFDataset for val, so generated rows can never
#     enter validation from the other direction).
#   - Multimodal (dvd 2026-08-03): ECG + chest-X-ray paths were remapped to
#     AICR-local copies (they pointed at the old cluster and silently rendered
#     as black placeholders); train 6604/6604 and test 3219/3219 refs resolve.
#   - "infinite" RL: total_training_steps/epochs raised past any horizon; the
#     train_chain.sbatch 24h chain + trainer.resume_mode=auto keep it going
#     until a STOP file ends it. save_freq=20 with max_actor_ckpt_to_keep=2
#     bounds disk; checkpoints land on /scratch via the compat bind.
#
# Launch:
#   RUN_SCRIPT=$WORK/mirl/scripts/self_evolving/aicr/run_phase1_mimic.sh FRP_PORT=2337 \
#     sbatch --job-name=p1-mimic -c 48 --mem=800G \
#     $WORK/mirl/scripts/self_evolving/aicr/train_chain.sbatch
set -uo pipefail

WORK=/work/mit/ppliang_mit/dvdai
S=/scratch/dvdai_mit/self_evolving
SIF=$S/sif/verl-selfevolving-cu130-vllm0.22.1.sif
MSR=/scratch/sheng/self_evolving
PHASE1_REPO=/root/mirl_evolve            # container-side view of mirl-phase1

# _v2 = the 2026-08-03 restart after the judge fix. Steps 1-29 of the original
# run trained against dead judges (every TRAPI call 400d, so judge_acc_*=0.0,
# answer_quality=1.0, reasoning=3.0 constant); its checkpoints are deleted and
# its wandb history is kept under the un-suffixed name as the "before" record.
export EXP="${EXP:-mimiciv_rare_qwen36_27b_phase1_aicr_v2}"
export WANDB_API_KEY="$(cat "$S/.wandb_key")"   # dvd's key — the shared ~/.wandb_env belongs to weianxie
# Keep ONE wandb run across the 24h chain links: a fixed run id + resume=allow
# makes each restarted trainer append to the same run instead of minting a new
# one per day. verl resumes at the checkpoint step, so the x-axis stays clean.
export WANDB_RESUME=allow
export WANDB_RUN_ID="${WANDB_RUN_ID:-$EXP}"

BINDS=(--bind "$S:$MSR" --bind "$WORK/mirl-phase1:$PHASE1_REPO")
RUN_GPU=(apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF")
RUN_CPU=(apptainer exec --writable-tmpfs "${BINDS[@]}" "$SIF")

# --- teacher = GPT via TRAPI (2026-08-03, per dvd: this run is entirely GPT).
# The gen-server proposer/generator/validator used to hit the local Qwen
# (CHAT_PROVIDER=vllm); it now uses the same TRAPI gpt-chat-latest deployment as
# the reward judge, so the run no longer depends on the qwen-teacher chain job.
# The proxy exposes no /models route (404) — gate on a real chat completion.
TEACHER_BASE=http://point.dd.works:18890/v1
GPT_MODEL="${GPT_MODEL:-gpt-chat-latest_2026-05-28}"
TRAPI_KEY=$(cat "$S/.trapi_key")
tstart=$SECONDS
until [[ "$(curl -s -m 20 -o /dev/null -w '%{http_code}' \
        -X POST "$TEACHER_BASE/chat/completions" \
        -H 'Content-Type: application/json' -H "Authorization: Bearer $TRAPI_KEY" \
        -d "{\"model\":\"$GPT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}")" == "200" ]]; do
    (( SECONDS - tstart > 21600 )) && { echo "FATAL: TRAPI $GPT_MODEL unreachable for 6h" >&2; exit 1; }
    sleep 30
done
export TEACHER_BASE
echo "teacher healthy: $GPT_MODEL at $TEACHER_BASE"

INIT=$MSR/checkpoints/self_evolving_medical/mimiciv_rare_qwen36_27b_sft_distill_hf   # container path
INIT_HOST=$S/checkpoints/self_evolving_medical/mimiciv_rare_qwen36_27b_sft_distill_hf
if [[ ! -f "$INIT_HOST/config.json" ]]; then
    echo "FATAL: phase-1 SFT init not at $INIT_HOST (hf download still running?)" >&2
    exit 1
fi
for f in "$S/mimiciv_rare/test.jsonl" "$S/.climb_teacher_key" "$S/.trapi_key"; do
    [[ -e "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done

GEN_PORT=8007   # hb-evolve owns 8006; keep clear in case both land on one node
LOG_DIR_HOST=$S/logs_selfimprove_from_sft
mkdir -p "$LOG_DIR_HOST"

# --- 1. gen server (question generation, old-style, :8007, CPU-only) ---------
# Teacher/proposer = GPT via TRAPI; KB grounding = mib (embeddings + Milvus).
env DATA_DIR="$MSR/mimiciv_rare" \
    PYTHON_BIN=/usr/local/bin/python \
    API_BASE="$TEACHER_BASE" \
    API_KEY_FILE="$MSR/.trapi_key" \
    MODEL_NAME="$GPT_MODEL" \
    CHAT_PROVIDER=trapi \
    MILVUS_COLLECTION=medical_knowledge_v2 \
    TEST_SEEDS_PATH="" GEN_TEST_TARGET=0 \
    GEN_SERVER_PORT=$GEN_PORT \
    LOG_DIR="$MSR/logs_selfimprove_from_sft" \
    "${RUN_CPU[@]}" bash -c 'export API_KEY=$(cat "$API_KEY_FILE"); exec bash /root/mirl_evolve/scripts/self_evolving/serve/start_generation_server.sh' \
    > "$S/logs/gen_server_${EXP}_${SLURM_JOB_ID:-manual}.log" 2>&1 &
GEN_PID=$!

start=$SECONDS
until curl -sf "localhost:$GEN_PORT/healthz" > /dev/null; do
    kill -0 "$GEN_PID" 2>/dev/null || { echo "FATAL: gen server died during startup, see its log" >&2; exit 1; }
    (( SECONDS - start > 600 )) && { echo "FATAL: gen server not healthy after 600s" >&2; exit 1; }
    sleep 5
done
echo "gen server healthy on :$GEN_PORT (pid $GEN_PID)"

# --- 2. trainer: phase-1 recipe + external judge + infinite horizon ----------
env REPO="$PHASE1_REPO" \
    ACTOR_MODEL_PATH="$INIT" \
    EXP="$EXP" \
    GEN_SERVER_URL="http://localhost:$GEN_PORT" \
    REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-48}" \
    "${RUN_GPU[@]}" bash "$PHASE1_REPO/scripts/self_evolving/train/run_qwen36_27b_selfimprove_from_sft.sh" \
        ++reward.custom_reward_function.reward_kwargs.api_base=http://point.dd.works:18890/v1 \
        ++reward.custom_reward_function.reward_kwargs.api_key="$TRAPI_KEY" \
        ++reward.custom_reward_function.reward_kwargs.model_name=gpt-chat-latest_2026-05-28 \
        ++reward.custom_reward_function.reward_kwargs.provider=trapi \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
        trainer.total_epochs=1000 \
        trainer.total_training_steps=1000000 \
    2>&1 | tee -a "$LOG_DIR_HOST/${EXP}_train.log"
rc=${PIPESTATUS[0]}

echo "trainer exited rc=$rc"
kill "$GEN_PID" 2>/dev/null
exit $rc
