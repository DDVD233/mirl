#!/usr/bin/env bash
# Diagnostic runs isolating the v13-v16 healthbench val decline (dvd, 2026-08-02).
#
#   trainval : train DIRECTLY on the full HealthBench-Pro val set with its REAL
#              rubrics. No generation, no evolution, no retrieval loop — plain
#              single-turn RLHFDataset; train judge == val judge (gpt-chat-latest);
#              repetition penalty OFF. Deliberate train-on-test probe: if val
#              cannot climb when training ON the test set, the RL pipeline
#              itself (GRPO/reward/judge/rollout) is broken and data was never
#              the issue. Expect clear improvement within ~20-40 steps.
#   seeded   : IDENTICAL minimal single-turn algorithm, but tasks come from the
#              gen server style-seeded 100% from a 10% sample of REAL val tasks
#              (SEED_EXEMPLARS_PATH), /evolve off. Single-variable diff vs
#              trainval = the task source. If trainval climbs and this doesn't,
#              task GENERATION is the problem; if both climb, the original
#              self-generated distribution/curriculum was.
#
# Selected by $1 or the chain/DIAG_MODE file (see run_healthbench_evolve.sh hook).
set -uo pipefail

MODE="${1:-${HB_DIAG_MODE:-trainval}}"
WORK=/work/mit/ppliang_mit/dvdai
S=/scratch/dvdai_mit/self_evolving
SIF=$S/sif/verl-selfevolving-cu130-vllm0.22.1.sif
MSR=/scratch/sheng/self_evolving

export WANDB_API_KEY="$(cat "$S/.wandb_key")"
export WANDB_RESUME=allow
# Run-id policy: RESUME the same wandb run only when there is a checkpoint to
# resume from; otherwise mint a NEW id. A fixed id + a fresh step-0 start makes
# wandb reject every metric ("Tried to log to step 0 that is less than the
# current step") and leave the run displaying the previous crashed session.
new_or_resumed_run_id() {   # $1 = EXP
    local exp="$1" state="$S/chain/WANDB_RUNID.$1"
    local ckpt="$S/checkpoints/healthbench_rubric/$1"
    if [[ -f "$ckpt/latest_checkpointed_iteration.txt" && -s "$state" ]]; then
        cat "$state"                      # real resume: keep the curve going
    else
        local rid="${exp}_$(date +%m%d_%H%M)"
        printf '%s' "$rid" > "$state"
        printf '%s' "$rid"
    fi
}

BINDS=(--bind "$S:$MSR" --bind "$WORK/mirl:$MSR/verl_healthbench")
RUN_GPU=(apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF")
RUN_CPU=(apptainer exec --writable-tmpfs "${BINDS[@]}" "$SIF")

# All-GPT roles (dvd 2026-08-03): no local Qwen teacher dependency for the
# healthbench line. Generation + fallback judge run on TRAPI deployments
# (fallback on a DIFFERENT deployment than the primary judge so per-deployment
# flips don't take both out). The qwen-teacher chain now serves p1-mimic only.
TRAPI_BASE="http://point.dd.works:18890/v1"
TRAPI_KEY_="$(cat "$S/.trapi_key")"
probe_trapi() {
    curl -s -m 20 -H "Authorization: Bearer $TRAPI_KEY_" -H "Content-Type: application/json" \
        -d "{\"model\":\"gpt-5.4_2026-03-05\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_completion_tokens\":4}" \
        "$TRAPI_BASE/chat/completions" | grep -q '"choices"'
}
tstart=$SECONDS
until probe_trapi; do
    (( SECONDS - tstart > 3600 )) && { echo "FATAL: TRAPI gpt-5.4 unreachable for 1h" >&2; exit 1; }
    sleep 30
done
echo "TRAPI gpt-5.4 healthy"
export CHAT_PROVIDER=trapi
export API_BASE="$TRAPI_BASE"
export API_KEY="$(cat "$S/.trapi_key")"
export MODEL_NAME="gpt-5.4_2026-03-05"
export FALLBACK_JUDGE_BASE="$TRAPI_BASE"
export FALLBACK_JUDGE_MODEL="gpt-5.4_2026-03-05"
export FALLBACK_JUDGE_PROVIDER="trapi"
export FALLBACK_JUDGE_KEY="$TRAPI_KEY_"

INIT_HOST=$S/checkpoints/self_evolving_medical/gpt56_sft_qwen36_27b/global_step_28/hf_merged
[[ -f "$INIT_HOST/config.json" ]] || { echo "FATAL: SFT init missing" >&2; exit 1; }
INIT=$MSR/checkpoints/self_evolving_medical/gpt56_sft_qwen36_27b/global_step_28/hf_merged   # container-side path

# Judge on BOTH sides = the benchmark judge; no extra reward shaping beyond the
# longstanding think/floor terms; new v16 repetition penalty disabled.
export TRAIN_JUDGE_BASE="http://point.dd.works:18890/v1"
export TRAIN_JUDGE_MODEL="gpt-chat-latest_2026-05-28"
export TRAIN_JUDGE_PROVIDER="trapi"
export TRAIN_JUDGE_KEY="$(cat "$S/.trapi_key")"
export VAL_JUDGE_MODEL="gpt-chat-latest_2026-05-28"
export HB_REP_PENALTY_MAX=0
export ENTROPY_COEFF=0.0
export EVOLVE_GENERATION=False
# Micro-batch token budget: must EXCEED the longest sequence (prompt+response)
# or rearrange_micro_batches asserts, but activation memory scales with it —
# 32768 OOM'd in update_actor (needed 14.5 GiB with 4.2 free). Max single-turn
# sequence is 8192 prompt + 12288 response = 20480, so 20736 is the smallest
# safe value; vLLM's share drops to 0.40 to fund the actor's activations.
# Validation decodes like TRAINING (temp 1.0) by default: greedy val while
# rollouts sample at temp 1.0 measures a different distribution than the one
# being optimized. Set VAL_DO_SAMPLE=False VAL_TEMP=0 for the greedy,
# benchmark-comparable protocol.
export VAL_TEMP="${VAL_TEMP:-1.0}"
export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
export PPO_MAX_TOKEN_LEN=20736
export LOGPROB_MAX_TOKEN_LEN=20736
export VLLM_GPU_UTIL=0.40
# LR (dvd 2026-08-04): 2e-7 was tuned when advantages were std-normalized to
# +-2.47; Dr.GRPO (norm_adv_by_std=False) leaves them at +-0.85, a ~3x smaller
# update, and 10 steps produced no measurable movement in either direction.
# 1e-6 restores roughly the historical effective step (and matches the v3-era
# value that ran stably) while keeping Dr.GRPO so judge noise isn't amplified.
export LR="${LR:-1e-6}"

case "$MODE" in
  trainval)
    export EXP="healthbench_diag_trainval"
    export WANDB_RUN_ID="$(new_or_resumed_run_id "$EXP")"
    export ACTOR_MODEL_PATH="$INIT"
    LOGF=$MSR/logs_healthbench_rubric/${EXP}_train.log
    # base recipe called DIRECTLY (no v7_retrieval wrapper): single-turn, no tools.
    # Minimal-fixed algorithm per the 2026-08-02 invariant audit: full-policy
    # sampling (the v11 top_k=20/top_p=0.95 truncation is uncorrected off-policy
    # bias, measured ~5% per-token prob gap -> tail leak -> entropy rise ->
    # collapse in all four runs), Dr.GRPO (std-normalization amplified binary
    # judge noise to +-2.5 advantages), symmetric clip, no shaping terms.
    export HB_SCORE_MIN=0.0
    export HB_THINK_PENALTY_PER_1K=0.0
    export VERL_THINK_BUDGET_TOKENS=0
    env EXP="$EXP" \
        "${RUN_GPU[@]}" bash -c "cd $MSR/verl_healthbench && \
          bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric.sh \
            data.custom_cls.path=null data.custom_cls.name=null \
            data.shuffle=True \
            algorithm.norm_adv_by_std_in_grpo=False \
            actor_rollout_ref.actor.clip_ratio_high=0.2 \
            data.max_response_length=12288 \
            actor_rollout_ref.rollout.max_model_len=24576 \
            data.val_files=$MSR/healthbench_pro_val.parquet \
            ++data.val_max_samples=-1 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
            actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMP \
            trainer.total_epochs=100 \
            trainer.total_training_steps=300 \
            +trainer.rollout_data_dir=$MSR/logs_healthbench_rubric/rollouts/\$EXP" \
        2>&1 | tee "$S/logs_healthbench_rubric/${EXP}_train.log"
    exit "${PIPESTATUS[0]}"
    ;;

  seeded)
    export EXP="healthbench_diag_seeded"
    export WANDB_RUN_ID="$(new_or_resumed_run_id "$EXP")"
    export ACTOR_MODEL_PATH="$INIT"
    GEN_PORT=8017
    export GEN_SERVER_URL="http://localhost:$GEN_PORT"

    # One-time: exemplar pool from a fixed 10% sample of the real val set.
    EX=$S/seed_corpora/val10_exemplars.jsonl
    if [[ ! -f "$EX" ]]; then
        "${RUN_CPU[@]}" python3 - "$MSR/healthbench_pro_val.parquet" "$MSR/seed_corpora/val10_exemplars.jsonl" <<'PY'
import sys, json, pandas as pd
df = pd.read_parquet(sys.argv[1]).sample(frac=0.10, random_state=0)
with open(sys.argv[2], "w") as f:
    for _, r in df.iterrows():
        msgs = list(r["prompt"])
        user_texts = [m["content"] for m in msgs if m.get("role") == "user"]
        text = user_texts[-1] if user_texts else ""
        ri = r["extra_info"].get("rubric_items")
        items = list(ri) if ri is not None and len(ri) else []
        crits = [it.get("criterion") or it.get("criterion_text") or "" for it in items]
        f.write(json.dumps({
            "text": text, "source": "healthbench_pro_val_10pct", "kind": "val_seed",
            "chars": len(text), "starts_lower": text[:1].islower(),
            "has_question_mark": "?" in text,
            "criterion_exemplars": [c for c in crits if c][:6],
        }) + "\n")
print("wrote", sys.argv[2])
PY
        echo "built val10 exemplar pool: $(wc -l < "$EX") exemplars"
    fi

    LOG_DIR=$S/logs_healthbench_rubric/$EXP
    PROMPT_DIR=$LOG_DIR/prompts
    if [[ ! -d "$PROMPT_DIR" ]]; then
        mkdir -p "$LOG_DIR"
        cp -r "$S/logs_healthbench_rubric/v16_prompt_seed/prompts" "$PROMPT_DIR"
    fi

    if curl -sf "localhost:$GEN_PORT/healthz" > /dev/null 2>&1; then
        echo "FATAL: port $GEN_PORT already serves a gen server" >&2; exit 1
    fi
    env EXP="$EXP" GEN_SERVER_PORT="$GEN_PORT" \
        SEED_EXEMPLARS_PATH="$MSR/seed_corpora/val10_exemplars.jsonl" \
        HB_STYLE_SEED_SHARE=1.0 \
        "${RUN_CPU[@]}" bash "$MSR/verl_healthbench/scripts/self_evolving/serve/start_gen_server_rubric_s4.sh" \
        > "$S/logs/gen_server_${EXP}_${SLURM_JOB_ID:-manual}.log" 2>&1 &
    GEN_PID=$!
    start=$SECONDS
    until curl -sf "localhost:$GEN_PORT/healthz" > /dev/null; do
        kill -0 "$GEN_PID" 2>/dev/null || { echo "FATAL: gen server died at startup" >&2; exit 1; }
        (( SECONDS - start > 600 )) && { echo "FATAL: gen server unhealthy after 600s" >&2; exit 1; }
        sleep 5
    done
    echo "gen server healthy on :$GEN_PORT"

    # Same minimal-fixed single-turn algorithm as trainval; ONLY the task source
    # differs (SelfEvolvingDataset fetching from the seeded gen server — the
    # base script wires custom_cls + gen_server_url by default).
    export HB_SCORE_MIN=0.0
    export HB_THINK_PENALTY_PER_1K=0.0
    export VERL_THINK_BUDGET_TOKENS=0
    env EXP="$EXP" \
        "${RUN_GPU[@]}" bash -c "cd $MSR/verl_healthbench && \
          bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric.sh \
            algorithm.norm_adv_by_std_in_grpo=False \
            actor_rollout_ref.actor.clip_ratio_high=0.2 \
            data.max_response_length=12288 \
            actor_rollout_ref.rollout.max_model_len=24576 \
            data.val_files=$MSR/healthbench_pro_val.parquet \
            ++data.val_max_samples=-1 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
            actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMP \
            trainer.total_epochs=100 \
            trainer.total_training_steps=300 \
            +trainer.rollout_data_dir=$MSR/logs_healthbench_rubric/rollouts/\$EXP" \
        2>&1 | tee "$S/logs_healthbench_rubric/${EXP}_train.log"
    rc="${PIPESTATUS[0]}"
    kill "$GEN_PID" 2>/dev/null
    exit $rc
    ;;

  *) echo "usage: $0 {trainval|seeded}" >&2; exit 2;;
esac
