#!/usr/bin/env bash
# Retrieval on/off evaluation queue for the stage-2 paper (2026-09-13).
#
# Runs, in order, the val-only evaluations that do not already exist as in-loop
# dumps, one after another on the pod this is started on. Every item is
#   <exp name> | <checkpoint> | RETRIEVAL WEB_SEARCH_TOOL | extra env
# and lands in $S/logs_hb9b/val_generations/<exp>/0.jsonl for regrading with
# scripts/self_evolving/analysis/regrade_hbpro_dumps.py.
#
# QUEUE=A  27B, no tools (needs 4 GPUs): the surviving block-B checkpoints and base.
# QUEUE=B  9B: block-D checkpoints WITH tools (2 policy GPUs + summarizer on GPU 3),
#          then the held-out checkpoints without tools, then MedXpertQA with tools.
# Items already finished (dump present with 525 / expected rows) are skipped, so the
# queue can be restarted after a pod reclaim.
#
#   QUEUE=A bash scripts/self_evolving/eval/onoff_queue.sh
set -uo pipefail
S=/scratch/sheng/self_evolving
CK=$S/checkpoints/hf_merged
W=$S/verl_specgap/scripts/self_evolving/eval/run_hbpro_valonly.sh
LOG=$S/paper_refresh/onoff_logs; mkdir -p "$LOG"
QUEUE="${QUEUE:?set QUEUE=A|B}"

run() {  # run <exp> <model> <retrieval> <web> <n_gpus> <expected rows> [ENV=VAL ...]
    local exp=$1 model=$2 retr=$3 web=$4 ngpus=$5 want=$6; shift 6
    local dump=$S/logs_hb9b/val_generations/$exp/0.jsonl
    if [ -f "$dump" ] && [ "$(wc -l < "$dump")" -ge "$want" ]; then
        echo "=== skip $exp (dump complete)"; return 0
    fi
    echo "=== $(date -u +%FT%TZ) START $exp"
    env "$@" MODEL="$model" EXP="$exp" RETRIEVAL="$retr" WEB_SEARCH_TOOL="$web" N_GPUS="$ngpus" \
        bash "$W" > "$LOG/$exp.log" 2>&1
    local rc=$?
    # The launcher's trap kills its own children; make sure nothing survives between items.
    for pat in '^VLLM::EngineCore' '^ray::' '^/usr/local/bin/python scripts/self_evolving/generation_server.py' \
               '^/usr/local/bin/python scripts/self_evolving/kb/serper_cache_server.py' '^/usr/bin/python3 /usr/local/bin/vllm serve'; do
        pkill -f "$pat" 2>/dev/null || true
    done
    sleep 20
    if [ -f "$dump" ]; then
        echo "=== $(date -u +%FT%TZ) DONE $exp rc=$rc rows=$(wc -l < "$dump")"
    else
        echo "=== $(date -u +%FT%TZ) FAILED $exp rc=$rc (no dump); tail:"; tail -5 "$LOG/$exp.log"
    fi
}

case "$QUEUE" in
A)
    run hbpro_27b_ser200_notools   "$CK/hb27b_ser_step200"    0 0 4 525
    run hbpro_27b_fixed60_notools  "$CK/hb27b_fixed_step60"   0 0 4 525
    run hbpro_27b_base_notools     Qwen/Qwen3.6-27B           0 0 4 525
    ;;
B)
    # Block D (trained without tools) answering WITH the tool loop.
    run hbpro_9bD_ser480_tools    "$CK/hb9b_noretr_ser_step480"   1 1 2 525 ROLLOUT_TP=2 FROZEN_GPU=3
    run hbpro_9bD_fixed860_tools  "$CK/hb9b_noretr_fixed_step860" 1 1 2 525 ROLLOUT_TP=2 FROZEN_GPU=3
    # Held-out arms, the surviving checkpoint, WITHOUT its tool (web search / retrieval).
    run prbench_9b_ship420_notools "$CK/prbench9b_ship_step420" 0 0 4 550 \
        WEB_ONLY=0 SE_DOMAIN=prbench VAL_PARQUET=$S/prbench_hard_val.parquet \
        MAX_PROMPT_LEN=16384 ROLLOUT_MAX_LEN=24576 HB_VAL_LENGTH_PENALTY_PER_500=0 \
        HB_LENGTH_CENTER=4000 HB_KB_ANCHOR_SHARE=0 HB_STYLE_SEED_SHARE=0
    run profbench_9b_ship120_notools "$CK/profbench9b_ship_step120" 0 0 4 40 \
        WEB_ONLY=0 SE_DOMAIN=profbench VAL_PARQUET=$S/profbench_val.parquet \
        HB_VAL_LENGTH_PENALTY_PER_500=0 HB_LENGTH_CENTER=4000 HB_KB_ANCHOR_SHARE=0 HB_STYLE_SEED_SHARE=0
    run medxpert_9b_ship180_notools "$CK/medxpert9b_ship_step180" 0 0 4 4450 \
        SE_DOMAIN=medxpert VAL_PARQUET=$S/medxpertqa_text_val.parquet,$S/medxpertqa_mm_val.parquet \
        MAX_PROMPT_LEN=16384 MAX_RESP_LEN=12288 ROLLOUT_MAX_LEN=28672 HB_VAL_LENGTH_PENALTY_PER_500=0 \
        HB_KB_ANCHOR_SHARE=0.10 HB_STYLE_SEED_SHARE=0
    # MedXpertQA step 180 was never validated in-loop (the run died at 178), so its
    # with-tools number needs a run too.
    run medxpert_9b_ship180_tools "$CK/medxpert9b_ship_step180" 1 1 2 4450 ROLLOUT_TP=2 FROZEN_GPU=3 \
        SE_DOMAIN=medxpert VAL_PARQUET=$S/medxpertqa_text_val.parquet,$S/medxpertqa_mm_val.parquet \
        MAX_PROMPT_LEN=16384 MAX_RESP_LEN=12288 ROLLOUT_MAX_LEN=28672 HB_VAL_LENGTH_PENALTY_PER_500=0 \
        HB_KB_ANCHOR_SHARE=0.10 HB_STYLE_SEED_SHARE=0 SEARCH_SNAPSHOT=$S/kb/search_cache_valonly_medx.sqlite
    ;;
*) echo "unknown QUEUE=$QUEUE" >&2; exit 1 ;;
esac
echo "QUEUE_${QUEUE}_DONE $(date -u +%FT%TZ)"
