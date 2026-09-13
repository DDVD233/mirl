#!/usr/bin/env bash
# Retrieval on/off evaluation queue for the stage-2 paper (2026-09-13).
#
# One ordered list of val-only evaluations that do not already exist as in-loop
# dumps. Every box that runs this script works through the same list: an item is
# claimed with an atomic lock directory, skipped when its dump is complete, and the
# next unclaimed item follows -- so a 2-GPU box and a 4-GPU pod can share the list and
# a reclaimed pod's item is picked up by whoever comes next (remove its stale lock).
# Each item lands in $S/logs_hb9b/val_generations/<exp>/0.jsonl for regrading with
# scripts/self_evolving/analysis/regrade_hbpro_dumps.py.
#
#   NGPU=<gpus on this box> bash scripts/self_evolving/eval/onoff_queue.sh
#
# Tools-off items use all NGPU GPUs (27B: TP=2, so NGPU must be even). Tools-on
# items keep the last GPU for the frozen-9B summarizer and run the 9B policy on the
# rest with TP=1.
set -uo pipefail
S=/scratch/sheng/self_evolving
CK=$S/checkpoints/hf_merged
W=$S/verl_specgap/scripts/self_evolving/eval/run_hbpro_valonly.sh
LOG=$S/paper_refresh/onoff_logs; mkdir -p "$LOG"
NGPU="${NGPU:-4}"
POL=$(( NGPU - 1 ))            # policy GPUs when the summarizer takes one
FRZ=$(( NGPU - 1 ))            # the summarizer's GPU

run() {  # run <exp> <model> <retrieval> <web> <n_gpus> <expected rows> [ENV=VAL ...]
    local exp=$1 model=$2 retr=$3 web=$4 ngpus=$5 want=$6; shift 6
    local dump=$S/logs_hb9b/val_generations/$exp/0.jsonl
    if [ -f "$dump" ] && [ "$(wc -l < "$dump")" -ge "$want" ]; then
        echo "=== skip $exp (dump complete)"; return 0
    fi
    if ! mkdir "$LOG/$exp.lock" 2>/dev/null; then
        echo "=== skip $exp (claimed by $(cat "$LOG/$exp.lock/owner" 2>/dev/null))"; return 0
    fi
    echo "$(hostname) $(date -u +%FT%TZ)" > "$LOG/$exp.lock/owner"
    echo "=== $(date -u +%FT%TZ) START $exp on $(hostname) ngpus=$ngpus"
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

MEDX_ENV=(SE_DOMAIN=medxpert VAL_PARQUET=$S/medxpertqa_text_val.parquet,$S/medxpertqa_mm_val.parquet
          MAX_PROMPT_LEN=16384 MAX_RESP_LEN=12288 ROLLOUT_MAX_LEN=28672 HB_VAL_LENGTH_PENALTY_PER_500=0
          HB_KB_ANCHOR_SHARE=0.10 HB_STYLE_SEED_SHARE=0)

# 27B block B without tools (the with-tools numbers are the in-loop dumps at the same steps).
run hbpro_27b_ser200_notools   "$CK/hb27b_ser_step200"    0 0 "$NGPU" 525 ROLLOUT_TP=2
run hbpro_27b_fixed60_notools  "$CK/hb27b_fixed_step60"   0 0 "$NGPU" 525 ROLLOUT_TP=2
run hbpro_27b_base_notools     Qwen/Qwen3.6-27B           0 0 "$NGPU" 525 ROLLOUT_TP=2
# 9B block D (trained without tools) answering WITH the tool loop.
run hbpro_9bD_ser480_tools    "$CK/hb9b_noretr_ser_step480"   1 1 "$POL" 525 ROLLOUT_TP=1 FROZEN_GPU="$FRZ"
run hbpro_9bD_fixed860_tools  "$CK/hb9b_noretr_fixed_step860" 1 1 "$POL" 525 ROLLOUT_TP=1 FROZEN_GPU="$FRZ"
# Held-out arms, the surviving checkpoint, WITHOUT its tool (web search / retrieval).
run prbench_9b_ship420_notools "$CK/prbench9b_ship_step420" 0 0 "$NGPU" 550 ROLLOUT_TP=1 \
    WEB_ONLY=0 SE_DOMAIN=prbench VAL_PARQUET=$S/prbench_hard_val.parquet \
    MAX_PROMPT_LEN=16384 ROLLOUT_MAX_LEN=24576 HB_VAL_LENGTH_PENALTY_PER_500=0 \
    HB_LENGTH_CENTER=4000 HB_KB_ANCHOR_SHARE=0 HB_STYLE_SEED_SHARE=0
run profbench_9b_ship120_notools "$CK/profbench9b_ship_step120" 0 0 "$NGPU" 40 ROLLOUT_TP=1 \
    WEB_ONLY=0 SE_DOMAIN=profbench VAL_PARQUET=$S/profbench_val.parquet \
    HB_VAL_LENGTH_PENALTY_PER_500=0 HB_LENGTH_CENTER=4000 HB_KB_ANCHOR_SHARE=0 HB_STYLE_SEED_SHARE=0
run medxpert_9b_ship180_notools "$CK/medxpert9b_ship_step180" 0 0 "$NGPU" 4450 ROLLOUT_TP=1 "${MEDX_ENV[@]}"
# MedXpertQA step 180 was never validated in-loop (the run died at 178), so its
# with-tools number needs a run too.
run medxpert_9b_ship180_tools "$CK/medxpert9b_ship_step180" 1 1 "$POL" 4450 ROLLOUT_TP=1 FROZEN_GPU="$FRZ" \
    "${MEDX_ENV[@]}" SEARCH_SNAPSHOT=$S/kb/search_cache_valonly_medx.sqlite
echo "QUEUE_DONE on $(hostname) $(date -u +%FT%TZ)"
