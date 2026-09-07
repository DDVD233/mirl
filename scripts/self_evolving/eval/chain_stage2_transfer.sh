#!/usr/bin/env bash
# Stage-2 transfer evals, run as two chains on one 4-GPU pod (2026-09-07 plan, section B4).
#
#   bash chain_stage2_transfer.sh valonly   # GPUs 0,1: HealthBench Hard through the in-loop
#                                          # validator (tools on for the 27B setting, off for
#                                          # the no-retrieval 9B setting); also does the merges
#   bash chain_stage2_transfer.sh vllm      # GPUs 2,3: MedXpertQA text via vLLM
#
# Checkpoints rsynced from AICR land under $S/checkpoints/hb9b_aicr/...; the AICR-side
# watcher (chain_stage2_markers.sh) touches <ckpt>/RSYNC_DONE when a transfer completed
# with rc=0. Merged HF weights go to $S/checkpoints/hf_merged/<tag>, with MERGE_DONE.
# Everything is idempotent: a finished eval (dump / result file present) is skipped.
set -u
S=/scratch/sheng/self_evolving
REPO=$S/verl_specgap
A=$S/checkpoints/hb9b_aicr
M=$S/checkpoints/hf_merged
LOG=$S/logs_hb9b
mkdir -p "$M" "$S/eval_out" "$S/logs/medxpertqa_ckpt"
cd "$REPO"
export HF_HOME=$S/hf_cache

wait_file () { until [ -e "$1" ]; do sleep 120; done; }
wait_no_proc () { while pgrep -f "$1" > /dev/null; do sleep 120; done; }

merge () {  # merge <tag> <fsdp actor dir>
  local tag=$1 actor=$2
  [ -e "$M/$tag/MERGE_DONE" ] && return 0
  echo "=== $(date -u +%FT%TZ) merge $tag from $actor"
  /usr/local/bin/python -m verl.model_merger merge --backend fsdp --local_dir "$actor" \
      --target_dir "$M/$tag" > "$S/paper_refresh/merge_${tag}.log" 2>&1 \
      && [ -f "$M/$tag/model.safetensors.index.json" ] && touch "$M/$tag/MERGE_DONE"
  echo "=== $(date -u +%FT%TZ) merge $tag rc=$? "
}

valonly () {  # valonly <exp> <model> <retrieval 0|1>
  local exp=$1 model=$2 retr=$3
  [ -f "$LOG/val_generations/$exp/0.jsonl" ] && { echo "skip $exp (dump exists)"; return 0; }
  echo "=== $(date -u +%FT%TZ) val-only $exp"
  MODEL="$model" EXP="$exp" N_GPUS=2 RETRIEVAL="$retr" WEB_SEARCH_TOOL="$retr" \
      bash scripts/self_evolving/eval/run_hbhard_valonly.sh > "$LOG/hbhard_${exp}_launch.log" 2>&1
  echo "=== $(date -u +%FT%TZ) val-only $exp rc=$? dump=$(wc -l < "$LOG/val_generations/$exp/0.jsonl" 2>/dev/null)"
  pkill -f "generation_server.py --host 0.0.0.0 --port 8061" 2>/dev/null; sleep 20
}

CK9_SER=$A/hb9b/hb9b_specgap_ship_noretrieval_sj_aicr/global_step_480
CK9_FIX=$A/hb9b/hb9b_specgap_simple_noretrieval_sj_aicr/global_step_860
CK27_FIX=$A/hb9b/hb27b_specgap_simple_retrieval_aicr/global_step_60

case "${1:?valonly|vllm}" in
  valonly)
    wait_no_proc "main_ppo.*hb27b_ser200_hbhard"
    valonly base27b_hbhard Qwen/Qwen3.6-27B 1
    wait_file "$CK9_SER/RSYNC_DONE";  merge hb9b_noretr_ser_step480 "$CK9_SER/actor"
    valonly hb9b_noretr_ser480_hbhard "$M/hb9b_noretr_ser_step480" 0
    valonly base9b_noretr_hbhard Qwen/Qwen3.5-9B 0
    wait_file "$CK27_FIX/RSYNC_DONE"; merge hb27b_fixed_step60 "$CK27_FIX/actor"
    valonly hb27b_fixed60_hbhard "$M/hb27b_fixed_step60" 1
    wait_file "$CK9_FIX/RSYNC_DONE";  merge hb9b_noretr_fixed_step860 "$CK9_FIX/actor"
    valonly hb9b_noretr_fixed860_hbhard "$M/hb9b_noretr_fixed_step860" 0
    echo "=== $(date -u +%FT%TZ) VALONLY CHAIN DONE" ;;
  vllm)
    wait_no_proc "run_medxpertqa_ckpt.sh hb27b_ser_step200"
    wait_file "$M/hb9b_noretr_ser_step480/MERGE_DONE"
    GPUS=2,3 TP=2 bash scripts/self_evolving/eval/run_medxpertqa_ckpt.sh \
        hb9b_noretr_ser_step480=$M/hb9b_noretr_ser_step480 base9b=Qwen/Qwen3.5-9B
    wait_file "$M/hb27b_fixed_step60/MERGE_DONE"
    GPUS=2,3 TP=2 bash scripts/self_evolving/eval/run_medxpertqa_ckpt.sh hb27b_fixed_step60=$M/hb27b_fixed_step60
    wait_file "$M/hb9b_noretr_fixed_step860/MERGE_DONE"
    GPUS=2,3 TP=2 bash scripts/self_evolving/eval/run_medxpertqa_ckpt.sh hb9b_noretr_fixed_step860=$M/hb9b_noretr_fixed_step860
    echo "=== $(date -u +%FT%TZ) VLLM CHAIN DONE" ;;
esac
