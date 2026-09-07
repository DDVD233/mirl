#!/usr/bin/env bash
# AICR-side finalizer for the stage-2 checkpoint transfers (replaces the first watcher).
# For each item: wait until no other rsync of that item is running (the initial streams),
# then re-run rsync until it exits 0 (the frp tunnel drops now and then; rsync resumes in
# place), then touch <dest>/RSYNC_DONE on the MSR pod so chain_stage2_transfer.sh proceeds.
set -u
S=/scratch/dvdai_mit/self_evolving/checkpoints
D=/scratch/sheng/self_evolving/checkpoints/hb9b_aicr
ITEMS="hb9b/hb9b_specgap_ship_noretrieval_sj_aicr/global_step_480 hb9b/hb27b_specgap_simple_retrieval_aicr/global_step_60 hb9b/hb9b_specgap_simple_noretrieval_sj_aicr/global_step_860 self_evolving_medical/gpt56_sft_qwen36_27b"
for p in $ITEMS; do
  while pgrep -f "rsync -a --partial --inplace.* $S/$p/" > /dev/null; do sleep 60; done
  for attempt in $(seq 1 30); do
    echo "=== $(date -u +%FT%TZ) finalize $p attempt $attempt"
    ssh -o BatchMode=yes msr3 "mkdir -p $D/$p" && rsync -a --partial --inplace "$S/$p/" "msr3:$D/$p/" && break
    sleep 120
  done
  if ssh -o BatchMode=yes msr3 "touch $D/$p/RSYNC_DONE"; then echo "=== $(date -u +%FT%TZ) marker $p"; else echo "=== marker FAILED $p"; fi
done
echo ALL_MARKERS
