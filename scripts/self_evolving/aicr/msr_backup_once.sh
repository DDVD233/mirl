#!/usr/bin/env bash
# ONE-TIME transfer of a finished arm from the MSR NFS to AICR: its pinned best checkpoint
# (best_global_step_*; falls back to the latest global_step_*), its ledgers, prompts,
# rollouts, launch logs and validation dumps. Run on the AICR login node when the arm is
# done (2026-09-15: replaces the hourly pulls of every new checkpoint).
#
#   bash msr_backup_once.sh <exp>            e.g. hb27b_general_specgap_ship_retrieval_websearch
set -uo pipefail
EXP="${1:?exp name}"
B=/scratch/dvdai_mit/msr_backup_2026-09-11
M=/scratch/sheng/self_evolving
RS=(rsync -a --partial --timeout=600 --info=stats1 --no-inc-recursive)
pull() { local rel=$1; shift; mkdir -p "$B/$(dirname "$rel")"
    "${RS[@]}" "$@" "msr:$M/$rel" "$B/$(dirname "$rel")/" 2>&1 | grep -E "Total transferred|rsync error"; echo " <- $rel rc=${PIPESTATUS[0]}"; }
best=$(ssh -o ConnectTimeout=30 msr "ls -d $M/checkpoints/hb9b/$EXP/best_global_step_* 2>/dev/null | tail -1")
[ -z "$best" ] && best=$(ssh -o ConnectTimeout=30 msr "ls -d $M/checkpoints/hb9b/$EXP/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1")
echo "=== $(date -u +%FT%TZ) one-time backup of $EXP; checkpoint: ${best:-none}"
[ -n "$best" ] && pull "${best#$M/}"
pull "logs_hb9b/$EXP"
pull "logs_hb9b/rollouts/$EXP"
pull "logs_hb9b/val_generations/$EXP"
pull logs_hb9b --include "launch_*${EXP#hb}*" --include "gen_server_${EXP}*.log" --include "frozen9b_${EXP}*.log" --exclude '*' --no-recursive
pull paper_refresh --exclude '*.lock'
echo "=== $(date -u +%FT%TZ) done; AICR used: $(du -sh $B | cut -f1)"
