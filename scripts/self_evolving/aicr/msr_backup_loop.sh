#!/usr/bin/env bash
# Continuous backup of the stage-2 results from the MSR NFS to AICR scratch, run on
# the AICR login node (tmux), because MSR access may end at any time.
#
# Pulls, every INTERVAL seconds, through the `msr` ssh alias (pod 2333 with the
# transfer key): the on/off evaluation dumps and regrades, the general-arm ledgers,
# prompts, rollouts and launch logs, and the LATEST training checkpoint of each
# general arm (verl keeps one on the NFS; here the last two are kept so a pull that
# lands mid-save never leaves nothing usable). Layout mirrors
# /scratch/sheng/self_evolving under $B, like the 2026-09-11 rescue.
#
#   bash scripts/self_evolving/aicr/msr_backup_loop.sh      (tmux window msrbk2)
set -uo pipefail
B=/scratch/dvdai_mit/msr_backup_2026-09-11
M=/scratch/sheng/self_evolving
INTERVAL="${INTERVAL:-3600}"
RS=(rsync -a --partial --timeout=600 --info=stats1 --no-inc-recursive)
pull() {  # pull <relative path> [extra rsync args]
    local rel=$1; shift
    mkdir -p "$B/$(dirname "$rel")"
    for i in 1 2 3; do
        "${RS[@]}" "$@" "msr:$M/$rel" "$B/$(dirname "$rel")/" 2>&1 | grep -E "Number of regular files transferred|Total transferred|rsync error" | tr '\n' ' '
        local rc=${PIPESTATUS[0]}
        echo " <- $rel rc=$rc"
        [ "$rc" = 0 ] && return 0
        sleep 30
    done
    return 1
}
while :; do
    echo "=== $(date -u +%FT%TZ) backup pass"
    pull paper_refresh --exclude '*.lock'
    pull logs_hb9b/val_generations --include 'hbpro_*/' --include 'prbench_9b_*/' --include 'profbench_9b_*/' --include 'medxpert_9b_ship180_*/' --include '*general*/' --exclude '/*/' --include '*'
    for exp in hb9b_general_specgap_ship_retrieval_websearch hb27b_general_specgap_ship_retrieval_websearch \
               hb9b_general_simple_retrieval_websearch hb27b_general_simple_retrieval_websearch; do
        pull "logs_hb9b/$exp"
        pull "logs_hb9b/rollouts/$exp"
        # Latest full checkpoint only: verl rotates the NFS copy; a shell dir (data.pt
        # alone) is skipped by requiring the actor weights.
        steps=$(ssh -o ConnectTimeout=30 msr "ls -d $M/checkpoints/hb9b/$exp/global_step_* 2>/dev/null | while read d; do [ -d \$d/actor ] && ls \$d/actor/*.pt >/dev/null 2>&1 && [ -f \$d/data.pt ] && echo \$d; done | sort -t_ -k3 -n | tail -1" 2>/dev/null)
        for d in $steps; do
            rel=${d#$M/}
            pull "$rel"
            # keep the newest two on AICR
            ls -d "$B/checkpoints/hb9b/$exp"/global_step_* 2>/dev/null | sort -t_ -k3 -n | head -n -2 | xargs -r rm -rf
        done
    done
    pull logs_hb9b --include 'launch_arm2*.log' --include 'launch_arm2*.attempt*.log' --include 'gen_server_*general*.log' --include 'frozen9b_*general*.log' --exclude '*' --no-recursive 2>/dev/null || true
    pull kb/search_cache_arm23.sqlite
    echo "=== $(date -u +%FT%TZ) pass done; used: $(du -sh $B 2>/dev/null | cut -f1)"
    sleep "$INTERVAL"
done
