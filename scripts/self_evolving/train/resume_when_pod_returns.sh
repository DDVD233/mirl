#!/usr/bin/env bash
# Wait for a preempted MSR pod to come back, then resume its arm from checkpoint.
#
# WHY THIS IS SEPARATE FROM watchdog_specgap.sh: the watchdog treats an UNREACHABLE port
# as "no evidence" and deliberately does nothing, because a port that stops answering
# usually means the frp tunnel moved, not that the run died -- acting on it once nearly
# started a second trainer on a live run's checkpoint. Pod LOSS looks identical from the
# outside. So recovery from preemption is a separate, explicitly-invoked job: the human
# has already confirmed the pod is gone and recreated it, and this only waits for the new
# one to answer.
#
# 2026-08-11: both half-node pods are on the volcano OPPORTUNISTIC queue
# (volcano.sh/preemptable: "true", docker/kub_config/config{2,3}_opp.yaml). They borrow
# idle capacity and are reclaimed without warning; both were reclaimed at once. Their
# checkpoints live on shared NFS, so nothing is lost -- only the compute stops.
#
# Usage (detached, on the local machine):
#   setsid bash scripts/self_evolving/train/resume_when_pod_returns.sh >> log 2>&1 &
set -uo pipefail

REPO=/scratch/sheng/self_evolving/verl_specgap
POLL_S="${POLL_S:-120}"
MAX_WAIT_H="${MAX_WAIT_H:-72}"

# port:arm:window:exp_suffix:logfile:extra_env
ARMS=(
  "2335:1:arm1:_fixedprompt:specgap_arm1_fixedprompt_launch.log:"
  "2336:2:arm2rw:_rewrite:specgap_arm2_rewrite_launch.log:HB_REFINE_MODE=rewrite"
)

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

deadline=$(( SECONDS + MAX_WAIT_H * 3600 ))
declare -A DONE
for a in "${ARMS[@]}"; do DONE["${a%%:*}"]=0; done

log "waiting for pods on ports: $(for a in "${ARMS[@]}"; do printf '%s ' "${a%%:*}"; done)"

while :; do
    all=1
    for a in "${ARMS[@]}"; do
        IFS=: read -r port arm win sfx logf extra <<< "$a"
        [ "${DONE[$port]}" = "1" ] && continue
        all=0

        # A recreated pod has a NEW host key; a stale known_hosts entry fails the connect
        # with "REMOTE HOST IDENTIFICATION HAS CHANGED" and would look like "still down".
        ssh-keygen -f "$HOME/.ssh/known_hosts" -R "[point.dd.works]:$port" >/dev/null 2>&1
        host=$(timeout 40 ssh -o ConnectTimeout=12 -o BatchMode=yes \
                 -o StrictHostKeyChecking=accept-new -p "$port" root@point.dd.works \
                 'hostname' 2>/dev/null)
        [ -z "$host" ] && continue

        # Refuse to launch onto GPUs that are not actually free: a pod can answer ssh
        # while another workload still holds memory, and engine init would OOM.
        used=$(timeout 40 ssh -o ConnectTimeout=12 -o BatchMode=yes -p "$port" root@point.dd.works \
                 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1' 2>/dev/null)
        if [ "${used:-99999}" -gt 4000 ] 2>/dev/null; then
            log "[$port] $host up but ${used}MiB held; waiting"
            continue
        fi

        log "[$port] $host is up and idle -- resuming ARM=$arm$sfx (resume_mode=auto continues from checkpoint)"
        timeout 120 ssh -o ConnectTimeout=15 -o BatchMode=yes -p "$port" root@point.dd.works \
            "rm -f /dev/shm/v''llm* 2>/dev/null
             tmux has-session -t hb 2>/dev/null || tmux new-session -d -s hb -n idle 'sleep infinity'
             tmux kill-window -t hb:$win 2>/dev/null
             tmux new-window -t hb -n $win \"cd $REPO && $extra ARM=$arm EXP_SUFFIX=$sfx \
                 bash scripts/self_evolving/train/launch_specgap_when_free.sh \
                 2>&1 | tee -a /scratch/sheng/self_evolving/logs_hb9b/$logf\"" \
            && { DONE[$port]=1; log "[$port] launched"; } \
            || log "[$port] launch command failed; will retry next poll"
    done

    [ "$all" = "1" ] && { log "all arms resumed"; exit 0; }
    if [ "$SECONDS" -ge "$deadline" ]; then
        log "giving up after ${MAX_WAIT_H}h; pods still not back"
        exit 1
    fi
    sleep "$POLL_S"
done
