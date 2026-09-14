#!/usr/bin/env bash
# Relaunch a training arm on an OPPORTUNISTIC MSR pod after Volcano preempts it.
# The pod comes back as a fresh container (no tmux, no local disk state) on the same
# frp ssh port once a half node frees up; the launcher's resume_mode=auto then picks
# the run up from its last NFS checkpoint. Polls kubectl every two minutes.
#
#   POD_MATCH=evolving3-opp PORT=2336 ARM=24 EXP_SUFFIX=_websearch ATTEMPT=8 \
#       bash scripts/self_evolving/train/resume_arm_when_pod_returns.sh
set -uo pipefail
S=/scratch/sheng/self_evolving
POD_MATCH="${POD_MATCH:?}"; PORT="${PORT:?}"; ARM="${ARM:?}"; EXP_SUFFIX="${EXP_SUFFIX:-}"; ATTEMPT="${ATTEMPT:?}"
EXTRA="${EXTRA:-}"          # extra VAR=val pairs for the launcher
LOG=launch_arm${ARM}_${PORT}.log
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@point.dd.works"
while :; do
    st=$(kubectl get pods -n bonete52 --no-headers 2>/dev/null | awk -v m="$POD_MATCH" '$0 ~ m {print $3}' | head -1)
    echo "$(date -u +%FT%TZ) pod $POD_MATCH: ${st:-unknown}"
    if [ "$st" = Running ]; then
        gpus=$($SSH 'test -d /scratch/sheng/self_evolving/verl_specgap && nvidia-smi -L | grep -c B200' 2>/dev/null)
        if [ "${gpus:-0}" = 4 ]; then
            $SSH "S=$S; mv \$S/logs_hb9b/$LOG \$S/logs_hb9b/launch_arm${ARM}_${PORT}.attempt$((ATTEMPT-1)).log 2>/dev/null; tmux new-session -d -s main -n arm$ARM \"cd \$S/verl_specgap && ARM=$ARM EXP_SUFFIX=$EXP_SUFFIX $EXTRA bash scripts/self_evolving/train/launch_specgap_when_free.sh > \$S/logs_hb9b/$LOG 2>&1\"; sleep 5; tmux list-windows; hostname"
            echo "$(date -u +%FT%TZ) RELAUNCHED ARM=$ARM attempt $ATTEMPT on port $PORT"
            break
        fi
        echo "$(date -u +%FT%TZ) pod Running but not reachable/ready yet (gpus=${gpus:-none})"
    fi
    sleep 120
done
