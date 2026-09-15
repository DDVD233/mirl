#!/usr/bin/env bash
# Keep a training arm alive on an OPPORTUNISTIC MSR pod across Volcano preemptions.
# The pod comes back as a fresh container (no tmux, no local disk state) on the same
# frp ssh port once a half node frees up; the launcher's resume_mode=auto then picks
# the run up from its last NFS checkpoint. Polls kubectl every two minutes, relaunches
# on every return (attempt number increments), and optionally runs AFTER_CMD on the
# pod once its gen server answers (e.g. restart evaluation sweeps that share the pod).
#
#   POD_MATCH=evolving3-opp PORT=2336 ARM=24 EXP_SUFFIX=_websearch ATTEMPT=8 \
#       EXTRA='SAVE_FREQ=10' AFTER_CMD='...' [ATTACHED=1] bash scripts/self_evolving/train/resume_arm_when_pod_returns.sh
set -uo pipefail
S=/scratch/sheng/self_evolving
POD_MATCH="${POD_MATCH:?}"; PORT="${PORT:?}"; ARM="${ARM:?}"; EXP_SUFFIX="${EXP_SUFFIX:-}"; ATTEMPT="${ATTEMPT:?}"
EXTRA="${EXTRA:-}"          # extra VAR=val pairs for the launcher
AFTER_CMD="${AFTER_CMD:-}"  # run on the pod after the gen server is healthy
LOG=launch_arm${ARM}_${PORT}.log
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@point.dd.works"
pod_state() { kubectl get pods -n bonete52 --no-headers 2>/dev/null | awk -v m="$POD_MATCH" '$0 ~ m {print $3}' | head -1; }
if [ "${ATTACHED:-0}" = 1 ]; then   # the arm is already running on the pod: only take over after its next eviction
    echo "$(date -u +%FT%TZ) attached to a running arm on port $PORT; waiting for the pod to leave Running"
    while [ "$(pod_state)" = Running ]; do sleep 120; done
    echo "$(date -u +%FT%TZ) pod $POD_MATCH left Running; waiting for it to return"
fi
while :; do
    st=$(pod_state)
    echo "$(date -u +%FT%TZ) pod $POD_MATCH: ${st:-unknown}"
    if [ "$st" = Running ]; then
        gpus=$($SSH 'test -d /scratch/sheng/self_evolving/verl_specgap && nvidia-smi -L | grep -c B200' 2>/dev/null)
        if [ "${gpus:-0}" = 4 ]; then
            $SSH "S=$S; mv \$S/logs_hb9b/$LOG \$S/logs_hb9b/launch_arm${ARM}_${PORT}.attempt$((ATTEMPT-1)).log 2>/dev/null; tmux new-session -d -s main -n arm$ARM \"cd \$S/verl_specgap && ARM=$ARM EXP_SUFFIX=$EXP_SUFFIX $EXTRA bash scripts/self_evolving/train/launch_specgap_when_free.sh > \$S/logs_hb9b/$LOG 2>&1\"; sleep 5; tmux list-windows; hostname"
            echo "$(date -u +%FT%TZ) RELAUNCHED ARM=$ARM attempt $ATTEMPT on port $PORT"
            ATTEMPT=$((ATTEMPT+1))
            if [ -n "$AFTER_CMD" ]; then
                for i in $(seq 1 40); do   # up to 40 min for the gen server
                    sleep 60
                    [ "$(pod_state)" = Running ] || break
                    if $SSH 'curl -sf -m 5 localhost:8041/healthz >/dev/null' 2>/dev/null; then
                        $SSH "$AFTER_CMD" && echo "$(date -u +%FT%TZ) AFTER_CMD done on port $PORT"
                        break
                    fi
                done
            fi
            # Stay attached until the pod leaves Running, then wait for the next return.
            while [ "$(pod_state)" = Running ]; do sleep 120; done
            echo "$(date -u +%FT%TZ) pod $POD_MATCH left Running; waiting for it to return"
            continue
        fi
        echo "$(date -u +%FT%TZ) pod Running but not reachable/ready yet (gpus=${gpus:-none})"
    fi
    sleep 120
done
