#!/usr/bin/env bash
# Wait for the node's GPUs to drain, then launch a training script.
#
# The node is shared, so queued work must not preempt whatever is running. This
# polls until every GPU has been below MAX_USED_MIB for HOLD consecutive checks
# (a single quiet sample is not enough - a job between phases can look idle), then
# runs the command once and exits.
#
#   nohup bash queue_when_gpus_free.sh scripts/self_evolving/train/run_trainset_rl_resume.sh \
#         > /scratch/sheng/self_evolving/logs_baseline/queue_trainset_rl.log 2>&1 &
set -uo pipefail

CMD=${*:?usage: queue_when_gpus_free.sh <script> [args...]}
MAX_USED_MIB=${MAX_USED_MIB:-8000}   # a truly idle B200 sits near 0; 8 GB tolerates stragglers
NEED_GPUS=${NEED_GPUS:-4}
HOLD=${HOLD:-3}                      # consecutive quiet polls required
INTERVAL=${INTERVAL:-300}

quiet=0
while :; do
    free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
           awk -v m="$MAX_USED_MIB" '$1 < m {n++} END {print n+0}')
    if [ "$free" -ge "$NEED_GPUS" ]; then
        quiet=$((quiet + 1))
    else
        quiet=0
    fi
    echo "[queue] $(date -Is) ${free}/${NEED_GPUS} GPUs free, holds ${quiet}/${HOLD}"
    if [ "$quiet" -ge "$HOLD" ]; then
        echo "[queue] $(date -Is) launching: $CMD"
        exec bash $CMD
    fi
    sleep "$INTERVAL"
done
