#!/usr/bin/env bash
# Autonomous bridge: wait for direct-trace generation to finish, then run the full
# SELECTIVE fix pipeline (kill teacher -> combine -> SFT -> merge -> v7-selective RL).
# Launched under nohup in a tmux window so the whole chain completes even if the
# interactive session drops.
set -uo pipefail
cd /scratch/sheng/self_evolving/verl_healthbench
LOGDIR=/scratch/sheng/self_evolving/logs_healthbench_rubric
DIRECT_JSONL=/scratch/sheng/self_evolving/direct_sft_traces.jsonl

echo "[$(date -u)] waiter: waiting for direct-trace gen to finish..."
# NOTE: the pattern lives in this file, not in this process's cmdline, so pgrep
# will not self-match (the process is 'bash wait_then_pipeline.sh').
# Max wait ~45 min, then proceed with whatever traces exist (>=150).
for _ in $(seq 1 90); do
    pgrep -f "make_selective_traces" >/dev/null 2>&1 || break
    sleep 30
done
# ensure gen is truly stopped before we grab the teacher GPUs
pkill -9 -f "make_selective_traces" 2>/dev/null || true
sleep 3
N=$(wc -l < "$DIRECT_JSONL" 2>/dev/null || echo 0)
echo "[$(date -u)] waiter: gen finished, $N direct traces."
if [ "$N" -lt 150 ]; then
    echo "[$(date -u)] waiter: FATAL too few direct traces ($N < 150), not launching pipeline."
    exit 1
fi
echo "[$(date -u)] waiter: launching run_selective_pipeline.sh"
bash scripts/self_evolving/train/run_selective_pipeline.sh
echo "[$(date -u)] waiter: pipeline returned (exit $?)"
