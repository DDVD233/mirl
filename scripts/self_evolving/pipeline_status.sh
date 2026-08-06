#!/usr/bin/env bash
# One-shot status of the selective-SFT -> merge -> RL overnight pipeline.
L=/scratch/sheng/self_evolving/logs_healthbench_rubric
echo "=== $(date -u) ==="
echo "[direct-gen] traces=$(wc -l < /scratch/sheng/self_evolving/direct_sft_traces.jsonl 2>/dev/null)  running=$(ps aux | grep -c '[m]ake_selective_traces')"
grep -E "direct=" "$L/direct_tracegen.log" 2>/dev/null | tail -1
echo "[waiter] $(tail -1 "$L/pipeline_waiter.log" 2>/dev/null)"
echo "[SFT] $(grep -E 'step:|loss|Epoch' "$L/selective_sft.log" 2>/dev/null | tail -1)"
echo "[merge] $(tail -1 "$L/selective_merge.log" 2>/dev/null)"
echo "[RL val] $(grep 'val-core/overall/acc/mean' "$L/v7_selective_train.log" 2>/dev/null | tail -1 | grep -oE 'val-core/overall/acc/mean:[0-9.]+')"
echo "[RL last] $(grep -E 'Validation \(step|step:[0-9]' "$L/v7_selective_train.log" 2>/dev/null | tail -1)"
echo "[GPU] $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
