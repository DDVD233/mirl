#!/usr/bin/env bash
# Generate MIMIC-IV rare-disease test responses for off-the-shelf baselines on a
# 4xB200 pod (server4). Then run run_rejudge_baselines.sh to score with
# gpt-chat-latest (same judge as the trained-checkpoint table). Launch in tmux.
set -uo pipefail
export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /scratch/sheng/self_evolving/verl
mkdir -p /scratch/sheng/self_evolving/eval_baselines
/usr/local/bin/python scripts/self_evolving/eval/run_baseline_eval_sweep.py "$@" \
  2>&1 | tee /scratch/sheng/self_evolving/eval_baselines/baseline_gen.log
