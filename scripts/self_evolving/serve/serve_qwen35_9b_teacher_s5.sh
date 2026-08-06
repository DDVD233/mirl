#!/usr/bin/env bash
# Serve Qwen3.5-9B as the server-5 generator/judge endpoint for the 9B phase-1
# ablation (replicates j75o3rrt with the base model as its own teacher/judge),
# reachable at http://point.dd.works:18184/v1 (frpc: pod 8188 -> public 18184).
# This pod has a single B200 -> TP1. Text-only model: no mm-encoder flags
# (unlike serve_qwen36_27b_teacher_s5.sh).
set -xeuo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8188}"
MEM="${MEM:-0.90}"
MAXLEN="${MAXLEN:-32768}"

export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec vllm serve "$MODEL" \
    --trust-remote-code \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN"
