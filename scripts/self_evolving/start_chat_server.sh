#!/usr/bin/env bash
# Start vLLM chat server for proposer/generator/validator/judge agents.
# Use TP=2 on 2 GPUs for Qwen3-VL-8B-Instruct.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash start_chat_server.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
MEM="${MEM:-0.85}"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM" \
    --trust-remote-code \
    "$@"
