#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for the proposer/judge API.
# Run this on a separate machine or GPU before starting training.
#
# Usage: bash start_vllm_server.sh [--model MODEL] [--port PORT]

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    "$@"
