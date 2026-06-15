#!/usr/bin/env bash
# Start vLLM embedding server for Qwen3-VL-Embedding (pooling runner).
# Used for Milvus retrieval queries.
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash start_embedding_server.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
PORT="${PORT:-8001}"
TP="${TP:-1}"
MEM="${MEM:-0.3}"

# vLLM >= 0.19 uses --runner pooling --convert embed
# vLLM <= 0.11 uses --task embed
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --runner pooling \
    --convert embed \
    --port "$PORT" \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM" \
    --trust-remote-code \
    "$@"
