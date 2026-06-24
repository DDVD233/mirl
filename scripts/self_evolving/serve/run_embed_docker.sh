#!/usr/bin/env bash
# Run the Qwen3-VL embedding service as a Docker container on mib.
#   - GPU 3 (same as the bare process it replaces)
#   - host port 18001 -> container 8000
#   - same vLLM config: pooling/embed runner, gpu_util 0.35, max_model_len 8192
#   - image pinned to v0.20.2 (matches the known-working bare build)
#   - --restart unless-stopped: auto-restarts on crash AND on host reboot
#     (Docker daemon is enabled on boot on mib, so the container comes back up).
# Idempotent: removes any existing container of the same name first.
set -euo pipefail
NAME="${NAME:-vllm-embed-qwen3vl}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.20.2}"
GPU="${GPU:-3}"
PORT="${PORT:-18001}"
# GPU 3 is shared; vLLM aborts at startup if free mem < UTIL*total. 0.35 needed
# ~33GB free and flapped when GPU3 dipped below that. 0.22 (~21GB) fits reliably
# while still leaving KV headroom for the 2B embed model. Override via GPU_UTIL.
GPU_UTIL="${GPU_UTIL:-0.22}"

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
    --restart unless-stopped \
    --gpus "device=$GPU" \
    --ipc=host \
    -p "${PORT}:8000" \
    -v /home/dvd/.cache/huggingface:/root/.cache/huggingface \
    "$IMAGE" \
    --model Qwen/Qwen3-VL-Embedding-2B \
    --runner pooling --convert embed \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 8192 \
    --trust-remote-code
echo "started container '$NAME' on GPU $GPU, host port $PORT (image $IMAGE)"
