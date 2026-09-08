#!/usr/bin/env bash
# Dedicated frozen 9B inference on idle GPUs of an existing node.
set -euo pipefail
export HF_HOME=${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export OMP_NUM_THREADS=8
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 4000 {busy=1} END {exit !busy}'; then
    echo "A GPU is occupied; refusing to start the baseline server." >&2
    exit 1
fi
exec vllm serve Qwen/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B \
    --host 0.0.0.0 --port 8188 --trust-remote-code \
    --data-parallel-size 4 --tensor-parallel-size 1 --api-server-count 4 \
    --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 32768 \
    --max-num-seqs 64 --reasoning-parser qwen3 \
    --mm-encoder-attn-backend TORCH_SDPA --limit-mm-per-prompt '{"image":4}'
