#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
exec vllm serve Qwen/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8188 --trust-remote-code \
  --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.65 \
  --max-model-len 32768 --max-num-seqs 32 --reasoning-parser qwen3 \
  --mm-encoder-attn-backend TORCH_SDPA --limit-mm-per-prompt '{"image":4}'
