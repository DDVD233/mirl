#!/usr/bin/env bash
# Serve Qwen3.6-27B on ONE 4xB200 node as a local teacher endpoint, using
# data-parallel 2 x tensor-parallel 2 (2 replicas, 2 GPUs each = 4 GPUs) behind a
# single OpenAI-compatible endpoint. Used as the distillation teacher for
# make_distill_traces.py so distillation hits localhost (no shared-ingress cap).
#
#   CUDA_VISIBLE_DEVICES handled by vLLM across all 4 GPUs.
#   B200 fix: --mm-encoder-attn-backend TORCH_SDPA (ViT cute-kernel crash).
#   NO --reasoning-parser: distillation needs the full reasoning prose + boxed
#   answer in `content` (enable_thinking=False), matching the server-5 teacher.
set -xeuo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-8000}"
DP="${DP:-2}"
TP="${TP:-2}"
MEM="${MEM:-0.90}"
MAXLEN="${MAXLEN:-32768}"

export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec /usr/local/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --data-parallel-size "$DP" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN" \
    --mm-encoder-attn-backend TORCH_SDPA \
    --trust-remote-code
