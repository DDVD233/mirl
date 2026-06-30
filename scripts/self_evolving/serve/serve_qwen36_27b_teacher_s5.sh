#!/usr/bin/env bash
# Serve Qwen3.6-27B as the server-5 teacher / self-judge endpoint, reachable at
# http://point.dd.works:18184/v1 (frpc maps this pod's local 8188 -> public 18184;
# see /etc/frp/frpc.toml). Runs DP2 x TP2 = 4 GPUs (2 replicas behind one
# OpenAI-compatible endpoint) for throughput — the previous 2-GPU (TP2) instance
# crashed under the combined gen-server + reward-judge load.
#
# Flags per the requested config: tool-choice + qwen3_coder tool parser +
# qwen3 reasoning parser + data mm-encoder TP. --mm-encoder-attn-backend
# TORCH_SDPA is the documented B200 fix for the ViT cute-kernel crash.
set -xeuo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-8188}"            # frpc: local 8188 -> public point.dd.works:18184
DP="${DP:-2}"
TP="${TP:-2}"
MEM="${MEM:-0.90}"
MAXLEN="${MAXLEN:-32768}"

export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec vllm serve "$MODEL" \
    --trust-remote-code \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --data-parallel-size "$DP" \
    --tensor-parallel-size "$TP" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --mm-encoder-attn-backend TORCH_SDPA \
    --dtype bfloat16 \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN"
