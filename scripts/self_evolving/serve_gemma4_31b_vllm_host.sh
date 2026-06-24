#!/usr/bin/env bash
# Gemma-4-31B-it vLLM HOST for the SPLIT self-improve run.
# This node (server2, 2x B200) serves the gen-server backend + reward judge for the
# trainer running on server4. Key-gated; exposed at point.dd.works:18187 (internal :8188).
# TP=2. Pair with serve_gen_gemma4_31b_remote.sh + run_gemma4_31b_selfimprove_split.sh on server4.
set -xeuo pipefail
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
export CUDA_VISIBLE_DEVICES="${VLLM_GPUS:-0,1}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
GEN_HOST_KEY="${GEN_HOST_KEY:-$(cat /scratch/sheng/self_evolving/.gen_host_key)}"
LOG_DIR=/scratch/sheng/self_evolving/logs_si_gemma31b_split; mkdir -p "$LOG_DIR"
vllm serve google/gemma-4-31B-it --served-model-name google/gemma-4-31B-it \
  --tensor-parallel-size "${VLLM_TP:-2}" --host 0.0.0.0 --port "${VLLM_PORT:-8188}" \
  --api-key "$GEN_HOST_KEY" \
  --gpu-memory-utilization 0.9 --max-model-len 32768 \
  --attention-config '{"flash_attn_version": 2}' --trust-remote-code \
  2>&1 | tee "$LOG_DIR/vllm_host.log"
