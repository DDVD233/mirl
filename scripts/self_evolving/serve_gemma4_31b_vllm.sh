#!/usr/bin/env bash
# Standalone vLLM server for google/gemma-4-31B-it: the gen-server backend AND the
# reward judge for the Gemma-4-31B self-improvement run. TP=4 on GPUs 0-3.
# Run alongside serve_gen_gemma4_31b.sh (:8005) and run_gemma4_31b_selfimprove.sh (GPUs 4-7).
set -xeuo pipefail
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
export CUDA_VISIBLE_DEVICES="${VLLM_GPUS:-0,1,2,3}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
LOG_DIR=/scratch/sheng/self_evolving/logs_si_gemma31b; mkdir -p "$LOG_DIR"
vllm serve google/gemma-4-31B-it --served-model-name google/gemma-4-31B-it \
  --tensor-parallel-size 4 --host 0.0.0.0 --port "${VLLM_PORT:-8100}" \
  --gpu-memory-utilization 0.9 --max-model-len 32768 \
  --attention-config '{"flash_attn_version": 2}' --trust-remote-code \
  2>&1 | tee "$LOG_DIR/vllm.log"
