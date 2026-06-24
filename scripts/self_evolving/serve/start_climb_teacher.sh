#!/usr/bin/env bash
# Standalone teacher / judge vLLM endpoint for the CLIMB self-evolving pipeline.
#
# Runs on a DEDICATED node (e.g. server5, 2x B200) so the trainer node keeps all
# its GPUs for the actor. Both the generation server (question proposer/generator,
# which sees retrieved images) and the medical reward judge call this endpoint, so
# the model must be vision-capable. The API key is read from CLIMB_API_KEY (env)
# and never hardcoded.
#
# Reachability: bind inside the pod to $PORT; the pod exposes it on an external
# port (e.g. 8188 -> 18184). Point the trainer node at the external address:
#   API_BASE=http://point.dd.works:<external_port>/v1
set -xeuo pipefail

export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-8188}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LOG_FILE="${LOG_FILE:-/root/climb_teacher.log}"

: "${CLIMB_API_KEY:?set CLIMB_API_KEY (the teacher/judge API key) in the env}"

vllm serve "$MODEL_NAME" --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" \
  --attention-config '{"flash_attn_version": 2}' --trust-remote-code \
  --api-key "$CLIMB_API_KEY" \
  --limit-mm-per-prompt '{"image": 12}' \
  2>&1 | tee "$LOG_FILE"
