#!/usr/bin/env bash
# Serve the OPD teacher (Qwen3.5-397B-A17B-FP8) via the official
# vllm/vllm-openai docker image, run under apptainer on vps3 (4xH200).
# Lmod's apptainer module needs to be sourced first.
#
# Defaults match the existing frp tunnel layout (host 0.0.0.0, port 8005 →
# vps3.dd.works:18005 via frpc).
#
# Override via env to swap model / port / parallelism without editing the file.

set -e

SIF="${SIF:-/scratch/dvdai/vllm-openai-v0.20.0-cu130.sif}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-8005}"
TP="${TP:-4}"
HF_CACHE="${HF_CACHE:-/home/dvdai/.cache/huggingface}"
LOG_FILE="${LOG_FILE:-/tmp/teacher_vllm.log}"

source /usr/share/lmod/lmod/init/bash
module load apptainer/1.4.2

echo "=== launching vllm via apptainer run at $(date) ==="
echo "    SIF=$SIF"
echo "    MODEL=$MODEL  PORT=$PORT  TP=$TP"
echo "    HF_CACHE=$HF_CACHE  LOG_FILE=$LOG_FILE"

# Image's runscript is `vllm serve`, so args after the SIF are appended to
# that (no need to specify `vllm serve` ourselves). --writable-tmpfs gives
# the container a writable /tmp without persisting changes. --nv exposes
# host GPUs via the NVIDIA Container Toolkit shim apptainer ships.
exec apptainer run --nv --writable-tmpfs \
  --bind "${HF_CACHE}":/root/.cache/huggingface \
  "$SIF" \
  --model "$MODEL" \
  -tp "$TP" \
  --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --trust-remote-code 2>&1 | tee "$LOG_FILE"
