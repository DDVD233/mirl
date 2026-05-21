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

SIF="${SIF:-${HOME}/scratch/dvd/sif/vllm-openai-v0.20.0-cu130.sif}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-8005}"
TP="${TP:-4}"
HF_CACHE="${HF_CACHE:-/home/dvdai/.cache/huggingface}"
# Resolve symlinks before binding: on vps3, ~/.cache/huggingface is a symlink
# into /orcd/compute/ppliang/001/dvdai/huggingface. Binding the symlink source
# doesn't help the container — apptainer doesn't auto-mount the symlink target,
# so the symlink dangles inside the container. Bind the real path instead and
# point HF_HOME at it so HuggingFace skips the symlink entirely.
HF_CACHE_REAL="$(readlink -f "$HF_CACHE")"
LOG_FILE="${LOG_FILE:-/tmp/teacher_vllm.log}"

source /usr/share/lmod/lmod/init/bash
module load apptainer/1.4.2

echo "=== launching vllm via apptainer run at $(date) ==="
echo "    SIF=$SIF"
echo "    MODEL=$MODEL  PORT=$PORT  TP=$TP"
echo "    HF_CACHE=$HF_CACHE  (resolved → $HF_CACHE_REAL)  LOG_FILE=$LOG_FILE"

# Image's runscript is `vllm serve`, so args after the SIF are appended to
# that (no need to specify `vllm serve` ourselves). --writable-tmpfs gives
# the container a writable /tmp without persisting changes. --nv exposes
# host GPUs via the NVIDIA Container Toolkit shim apptainer ships.
# --cleanenv: strip the host's env so things like CC/CXX/CUDA_HOME pointing at
# host module-loaded paths (e.g. /orcd/software/core/001/spack/...) don't leak
# into the container. DeepGEMM's JIT does string-based path existence checks
# on those variables, so each leaked var means a separate boot failure.
# Re-set what we actually need; HF_HOME has to be the host's symlink target
# (see `readlink -f` above), the toolchain ones point at the image's gcc/nvcc.
exec apptainer run --nv --writable-tmpfs --cleanenv \
  --bind "${HF_CACHE_REAL}":"${HF_CACHE_REAL}" \
  --env HF_HOME="${HF_CACHE_REAL}" \
  --env CUDA_HOME=/usr/local/cuda \
  --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin \
  --env CC=/usr/bin/gcc \
  --env CXX=/usr/bin/g++ \
  "$SIF" \
  --model "$MODEL" \
  -tp "$TP" \
  --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.95}" \
  --trust-remote-code 2>&1 | tee "$LOG_FILE"
