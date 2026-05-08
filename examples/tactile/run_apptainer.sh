#!/usr/bin/env bash
# Run a verl training script inside the verlai/verl apptainer container.
#
# Usage:
#   bash examples/tactile/run_apptainer.sh <inner_script.sh> [inner_script_args...]
#
# Example:
#   bash examples/tactile/run_apptainer.sh examples/tactile/run_gemma3-4b-fsdp_date.sh
#
# Recommended invocation (so the run survives the SSH session and is monitorable):
#   tmux new-session -d -s verl \
#     "bash examples/tactile/run_apptainer.sh examples/tactile/run_gemma3-4b-fsdp_date.sh \
#       2>&1 | tee \$HOME/verl/logs/verl-apptainer-\$(date +%s).log"
#   tmux attach -t verl
#
# Environment overrides:
#   IMAGE_TAG  — docker tag to pull (default: verlai/verl:sgl059.dev3)
#   SIF        — path to the .sif file (default: ~/scratch/apptainer/<image>.sif)
#   VERL_HOST  — host path to verl source we want shadowing the one in the image
#                (default: ~/verl)

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <training_script.sh> [args...]" >&2
  exit 1
fi
TRAIN_SCRIPT="$1"; shift

# Make sure apptainer is on PATH (cluster modules-style env).
if ! command -v apptainer >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load apptainer
  else
    echo "apptainer not found and 'module' command unavailable" >&2
    exit 1
  fi
fi

VERL_HOST="${VERL_HOST:-$HOME/verl}"
SCRATCH="${SCRATCH:-$HOME/scratch}"
IMAGE_TAG="${IMAGE_TAG:-verlai/verl:sgl059.dev3}"
SIF="${SIF:-$SCRATCH/apptainer/$(echo "$IMAGE_TAG" | tr ':/' '_').sif}"

mkdir -p "$(dirname "$SIF")" "$SCRATCH/apptainer/cache" "$SCRATCH/huggingface" "$VERL_HOST/logs"

# One-time pull. apptainer pull is idempotent only via the cachedir; the .sif
# itself is the persistent artifact, so guard with a file existence check.
if [ ! -f "$SIF" ]; then
  echo ">>> Pulling $IMAGE_TAG -> $SIF (one-time, multi-GB download)"
  APPTAINER_CACHEDIR="$SCRATCH/apptainer/cache" \
    apptainer pull "$SIF" "docker://$IMAGE_TAG"
fi

# Inside the container:
#   - $HOME is auto-bound, so $HOME/scratch/... paths in inner scripts resolve.
#   - Bind our verl source over /workspace/verl and prepend to PYTHONPATH so
#     our patched code shadows the verl that was pip-installed into the image.
#   - --writable-tmpfs gives a writable overlay for things like ray tmp dirs
#     and pip --user installs.
#   - We strip any leaked host build env (NVCC_PREPEND_FLAGS, CC/CXX, etc.)
#     so the container's own toolchain isn't hijacked.
# --cleanenv strips host env from the container shell. We then re-export only
# what verl needs. This avoids inheriting host BASH_ENV (which points at lmod
# init scripts that don't exist inside the image), CONDA_*, NVCC_PREPEND_FLAGS
# that leaked from a sibling conda env, etc.
exec apptainer exec --nv --writable-tmpfs --cleanenv \
  --env "HOME=$HOME" \
  --env "PYTHONPATH=/workspace/verl" \
  --env "HF_HOME=$SCRATCH/huggingface" \
  --env "VLLM_ALLREDUCE_USE_SYMM_MEM=0" \
  --env "NCCL_P2P_DISABLE=1" \
  --bind "$VERL_HOST:/workspace/verl" \
  "$SIF" \
  bash -c '
    set -x
    cd /workspace/verl
    bash '"$TRAIN_SCRIPT"' "$@"
  ' bash "$@"
