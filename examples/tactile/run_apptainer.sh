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
# On compute nodes apptainer is provided via lmod ("module load apptainer").
# 'module' is a shell function only set up after sourcing lmod's init script,
# which is normally done by /etc/profile.d/* — but we run as a non-interactive
# script, so we may need to source it ourselves.
if ! command -v apptainer >/dev/null 2>&1; then
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /usr/share/lmod/lmod/init/profile; do
      if [ -f "$init" ]; then
        # shellcheck source=/dev/null
        source "$init" 2>/dev/null || true
        break
      fi
    done
  fi
  if type module >/dev/null 2>&1; then
    module load apptainer
  fi
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer not on PATH and 'module load apptainer' did not help." >&2
  echo "       Try running 'module load apptainer' manually before launching." >&2
  exit 1
fi

VERL_HOST="${VERL_HOST:-$HOME/verl}"
SCRATCH="${SCRATCH:-$HOME/scratch}"
IMAGE_TAG="${IMAGE_TAG:-verlai/verl:sgl059.dev3}"
SIF="${SIF:-$SCRATCH/apptainer/$(echo "$IMAGE_TAG" | tr ':/' '_').sif}"
# Persistent overrides directory: holds python packages that we want to shadow
# the ones inside the image (notably transformers 4.57.1, since sglang 0.5.9's
# Gemma3 implementation has many incompatibilities with transformers >= 5).
PYOVERRIDES="${PYOVERRIDES:-$SCRATCH/python_overrides}"

# On this cluster, $HOME/{scratch,verl/outputs,verl/checkpoints} are symlinks
# into /orcd/compute/.../<user>. apptainer auto-binds $HOME, but does NOT
# follow those symlinks, so we must bind their real target dir explicitly.
# Walk a few candidate symlinks and collect the unique top-level prefixes.
declare -A _bind_set
for src in "$SCRATCH" "$VERL_HOST/outputs" "$VERL_HOST/checkpoints"; do
  if [ -L "$src" ]; then
    real="$(readlink -f "$src")"
    # Use the first 4 path components, e.g. /orcd/compute/ppliang/001
    prefix="$(echo "$real" | awk -F/ '{print "/"$2"/"$3"/"$4"/"$5}')"
    [ -d "$prefix" ] && _bind_set[$prefix]=1
  fi
done
EXTRA_BINDS=""
for p in "${!_bind_set[@]}"; do
  EXTRA_BINDS="$EXTRA_BINDS --bind $p:$p"
done

mkdir -p "$(dirname "$SIF")" "$SCRATCH/apptainer/cache" "$VERL_HOST/logs" "$PYOVERRIDES"

# One-time pull. apptainer pull is idempotent only via the cachedir; the .sif
# itself is the persistent artifact, so guard with a file existence check.
if [ ! -f "$SIF" ]; then
  echo ">>> Pulling $IMAGE_TAG -> $SIF (one-time, multi-GB download)"
  APPTAINER_CACHEDIR="$SCRATCH/apptainer/cache" \
    apptainer pull "$SIF" "docker://$IMAGE_TAG"
fi

# One-time pin: install transformers 4.57.1 (and its incompatible-with-image
# deps) into $PYOVERRIDES. We use --target so the install is location-agnostic
# and we control the path prepended to PYTHONPATH. --no-deps avoids dragging in
# torch / tokenizers / safetensors / etc. (those stay on the image), but a few
# packages that move in lockstep with transformers must be co-pinned:
#   - huggingface-hub<1.0 (image has 1.x; transformers 4.x rejects it)
#   - tokenizers<0.23 if image has 0.23+ (verlai image has 0.22.x already, OK)
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-4.57.1}"
HF_HUB_PIN="${HF_HUB_PIN:-0.34.4}"
if [ ! -d "$PYOVERRIDES/transformers" ] || [ ! -d "$PYOVERRIDES/huggingface_hub" ]; then
  echo ">>> Installing transformers==$TRANSFORMERS_PIN + huggingface-hub==$HF_HUB_PIN to $PYOVERRIDES (one-time)"
  apptainer exec --nv --writable-tmpfs --cleanenv \
    --env "HOME=$HOME" \
    --bind "$PYOVERRIDES:/pyoverrides" \
    "$SIF" \
    pip install --no-deps --target=/pyoverrides --upgrade \
      "transformers==$TRANSFORMERS_PIN" "huggingface-hub==$HF_HUB_PIN"
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
  --env "PYTHONPATH=/pyoverrides:/workspace/verl" \
  --env "VLLM_ALLREDUCE_USE_SYMM_MEM=0" \
  --env "NCCL_P2P_DISABLE=1" \
  --bind "$VERL_HOST:/workspace/verl" \
  --bind "$PYOVERRIDES:/pyoverrides" \
  $EXTRA_BINDS \
  "$SIF" \
  bash -c '
    set -x
    cd /workspace/verl
    bash '"$TRAIN_SCRIPT"' "$@"
  ' bash "$@"
