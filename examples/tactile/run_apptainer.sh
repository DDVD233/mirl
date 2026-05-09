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
IMAGE_TAG="${IMAGE_TAG:-verlai/verl:vllm017.latest}"
SIF="${SIF:-$SCRATCH/apptainer/$(echo "$IMAGE_TAG" | tr ':/' '_').sif}"
# Persistent overrides directory: holds python packages we want to shadow the
# ones inside the image. Empty by default; set ENABLE_TRANSFORMERS_PIN=1 to
# install a specific transformers version (was needed for sgl059.dev3 but not
# for vllm018.dev1).
PYOVERRIDES="${PYOVERRIDES:-$SCRATCH/python_overrides_$(echo "$IMAGE_TAG" | tr ':/' '_')}"

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

# Optional one-time transformers pin (only needed for sgl059.dev3; vllm018
# has a working transformers stack out of the box).
if [ "${ENABLE_TRANSFORMERS_PIN:-0}" = "1" ]; then
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
fi

# torchcodec for qwen_vl_utils video decoding. The vllm017.latest image has
# pyav 16 (which bundles ffmpeg 8 shared libs) but no system ffmpeg, so we
# (1) install torchcodec 0.10.0+cu129 (matches torch 2.10) into pyoverrides
# (2) build a directory of soname symlinks (libavcodec.so.62 etc.) pointing
#     at pyav's hash-suffixed libs so torchcodec's dlopen can find them
# (3) mount that dir + pyav's .libs dir on LD_LIBRARY_PATH at run time.
TORCHCODEC_VERSION="${TORCHCODEC_VERSION:-0.10.0}"
TORCHCODEC_INDEX_URL="${TORCHCODEC_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
FFMPEG_LINKS="${FFMPEG_LINKS:-$SCRATCH/ffmpeg_links_$(echo "$IMAGE_TAG" | tr ':/' '_')}"
mkdir -p "$FFMPEG_LINKS"
if [ ! -d "$PYOVERRIDES/torchcodec" ]; then
  echo ">>> Installing torchcodec==$TORCHCODEC_VERSION+cu129 to $PYOVERRIDES (one-time)"
  apptainer exec --writable-tmpfs --cleanenv \
    --env "HOME=$HOME" \
    --bind "$PYOVERRIDES:/pyoverrides" \
    "$SIF" \
    pip install --target=/pyoverrides --upgrade \
      --index-url="$TORCHCODEC_INDEX_URL" "torchcodec==$TORCHCODEC_VERSION"
fi
if [ ! -e "$FFMPEG_LINKS/libavcodec.so.62" ]; then
  echo ">>> Building ffmpeg soname symlinks at $FFMPEG_LINKS (one-time)"
  apptainer exec --cleanenv \
    --env "HOME=$HOME" \
    --bind "$FFMPEG_LINKS:/ffmpeg_links" \
    "$SIF" \
    bash -c '
      cd /ffmpeg_links
      # Hash-stripped copies (libavcodec.so.62.11.100 etc.)
      for f in /usr/local/lib/python3.12/dist-packages/av.libs/lib*-*.so.*; do
        base=$(basename "$f")
        short=$(echo "$base" | sed -E "s/-[a-f0-9]+\.so\./.so./")
        ln -sfn "$f" "$short"
      done
      # Soname symlinks (libavcodec.so.62 -> libavcodec.so.62.11.100)
      for f in libavcodec.so.62.* libavformat.so.62.* libavutil.so.60.* \
               libavfilter.so.11.* libavdevice.so.62.* \
               libswresample.so.6.* libswscale.so.9.*; do
        if [ -e "$f" ]; then
          short=${f%.[0-9]*.[0-9]*}
          ln -sfn "$f" "$short"
        fi
      done
    '
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
PYTHON_PATH_VAL="/workspace/verl"
PYOVERRIDES_BIND=""
# Mount pyoverrides whenever it has any installed package (e.g. decord for
# qwen_vl_utils video reading) — not just when ENABLE_TRANSFORMERS_PIN is set.
if [ "${ENABLE_TRANSFORMERS_PIN:-0}" = "1" ] || [ -d "$PYOVERRIDES" ] && [ "$(ls -A "$PYOVERRIDES" 2>/dev/null)" ]; then
  PYTHON_PATH_VAL="/pyoverrides:/workspace/verl"
  PYOVERRIDES_BIND="--bind $PYOVERRIDES:/pyoverrides"
fi

exec apptainer exec --nv --writable-tmpfs --cleanenv \
  --env "HOME=$HOME" \
  --env "PYTHONPATH=$PYTHON_PATH_VAL" \
  --env "LD_LIBRARY_PATH=/ffmpeg_links:/usr/local/lib/python3.12/dist-packages/av.libs" \
  --env "VLLM_ALLREDUCE_USE_SYMM_MEM=0" \
  --env "NCCL_P2P_DISABLE=1" \
  --env "FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-torchcodec}" \
  --bind "$VERL_HOST:/workspace/verl" \
  --bind "$FFMPEG_LINKS:/ffmpeg_links" \
  $PYOVERRIDES_BIND \
  $EXTRA_BINDS \
  "$SIF" \
  bash -c '
    set -x
    cd /workspace/verl
    bash '"$TRAIN_SCRIPT"' "$@"
  ' bash "$@"
