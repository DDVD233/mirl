#!/usr/bin/env bash
# Run a spec-gap arm on an AICR compute node, inside the apptainer SIF.
#
# WHY A WRAPPER. AICR differs from the MSR pods in two ways that matter:
#
#   1. There is no docker and no root, so everything runs inside
#      verl-selfevolving-cu130-vllm0.22.1.sif. The MSR pods are NOT containerised at
#      all -- they use their image's system python, which carries vllm 0.20.2rc1. The
#      SIF is a separate, newer build. Absolute accuracies from an AICR run are
#      therefore NOT automatically comparable with MSR runs; step-0 val is the probe
#      (every MSR arm starts at 0.2405-0.2759 from identical base weights, so a value
#      inside that band means the two environments are sampling equivalently).
#
#   2. Paths. $S on AICR is bind-mounted onto the MSR path $MSR, so every hardcoded
#      /scratch/sheng/self_evolving in the launcher resolves to AICR scratch. The repo
#      lives at $S/verl_specgap, which is therefore visible inside the container at
#      exactly $MSR/verl_specgap -- the path the launcher already expects, so no extra
#      bind is needed.
#
# The judge for self-judge arms is server5's Qwen3.5-9B over frp, reachable from AICR
# (verified 200), so no GPU on this node is spent hosting it.
#
# Usage, from a node that holds an allocation (the chain sbatch holds one after its
# RUN_SCRIPT exits):
#   ARM=5 bash scripts/self_evolving/aicr/run_specgap_aicr.sh
set -euo pipefail

S=/scratch/dvdai_mit/self_evolving
MSR=/scratch/sheng/self_evolving
SIF="${SIF:-$S/sif/verl-selfevolving-cu130-vllm0.22.1.sif}"
REPO_HOST="$S/verl_specgap"
ARM="${ARM:?set ARM=1..6}"
EXP_SUFFIX="${EXP_SUFFIX:-_aicr}"

[ -f "$SIF" ] || { echo "FATAL: SIF missing at $SIF" >&2; exit 1; }
[ -d "$REPO_HOST/.git" ] || { echo "FATAL: repo missing at $REPO_HOST" >&2; exit 1; }
[ -f "$S/healthbench_pro_val.parquet" ] || { echo "FATAL: val parquet missing" >&2; exit 1; }
[ -f "$S/.trapi_key" ] || { echo "FATAL: .trapi_key missing" >&2; exit 1; }

# wandb: the launcher recovers the key from ~/.netrc, but ~ inside the container is the
# host home only if we bind it, so pass the key through explicitly instead.
WANDB_API_KEY="${WANDB_API_KEY:-$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' \
    "$HOME/.netrc" 2>/dev/null || true)}"
[ -n "$WANDB_API_KEY" ] || { echo "FATAL: no WANDB_API_KEY and none in ~/.netrc" >&2; exit 1; }

echo "=== ARM=$ARM suffix=$EXP_SUFFIX on $(hostname) (job ${SLURM_JOB_ID:-none}) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

exec apptainer exec --nv --writable-tmpfs \
    --bind "$S:$MSR" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "ARM=$ARM" \
    --env "EXP_SUFFIX=$EXP_SUFFIX" \
    --env "SJUDGE_REMOTE=${SJUDGE_REMOTE:-http://point.dd.works:18184/v1}" \
    --env "HB_REFINE_MODE=${HB_REFINE_MODE:-rewrite}" \
    "$SIF" \
    bash -lc "cd $MSR/verl_specgap && bash scripts/self_evolving/train/launch_specgap_when_free.sh"
