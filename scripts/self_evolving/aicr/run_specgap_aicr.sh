#!/usr/bin/env bash
# Run a spec-gap arm on an AICR compute node, inside the apptainer SIF.
#
# WHY A WRAPPER. AICR differs from the MSR pods in two ways that matter:
#
#   1. There is no docker and no root, so everything runs inside a SIF. That SIF must be
#      built from the SAME image the MSR pods run -- docker://zjdavid/verl-selfevolving:cu130,
#      per docker/kub_config/config.yaml -- because the cu130-vllm0.22.1 tag in the same
#      repo was tried on this stack and rolled back. Building from the wrong tag is easy
#      and quiet, so the version is asserted below against the container itself rather
#      than trusted from the filename.
#
#      Matching the image also removes the comparability question: with the same vllm as
#      MSR there is no rollout-sampling difference to argue about. Step-0 val remains a
#      free sanity check -- every MSR arm starts at 0.2405-0.2759 from identical base
#      weights, so a value in that band confirms the environments agree.
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
# Built from docker://zjdavid/verl-selfevolving:cu130 -- the SAME image the MSR pods run
# (docker/kub_config/config.yaml). There is also a cu130-vllm0.22.1 tag in that repo; it
# was TRIED AND ROLLED BACK, and the first AICR SIF was unfortunately built from it.
SIF="${SIF:-$S/sif/verl-selfevolving-cu130.sif}"
REPO_HOST="$S/verl_specgap"
ARM="${ARM:?set ARM=1..12 (see launch_specgap_when_free.sh; 11/12 = the no-retrieval adversary/fixed pair)}"
EXP_SUFFIX="${EXP_SUFFIX:-_aicr}"
# The version the MSR pods run, and the only one this stack is known to work on.
EXPECT_VLLM="${EXPECT_VLLM:-0.20}"

[ -f "$SIF" ] || { echo "FATAL: SIF missing at $SIF" >&2; exit 1; }

# Check the vLLM the container ACTUALLY has, not what its filename claims. A rolled-back
# image is a silent-failure risk, and a filename is not evidence -- this is the check that
# would have caught running on 0.22.1.
_got=$(apptainer exec "$SIF" python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null | tr -d '\r')
case "$_got" in
    "$EXPECT_VLLM"*) echo "vllm $_got in $(basename "$SIF") -- matches MSR" ;;
    "") echo "FATAL: could not read vllm version from $SIF" >&2; exit 1 ;;
    *)  echo "FATAL: $(basename "$SIF") has vllm $_got, expected ${EXPECT_VLLM}.x." >&2
        echo "       vllm 0.22.1 was tried on this stack and rolled back; MSR pods run" >&2
        echo "       zjdavid/verl-selfevolving:cu130. Rebuild with:" >&2
        echo "         apptainer build $S/sif/verl-selfevolving-cu130.sif \\" >&2
        echo "           docker://zjdavid/verl-selfevolving:cu130" >&2
        echo "       Override deliberately with EXPECT_VLLM=$_got if you mean it." >&2
        exit 1 ;;
esac
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
    --env "GEN_WORKERS=${GEN_WORKERS:-8}" \
    --env "GEN_WARMUP_S=${GEN_WARMUP_S:-5400}" \
    "$SIF" \
    bash -lc "cd $MSR/verl_specgap && bash scripts/self_evolving/train/launch_specgap_when_free.sh"
