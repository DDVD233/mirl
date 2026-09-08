#!/bin/bash
#SBATCH --job-name=hb-eval
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/dvdai_mit/self_evolving/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvdai@mit.edu
#
# One-shot HealthBench Professional eval for the paper's Table 2 refresh
# (2026-08-11). Serves MODEL_ID with vLLM inside the cu130 SIF (the tag that
# works; NOT the vllm0.22.1 one) and grades with the OFFICIAL
# leaderboard-comparable grader: gpt-5.4_2026-03-05 at low effort via the
# TRAPI proxy. The old table grader (gpt-5.3-chat_2026-03-03) is gone from
# TRAPI, so ALL rows are being regenerated+regraded under gpt-5.4-low for
# consistency.
#
# Env knobs:
#   MODEL_ID   HF id or path of the model under test (required)
#   LABEL      short run label for the output dir (required)
#   TP         tensor parallel size (default 1; must match --gres)
#   THINKING   "off" to pass --disable-thinking (qwen-family), else omit
#   MODE       "vllm" (default) or "api" (TRAPI model under test; no GPU used)
#   MAX_MODEL_LEN  default 16384
#
# Not a chain job: runs once and exits. No wandb (results land as JSON).

set -euo pipefail

S=/scratch/dvdai_mit/self_evolving
WORK=/work/mit/ppliang_mit/dvdai
REPO=$WORK/mirl
SIF=$S/sif/verl-selfevolving-cu130.sif

MODEL_ID="${MODEL_ID:?set MODEL_ID}"
LABEL="${LABEL:?set LABEL}"
TP="${TP:-1}"
MODE="${MODE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# Completion budget. 2048 matches the historical sweep for open models; API reasoning
# models (gpt-5.6-sol) need far more or reasoning eats the budget and content is empty.
MAXTOK="${MAXTOK:-2048}"
PORT=$((20000 + SLURM_JOB_ID % 10000))

OUT=$S/logs/healthbench_pro_gpt54/$LABEL
mkdir -p "$OUT"

TRAPI_BASE=http://point.dd.works:18890/v1
IFS= read -r TRAPI_KEY < "$S/.trapi_key" || true
test -n "$TRAPI_KEY"

THINK_FLAG=""
if [[ "${THINKING:-}" == "off" ]]; then THINK_FLAG="--disable-thinking"; fi

export HF_HOME=$S/hf_cache
export HF_HUB_DISABLE_XET=1
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=$S/pyuser
mkdir -p "$PYTHONUSERBASE"
# Compile/tool caches go to node-local tmp, NOT $HOME: the 100 GiB home quota is shared
# with collaborators and concurrent jobs' torch-compile caches blew it on 2026-08-11
# ("Disk quota exceeded" killing vLLM startup).
JOBCACHE=/tmp/hbev_cache_${SLURM_JOB_ID}
mkdir -p "$JOBCACHE"
# Keep FlashInfer's package-local symlink setup off the bounded writable overlay.
EXTRA_BIND=""
if [[ ${FIX_FLASHINFER_CUBIN:-0} == 1 ]]; then
    CUBIN_PACKAGE=$(apptainer exec "$SIF" python -c \
        'import importlib.util; print(importlib.util.find_spec("flashinfer_cubin").submodule_search_locations[0])')
    apptainer exec --bind "$JOBCACHE:$JOBCACHE" "$SIF" \
        cp -a "$CUBIN_PACKAGE" "$JOBCACHE/flashinfer_cubin"
    EXTRA_BIND="--bind $JOBCACHE/flashinfer_cubin:$CUBIN_PACKAGE"
fi
APP="apptainer exec --nv --writable-tmpfs --bind /work:/work --bind /scratch:/scratch \
     $EXTRA_BIND \
     --env HF_HOME=$HF_HOME --env HF_HUB_DISABLE_XET=1 \
     --env PYTHONUSERBASE=$PYTHONUSERBASE \
     --env XDG_CACHE_HOME=$JOBCACHE --env TORCHINDUCTOR_CACHE_DIR=$JOBCACHE/inductor \
     --env VLLM_CACHE_ROOT=$JOBCACHE/vllm \
     --env NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS --env PYTHONUNBUFFERED=1 $SIF"

# The SIF is missing a couple of pure-python deps of the eval script; install
# once into a /scratch user site shared by all eval jobs.
$APP python - <<'EOF' || $APP python -m pip install --user --no-warn-script-location blobfile jinja2
import blobfile, jinja2
EOF

run_eval() {
    local model_base=$1 model_name=$2 provider=$3
    $APP python "$REPO/scripts/self_evolving/eval/healthbench_professional_eval.py" \
        --model-base "$model_base" \
        --model-name "$model_name" \
        --model-key "${4:-EMPTY}" \
        --model-provider "$provider" \
        --temperature 0.0 --max-tokens "$MAXTOK" $THINK_FLAG \
        --grader trapi \
        --grader-base "$TRAPI_BASE" \
        --grader-key "$TRAPI_KEY" \
        --grader-model gpt-5.4_2026-03-05 \
        --grader-effort low \
        --concurrency 16 \
        --simple-evals-dir "$S/simple_evals_src" \
        --output-dir "$OUT" \
        --no-wandb
}

if [[ "$MODE" == "api" ]]; then
    run_eval "$TRAPI_BASE" "$MODEL_ID" trapi "$TRAPI_KEY"
    exit $?
fi

# --- pre-download weights so download time never eats the health budget ----
# (first ours attempt died here: ~30 GB pulled at ~13 MB/s inside vllm's own
# loader, silently, until the 60-min health wait expired)
if [[ -e "$MODEL_ID" ]]; then
    echo "MODEL_ID is a local path, skipping download"
else
    $APP huggingface-cli download "$MODEL_ID" --exclude "reward_evolution/*" \
        || $APP python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"
fi

# --- serve the model under test -------------------------------------------
# Flags mirror aicr/run_teacher.sh, the known-good serve of this model family
# on this SIF (TORCH_SDPA avoids the B200 ViT cute-kernel crash).
$APP vllm serve "$MODEL_ID" \
    --trust-remote-code \
    --dtype auto \
    --mm-encoder-attn-backend TORCH_SDPA \
    --tensor-parallel-size "$TP" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    > "$OUT/vllm_serve.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 480); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM died during startup; tail of log:"; tail -50 "$OUT/vllm_serve.log"; exit 1
    fi
    sleep 15
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "vLLM never became healthy"; exit 1; }
echo "vLLM healthy on :$PORT after $((i*15))s"

run_eval "http://localhost:$PORT/v1" "$MODEL_ID" vllm
RC=$?
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
exit $RC
