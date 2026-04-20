#!/usr/bin/env bash
# ---------------------------------------------------------------
# Merge verl FSDP checkpoint -> HuggingFace format + upload to Hub
# Run on the training cluster from inside the verl repo root.
# Requires: huggingface-cli login  (or export HF_TOKEN=...)
# ---------------------------------------------------------------
set -euo pipefail

# ── CONFIGURE THESE ────────────────────────────────────────────
ACTOR_DIR="/scratch/keane/human_behaviour/HARPO_Hier_Ema_Rerun/global_step_400/actor"
TARGET_DIR="/scratch/keane/human_behaviour/HARPO_Hier_Ema_Rerun/merged_hf"
HF_REPO="<your-hf-username>/HARPO-Hier-Ema-step400"
PRIVATE="--private"   # remove for a public repo
BASE_MODEL="Qwen/Qwen2.5-Omni-7B"
USE_CPU_INIT="true"   # set to "false" to use GPU for model init
# ───────────────────────────────────────────────────────────────

echo "=== Step 1: Ensure huggingface/ config subdir exists ==="
HF_SUBDIR="${ACTOR_DIR}/huggingface"
if [ ! -d "${HF_SUBDIR}" ]; then
    echo "  huggingface/ not found — copying config/tokenizer from ${BASE_MODEL}"
    python - <<PYEOF
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
import os

base = "${BASE_MODEL}"
out  = "${HF_SUBDIR}"
os.makedirs(out, exist_ok=True)
AutoConfig.from_pretrained(base).save_pretrained(out)
try:
    AutoProcessor.from_pretrained(base).save_pretrained(out)
    print(f"  Saved processor + config to {out}")
except Exception:
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print(f"  Saved tokenizer + config to {out}")
PYEOF
else
    echo "  huggingface/ already present — skipping."
fi

echo ""
echo "=== Step 2: Merge shards and upload ==="
mkdir -p "${TARGET_DIR}"

CPU_INIT_FLAG=""
[ "${USE_CPU_INIT}" = "true" ] && CPU_INIT_FLAG="--use_cpu_initialization"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  "${ACTOR_DIR}" \
    --target_dir "${TARGET_DIR}" \
    --hf_upload_path "${HF_REPO}" \
    ${CPU_INIT_FLAG} \
    ${PRIVATE}

echo ""
echo "=== Done ==="
echo "Local merged model : ${TARGET_DIR}"
echo "HuggingFace repo   : https://huggingface.co/${HF_REPO}"
