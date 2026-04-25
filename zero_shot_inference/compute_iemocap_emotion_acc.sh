#!/usr/bin/env bash
# Compute IEMOCAP emotion mean accuracy for each model's merged predictions JSONL.
# Results are saved alongside each JSONL and (optionally) logged to W&B.
#
# Edit the CONFIG section, then run:
#   bash compute_iemocap_emotion_acc.sh
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODELS=(
    # "keentomato/harpo_hier_step400"
    # "PhilipC/HumanOmniV2"
    # "ddvd233/OmniSapiens-7B-RL"
    # "Qwen/Qwen2.5-Omni-7B"
    "google/gemma-4-e4b-it"
)

# Directory containing the merged prediction JSONLs produced by run_inference.sh.
# Expected filename pattern: {model_slug}_iemocap_merged.jsonl
RESULTS_DIR="/home/keaneong/human-behavior/verl/zero_shot_inference/results/thinking_w_params"

# W&B config — must match the values used during run_inference.sh so that
# emotion accuracy metrics land in the same W&B runs as the original inference.
# Set WANDB_PROJECT="" to disable W&B logging entirely.
WANDB_PROJECT="zero-shot-inference"
WANDB_TAG="think_w_params"   # must match the tag used in run_inference.sh
WANDB_ENTITY=""               # W&B entity (org/team); empty = default

# ── END CONFIG ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPUTE_SCRIPT="$SCRIPT_DIR/data_utils/compute_iemocap_emotion_acc.py"

build_wandb_args() {
    local run_id="$1"
    if [[ -z "$WANDB_PROJECT" ]]; then
        echo ""
        return
    fi
    local args="--wandb_project $WANDB_PROJECT --wandb_run_id $run_id"
    [[ -n "$WANDB_ENTITY" ]] && args+=" --wandb_entity $WANDB_ENTITY"
    echo "$args"
}

for MODEL in "${MODELS[@]}"; do
    model_slug="${MODEL//\//_}"
    jsonl="${RESULTS_DIR}/${model_slug}_iemocap_merged.jsonl"

    if [[ ! -f "$jsonl" ]]; then
        echo "[SKIP] Missing predictions file: $jsonl"
        continue
    fi

    # Reproduce the run ID used by run_inference.sh so metrics land in the same run.
    run_id="${WANDB_TAG:+${WANDB_TAG}_}${model_slug}"

    echo ""
    echo "============================================================"
    echo "  Model : $MODEL"
    echo "  File  : $jsonl"
    [[ -n "$WANDB_PROJECT" ]] && echo "  W&B   : $WANDB_PROJECT / $run_id"
    echo "============================================================"

    # shellcheck disable=SC2046
    python "$COMPUTE_SCRIPT" \
        --predictions_jsonl "$jsonl" \
        --model             "$MODEL" \
        $(build_wandb_args "$run_id")
done

echo ""
echo "Done."
