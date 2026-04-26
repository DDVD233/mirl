#!/usr/bin/env bash
# Hardcoded wrapper for build_wandb_results.py.
#
# Edit the CONFIG block, then run:
#   bash verl/zero_shot_inference/wandb/build_wandb_results.sh

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Choose: "inference" or "reasoning"
TASK="reasoning"

# W&B location. Leave WANDB_ENTITY empty to use your default W&B entity.
WANDB_ENTITY=""
INFERENCE_PROJECT="zero-shot-inference"
REASONING_PROJECT="reasoning-evaluation"

# Models to include. These should match config.model in each W&B run.
# Leave empty if you prefer selecting only by RUN_NAMES or RUN_IDS.
MODELS=(
    # "PhilipC/HumanOmniV2"
    # "ddvd233/OmniSapiens-7B-RL"
    # "Qwen/Qwen2.5-Omni-7B"
    # "google/gemma-4-e4b-it"
    # "keentomato/harpo_hier_step400"
)

# Dataset order in the JSON/table. Remove datasets you do not want shown.
DATASETS=(
    "eatd"
    "mvsa"
    "av-asd"
    "iemocap"
    "sarcnet"
    "overall"
)

# Keep only the newest matching run for each config.model.
LATEST_PER_MODEL=1

# Optional filters. Leave empty to disable.
RUN_IDS=(
    # "abc123"
)

# Exact W&B run names/display names to extract.
RUN_NAMES=(
    "Qwen_Qwen2.5-Omni-7B_2026-04-24_10-11-02"
    "google_gemma-4-e4b-it_2026-04-25_21-37-30"
    "ddvd233_OmniSapiens-7B-RL_2026-04-24_10-11-02"
    "PhilipC_HumanOmniV2_2026-04-24_10-11-02"
    "keentomato_harpo_hier_step400_2026-04-23_08-41-45"
)

# Substring run-name/id/model filters.
RUN_NAME_CONTAINS=(
    # "think_w_params"
)

# Output location.
OUTPUT_DIR="/Users/keane/Desktop/research/human-behavior/verl/zero_shot_inference/results"

# Optional model display aliases for the TeX table. Format: OLD=NEW
MODEL_ALIASES=(
    # "PhilipC/HumanOmniV2=HumanOmniV2"
    # "ddvd233/OmniSapiens-7B-RL=OmniSapiens-7B-RL"
    # "Qwen/Qwen2.5-Omni-7B=Qwen2.5-Omni-7B"
    # "google/gemma-4-e4b-it=Gemma-4-E4B-IT"
)

# Set to 1 to render rates like 0.723 as 72.3.
PERCENT=1

# Table labels.
INFERENCE_CAPTION="Zero-shot inference results."
INFERENCE_LABEL="tab:zero-shot-inference"
REASONING_CAPTION="Reasoning evaluation results."
REASONING_LABEL="tab:reasoning-evaluation"

# Metrics to show in the table. JSON keeps all dataset/metric summary keys unless
# JSON_METRICS is set below.
INFERENCE_TABLE_METRICS=(
    "accuracy"
    "weighted_f1"
)
REASONING_TABLE_METRICS=(
    # "reasoning_accuracy"
    # "reasoning_weighted_f1"
    # "direct_accuracy"
    # "direct_weighted_f1"
    "self_consistency_rate"
    "self_consistency_correct"
    "self_consistency_incorrect"
    "para_consistency_rate"
    "para_consistency_correct"
    "para_consistency_incorrect"
    # "para_accuracy"
    # "para_weighted_f1"
    "mean_reasoning_tokens"
)

# Optional metric filter for the JSON. Leave empty to keep all W&B summary keys
# that match {dataset}/{metric}.
JSON_METRICS=(
    # "accuracy"
    # "weighted_f1"
)

# ── END CONFIG ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/build_wandb_results.py"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR"
fi

case "$TASK" in
    inference)
        PROJECT="$INFERENCE_PROJECT"
        OUTPUT_JSON="$OUTPUT_DIR/zero_shot_results.json"
        OUTPUT_TEX="$OUTPUT_DIR/zero_shot_results.tex"
        CAPTION="$INFERENCE_CAPTION"
        LABEL="$INFERENCE_LABEL"
        TABLE_METRICS=("${INFERENCE_TABLE_METRICS[@]}")
        ;;
    reasoning)
        PROJECT="$REASONING_PROJECT"
        OUTPUT_JSON="$OUTPUT_DIR/reasoning_results.json"
        OUTPUT_TEX="$OUTPUT_DIR/reasoning_results.tex"
        CAPTION="$REASONING_CAPTION"
        LABEL="$REASONING_LABEL"
        TABLE_METRICS=("${REASONING_TABLE_METRICS[@]}")
        ;;
    *)
        echo "[ERROR] TASK must be 'inference' or 'reasoning', got: $TASK" >&2
        exit 1
        ;;
esac

args=(
    "$PY_SCRIPT"
    --project "$PROJECT"
    --task "$TASK"
    --datasets "${DATASETS[@]}"
    --table_metrics "${TABLE_METRICS[@]}"
    --output_json "$OUTPUT_JSON"
    --output_tex "$OUTPUT_TEX"
    --caption "$CAPTION"
    --label "$LABEL"
)

if [[ -n "$WANDB_ENTITY" ]]; then
    args+=(--entity "$WANDB_ENTITY")
fi

if [[ "$LATEST_PER_MODEL" == "1" ]]; then
    args+=(--latest_per_model)
fi

if [[ "${#MODELS[@]}" -gt 0 ]]; then
    args+=(--models "${MODELS[@]}")
fi

if [[ "${#RUN_IDS[@]}" -gt 0 ]]; then
    args+=(--run_ids "${RUN_IDS[@]}")
fi

if [[ "${#RUN_NAMES[@]}" -gt 0 ]]; then
    args+=(--run_names "${RUN_NAMES[@]}")
fi

if [[ "${#RUN_NAME_CONTAINS[@]}" -gt 0 ]]; then
    args+=(--run_name_contains "${RUN_NAME_CONTAINS[@]}")
fi

if [[ "${#MODEL_ALIASES[@]}" -gt 0 ]]; then
    args+=(--model_alias "${MODEL_ALIASES[@]}")
fi

if [[ "${#JSON_METRICS[@]}" -gt 0 ]]; then
    args+=(--metrics "${JSON_METRICS[@]}")
fi

if [[ "$PERCENT" == "1" ]]; then
    args+=(--percent)
else
    args+=(--no-percent)
fi

mkdir -p "$OUTPUT_DIR"

echo "Task       : $TASK"
echo "W&B project: $PROJECT"
echo "Output JSON: $OUTPUT_JSON"
echo "Output TeX : $OUTPUT_TEX"

python "${args[@]}"
