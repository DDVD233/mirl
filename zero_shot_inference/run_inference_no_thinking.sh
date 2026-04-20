#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Zero-shot inference pipeline — no-thinking variant (direct answers, no <think>).
# Identical to run_inference.sh except:
#   • EXTRA_ARGS includes --no_thinking
#   • OUTPUT_DIR is under results/no_thinking/ to avoid overwriting thinking results
#
# Usage:
#   bash prepare_data.sh        # once, to build the EATD/MVSA input JSONLs
#   bash run_inference_no_thinking.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODELS=(
    "keentomato/harpo_hier_step400"
    "PhilipC/HumanOmniV2"
    "ddvd233/OmniSapiens-7B-RL"
)

# Directory where prediction JSONLs and metrics are written
OUTPUT_DIR="/home/keaneong/human-behavior/verl/zero_shot_inference/results/no_thinking"

# Batch size per GPU for each dataset type.
BATCH_SIZE_AUDIO=4
BATCH_SIZE_IMAGE=16
BATCH_SIZE_VIDEO=4

# Max tokens the model may generate per sample
MAX_NEW_TOKENS=512

# Set to "1" to enable torch.compile (first batch will be slow)
TORCH_COMPILE=0

# Optional: cap samples per dataset for a quick smoke-test (empty = full run)
MAX_SAMPLES="5"   # e.g. "20"

# No-thinking mode: model answers directly without <think> tags
EXTRA_ARGS="--no_thinking"

# GPUs to use. Leave empty to auto-detect all available GPUs.
GPUS=(0 1 2)

# ── W&B CONFIG ───────────────────────────────────────────────────────────────
WANDB_PROJECT="zero-shot-inference"   # W&B project name  (empty = disabled)
WANDB_TAG=""                          # optional tag prepended to run name: "{tag}_{model_slug}_{timestamp}"
WANDB_RUN_NAME=""                     # override full run name (ignores WANDB_TAG if set)
WANDB_ENTITY=""                       # W&B entity (org/team); empty = default

# ── END CONFIG ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/scratch/keane/hb_generalization_data/MVSA_EATD_zeroshot"
INFERENCE="$SCRIPT_DIR/inference.py"

RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="$OUTPUT_DIR/logs/$RUN_TIMESTAMP"

# ── GPU detection ─────────────────────────────────────────────────────────────
if [[ "${#GPUS[@]}" -gt 0 ]]; then
    NUM_GPUS="${#GPUS[@]}"
    echo "Using specified GPU(s): ${GPUS[*]}"
elif command -v nvidia-smi &>/dev/null; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
    NUM_GPUS="${#GPUS[@]}"
    echo "Detected $NUM_GPUS GPU(s): ${GPUS[*]}"
else
    GPUS=(0)
    NUM_GPUS=1
    echo "No nvidia-smi found, defaulting to GPU 0"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "Logs → $LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────

maybe_max_samples() {
    if [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]]; then
        echo "--max_samples $MAX_SAMPLES"
    fi
}

compile_flag() {
    [[ "$TORCH_COMPILE" == "1" ]] && echo "--torch_compile" || echo ""
}

build_wandb_args() {
    local dataset_name="$1"
    if [[ -z "$WANDB_PROJECT" ]]; then
        echo ""
        return
    fi
    local args="--wandb_project $WANDB_PROJECT --wandb_run_id $CURRENT_WANDB_RUN_ID --wandb_run_name $CURRENT_WANDB_RUN_NAME"
    [[ -n "$WANDB_ENTITY" ]] && args+=" --wandb_entity $WANDB_ENTITY"
    echo "$args"
}

run_parallel() {
    local model="$1"
    local dataset_name="$2"
    local input_jsonl="$3"
    local batch_size="$4"
    local dataset_extra_args="${5:-}"
    local model_slug="${model//\//_}"
    local out_base="$OUTPUT_DIR/${model_slug}_${dataset_name}"

    echo ""
    echo "============================================================"
    echo "  Model   : $model"
    echo "  Dataset : $dataset_name"
    echo "  GPUs    : ${GPUS[*]}   |   batch_size: $batch_size"
    echo "============================================================"

    local pids=()

    for (( shard=0; shard<NUM_GPUS; shard++ )); do
        local gpu="${GPUS[$shard]}"
        local shard_out="${out_base}_shard${shard}.jsonl"
        echo "  [GPU $gpu] shard $shard/$NUM_GPUS → $shard_out"

        CUDA_VISIBLE_DEVICES=$gpu python "$INFERENCE" \
            --model           "$model" \
            --input_jsonl     "$input_jsonl" \
            --output_jsonl    "$shard_out" \
            --batch_size      "$batch_size" \
            --max_new_tokens  "$MAX_NEW_TOKENS" \
            --num_shards      "$NUM_GPUS" \
            --shard_idx       "$shard" \
            $(maybe_max_samples) \
            $(compile_flag) \
            $EXTRA_ARGS \
            $dataset_extra_args \
            &> "$LOG_DIR/${model_slug}_${dataset_name}_shard${shard}.log" &

        pids+=($!)
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed + 1))
        fi
    done

    if [[ "$failed" -gt 0 ]]; then
        echo "[ERROR] $failed shard(s) failed for $model / $dataset_name — check logs in $LOG_DIR"
        return 1
    fi

    local merged_out="${out_base}_merged.jsonl"
    echo "  Merging shards → $merged_out"
    python "$INFERENCE" \
        --merge_shards "${out_base}_shard*.jsonl" \
        --output_jsonl "$merged_out" \
        --model        "$model" \
        $EXTRA_ARGS \
        $dataset_extra_args \
        $(build_wandb_args "$dataset_name")

    rm -f "${out_base}_shard"*.jsonl
    echo "  Shard files deleted."
}

# ── Verify input JSONLs exist ────────────────────────────────────────────────

EATD_JSONL="$DATA_DIR/test_eatd_prompts.jsonl"
MVSA_JSONL="$DATA_DIR/test_mvsa_prompts.jsonl"
AVASD_JSONL="/scratch/keane/hb_generalization_data/avasd/test_av_asd_promptsmultilabel.jsonl"
IEMOCAP_JSONL="/scratch/keane/hb_generalization_data/iemocap/latest_iemocap_test.jsonl"

for jsonl in "$EATD_JSONL" "$MVSA_JSONL" "$AVASD_JSONL" "$IEMOCAP_JSONL"; do
    if [[ ! -f "$jsonl" ]]; then
        echo "[ERROR] Missing input JSONL: $jsonl"
        echo "        Run prepare_data.sh first (for EATD/MVSA) or check server paths."
        exit 1
    fi
done

# ── Inference ─────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
    local _slug="${MODEL//\//_}"
    local _default_name="${WANDB_TAG:+${WANDB_TAG}_}${_slug}_nothink_${RUN_TIMESTAMP}"
    CURRENT_WANDB_RUN_ID="${_default_name}"
    CURRENT_WANDB_RUN_NAME="${WANDB_RUN_NAME:-${_default_name}}"

    run_parallel "$MODEL" "eatd"    "$EATD_JSONL"    "$BATCH_SIZE_AUDIO"
    run_parallel "$MODEL" "mvsa"    "$MVSA_JSONL"    "$BATCH_SIZE_IMAGE"
    run_parallel "$MODEL" "av-asd"  "$AVASD_JSONL"   "$BATCH_SIZE_VIDEO" "--multilabel"
    run_parallel "$MODEL" "iemocap" "$IEMOCAP_JSONL"  "$BATCH_SIZE_VIDEO"
done

echo ""
echo "All done. Results in: $OUTPUT_DIR"
echo "Logs in:   $LOG_DIR"
