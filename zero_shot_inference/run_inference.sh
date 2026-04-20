#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Zero-shot inference pipeline — EATD-Corpus and MVSA.
# Requires dataset JSONLs to be prepared first (run prepare_data.sh).
#
# Speed strategy:
#   • Flash Attention 2 (auto-detected in inference.py, falls back to SDPA)
#   • Each GPU runs one shard of the dataset in parallel (data parallelism)
#   • Shards are merged after all GPUs finish
#   • batch_size > 1 for image-only datasets (MVSA) to amortise GPU overhead
#
# Usage:
#   bash prepare_data.sh   # once, to build the input JSONLs
#   bash run_inference.sh  # runs inference and merges results
#
# Edit the CONFIG section below before running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODELS=(
    "PhilipC/HumanOmniV2"
)
       "ddvd233/OmniSapiens-7B-RL"
    # "ddvd233/OmniSapiens-7B-RL"


# Directory where prediction JSONLs and metrics are written
OUTPUT_DIR="/home/keaneong/human-behavior/verl/zero_shot_inference/results"

# Batch size per GPU for each dataset type.
# Audio (EATD): keep at 1 (variable-length audio padding can OOM at bs>1)
# Image (MVSA): 4–8 is usually safe on a 40 GB GPU
BATCH_SIZE_AUDIO=4
BATCH_SIZE_IMAGE=16

# Max tokens the model may generate per sample
MAX_NEW_TOKENS=512

# Set to "1" to enable torch.compile (first batch will be slow)
TORCH_COMPILE=0

# Optional: cap samples per dataset for a quick smoke-test (empty = full run)
MAX_SAMPLES=""   # e.g. "20"

# Extra flags forwarded to inference.py (e.g. "--no_thinking")
EXTRA_ARGS=""

# GPUs to use. Leave empty to auto-detect all available GPUs.
# Example: GPUS=(0 1)  or  GPUS=(2 3 4 5)
GPUS=(0 1)

# ── END CONFIG ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/scratch/keane/hb_generalization_data/MVSA_EATD_zeroshot"
INFERENCE="$SCRIPT_DIR/inference.py"

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

# ── Helpers ───────────────────────────────────────────────────────────────────

maybe_max_samples() {
    if [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]]; then
        echo "--max_samples $MAX_SAMPLES"
    fi
}

compile_flag() {
    [[ "$TORCH_COMPILE" == "1" ]] && echo "--torch_compile" || echo ""
}

# Run all shards for one (model, dataset, batch_size) combination in parallel.
# Blocks until all shards finish, then merges them.
run_parallel() {
    local model="$1"
    local dataset_name="$2"
    local input_jsonl="$3"
    local batch_size="$4"
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
            &> "$OUTPUT_DIR/${model_slug}_${dataset_name}_shard${shard}.log" &

        pids+=($!)
    done

    # Wait for all shards and collect exit codes
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed + 1))
        fi
    done

    if [[ "$failed" -gt 0 ]]; then
        echo "[ERROR] $failed shard(s) failed for $model / $dataset_name — check logs in $OUTPUT_DIR"
        return 1
    fi

    # ── Merge shards ─────────────────────────────────────────────────────────
    local merged_out="${out_base}_merged.jsonl"
    echo "  Merging shards → $merged_out"
    python "$INFERENCE" \
        --merge_shards "${out_base}_shard*.jsonl" \
        --output_jsonl "$merged_out" \
        --model        "$model"
}

# ── Verify input JSONLs exist ────────────────────────────────────────────────

EATD_JSONL="$DATA_DIR/test_eatd_prompts.jsonl"
MVSA_JSONL="$DATA_DIR/test_mvsa_prompts.jsonl"

for jsonl in "$EATD_JSONL" "$MVSA_JSONL"; do
    if [[ ! -f "$jsonl" ]]; then
        echo "[ERROR] Missing input JSONL: $jsonl"
        echo "        Run prepare_data.sh first."
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

# ── Inference ─────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
    run_parallel "$MODEL" "eatd" "$EATD_JSONL" "$BATCH_SIZE_AUDIO"
    run_parallel "$MODEL" "mvsa" "$MVSA_JSONL" "$BATCH_SIZE_IMAGE"
done

echo ""
echo "All done. Results in: $OUTPUT_DIR"
