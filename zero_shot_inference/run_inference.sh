#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Zero-shot inference pipeline — EATD-Corpus and MVSA.
#
# Speed strategy:
#   • Flash Attention 2 (auto-detected in inference.py, falls back to SDPA)
#   • Each GPU runs one shard of the dataset in parallel (data parallelism)
#   • Shards are merged after all GPUs finish
#   • batch_size > 1 for image-only datasets (MVSA) to amortise GPU overhead
#
# Steps:
#   1. Prepare per-dataset JSONLs (skipped if already exist)
#   2. Run inference: one shard per GPU, launched in background in parallel
#   3. Wait for all shards, then merge and compute final metrics
#
# Usage:
#   bash run_inference.sh
#
# Edit the CONFIG section below before running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODELS=(
    "PhilipC/HumanOmniV2"
    "ddvd233/OmniSapiens-7B-RL"
)

# Directory where prediction JSONLs and metrics are written
OUTPUT_DIR="/scratch/keane/zero_shot_outputs"     # <-- change before running

# Batch size per GPU for each dataset type.
# Audio (EATD): keep at 1 (variable-length audio padding can OOM at bs>1)
# Image (MVSA): 4–8 is usually safe on a 40 GB GPU
BATCH_SIZE_AUDIO=1
BATCH_SIZE_IMAGE=4

# Max tokens the model may generate per sample
MAX_NEW_TOKENS=1024

# Set to "1" to enable torch.compile (first batch will be slow)
TORCH_COMPILE=0

# Optional: cap samples per dataset for a quick smoke-test (empty = full run)
MAX_SAMPLES=""   # e.g. "20"

# Extra flags forwarded to inference.py (e.g. "--no_thinking")
EXTRA_ARGS=""

# ── END CONFIG ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/zero_shot_data"
INFERENCE="$SCRIPT_DIR/inference.py"

# ── GPU detection ─────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=1
fi
echo "Detected $NUM_GPUS GPU(s)"

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
    echo "  GPUs    : $NUM_GPUS   |   batch_size: $batch_size"
    echo "============================================================"

    local pids=()

    for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
        local shard_out="${out_base}_shard${gpu}.jsonl"
        echo "  [GPU $gpu] shard $gpu/$NUM_GPUS → $shard_out"

        CUDA_VISIBLE_DEVICES=$gpu python "$INFERENCE" \
            --model           "$model" \
            --input_jsonl     "$input_jsonl" \
            --output_jsonl    "$shard_out" \
            --data_base_dir   "$REPO_ROOT" \
            --batch_size      "$batch_size" \
            --max_new_tokens  "$MAX_NEW_TOKENS" \
            --num_shards      "$NUM_GPUS" \
            --shard_idx       "$gpu" \
            $(maybe_max_samples) \
            $(compile_flag) \
            $EXTRA_ARGS \
            &> "$OUTPUT_DIR/${model_slug}_${dataset_name}_shard${gpu}.log" &

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

# ── Step 1: Prepare dataset JSONLs ───────────────────────────────────────────

EATD_JSONL="$DATA_DIR/test_eatd_prompts.jsonl"
MVSA_JSONL="$DATA_DIR/test_mvsa_prompts.jsonl"

if [[ ! -f "$EATD_JSONL" ]]; then
    echo "[prepare] Generating EATD JSONL..."
    python "$SCRIPT_DIR/prepare_eatd.py" \
        --data_dir "$DATA_DIR/EATD-Corpus" \
        --output   "$EATD_JSONL"
else
    echo "[prepare] EATD JSONL: $EATD_JSONL (already exists)"
fi

if [[ ! -f "$MVSA_JSONL" ]]; then
    echo "[prepare] Generating MVSA JSONL..."
    python "$SCRIPT_DIR/prepare_mvsa.py" \
        --data_dir "$DATA_DIR/MVSA" \
        --output   "$MVSA_JSONL"
else
    echo "[prepare] MVSA JSONL: $MVSA_JSONL (already exists)"
fi

mkdir -p "$OUTPUT_DIR"

# ── Step 2: Inference ─────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
    run_parallel "$MODEL" "eatd" "$EATD_JSONL" "$BATCH_SIZE_AUDIO"
    run_parallel "$MODEL" "mvsa" "$MVSA_JSONL" "$BATCH_SIZE_IMAGE"
done

echo ""
echo "All done.  Results in: $OUTPUT_DIR"
