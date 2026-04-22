#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Zero-shot inference pipeline — EATD-Corpus, MVSA, AV-ASD, IEMOCAP.
# Requires dataset JSONLs to be prepared first (run prepare_data.sh for EATD/MVSA).
#
# Speed strategy:
#   • Flash Attention 2 (auto-detected in inference.py, falls back to SDPA)
#   • Each GPU runs one shard of the dataset in parallel (data parallelism)
#   • Shards are merged after all GPUs finish, then shard files are deleted
#   • batch_size > 1 for image-only datasets (MVSA) to amortise GPU overhead
#
# Usage:
#   bash prepare_data.sh   # once, to build the EATD/MVSA input JSONLs
#   bash run_inference.sh  # runs inference and merges results
#
# Edit the CONFIG section below before running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODELS=(
    "keentomato/harpo_hier_step400"
    "PhilipC/HumanOmniV2"
    "ddvd233/OmniSapiens-7B-RL"
    "Qwen/Qwen2.5-Omni-7B"
    # "google/gemma-4-e4b-it"
)

# Sampling parameters for generation.
# Thinking mode defaults: Temperature=0.6, TopP=0.95, TopK=20, MinP=0 (recommended).
# Set TEMPERATURE=0 (or empty) to fall back to greedy decoding.
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_P=0

# For GEMMA
# temperature=1.0
# top_p=0.95
# top_k=64
# Directory where prediction JSONLs and metrics are written

OUTPUT_DIR="/home/keaneong/human-behavior/verl/zero_shot_inference/results/thinking_w_params"

# Batch size per GPU for each dataset type.
# Audio (EATD): keep at 1–4 (variable-length audio padding can OOM at bs>1)
# Image (MVSA): 4–16 is usually safe on a 40 GB GPU
# Video+Audio (AV-ASD, IEMOCAP): keep at 1 (memory-intensive)
BATCH_SIZE_AUDIO=4
BATCH_SIZE_IMAGE=16
BATCH_SIZE_VIDEO=4

# Max tokens the model may generate per sample
MAX_NEW_TOKENS=512

# Set to "1" to enable torch.compile (first batch will be slow)
TORCH_COMPILE=0

# Optional: cap samples per dataset for a quick smoke-test (empty = full run)
MAX_SAMPLES=""   # e.g. "20"

# Data loading mode:
#   "verl_style" — uses qwen_vl_utils + torchaudio (matches harpo_hier / omnisapiens training)
#   "default"    — uses decord + soundfile (compatible with HumanOmniV2 and others)
DATA_LOADING="verl_style"

# Per-model overrides for DATA_LOADING. Models not listed use DATA_LOADING above.
# Gemma requires "default" to avoid Qwen-specific qwen_vl_utils video processing.
declare -A MODEL_DATA_LOADING=(
    ["PhilipC/HumanOmniV2"]="default"
    ["google/gemma-4-e4b-it"]="default"
)

# Datasets to run. Remove or comment out any you want to skip.
# Available: eatd  mvsa  av-asd  iemocap  dreaddit  sarcnet
DATASETS=(
    # "av-asd"
    # "iemocap"
    # "eatd"
    # "mvsa"
    "dreaddit"
    "sarcnet"
)

# ── DATASET JSONL PATHS ───────────────────────────────────────────────────────
EATD_JSONL="/scratch/keane/hb_generalization_data/MVSA_EATD_zeroshot/test_eatd_prompts.jsonl"
MVSA_JSONL="/scratch/keane/hb_generalization_data/MVSA_EATD_zeroshot/test_mvsa_prompts.jsonl"
AVASD_JSONL="/scratch/keane/hb_generalization_data/avasd/test_av_asd_promptsmultilabel.jsonl"
IEMOCAP_JSONL="/scratch/keane/hb_generalization_data/iemocap/latest_iemocap_test.jsonl"
DREADDIT_JSONL="/scratch/keane/hb_generalization_data/zero_shot_data_v2/test_dreaddit_prompts.jsonl"
SARCNET_JSONL="/scratch/keane/hb_generalization_data/zero_shot_data_v2/test_sarcnet_prompts.jsonl"

# Extra flags forwarded to inference.py for ALL datasets (e.g. "--no_thinking")
EXTRA_ARGS=""

# GPUs to use. Leave empty to auto-detect all available GPUs.
# Example: GPUS=(0 1)  or  GPUS=(2 3 4 5)
GPUS=(2 4)

# ── W&B CONFIG ───────────────────────────────────────────────────────────────
# Set WANDB_PROJECT to "" to disable W&B logging entirely.
# All datasets for a single model are logged into one W&B run (resume="allow"),
# with metrics keyed as {dataset}/accuracy, {dataset}/weighted_f1, etc.

WANDB_PROJECT="zero-shot-inference"   # W&B project name  (empty = disabled)
WANDB_TAG="think_w_params"                          # optional tag prepended to run name: "{tag}_{model_slug}_{timestamp}"
WANDB_RUN_NAME=""                     # override full run name (ignores WANDB_TAG if set)
WANDB_ENTITY=""                       # W&B entity (org/team); empty = default

# ── END CONFIG ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE="$SCRIPT_DIR/inference.py"

# Timestamp used for both log directory naming and W&B run IDs
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

sampling_args() {
    if [[ -n "$TEMPERATURE" && "$TEMPERATURE" != "0" ]]; then
        echo "--temperature $TEMPERATURE --top_p $TOP_P --top_k $TOP_K --min_p $MIN_P"
    fi
}

# Build W&B args string for the merge step (empty if W&B is disabled).
# Uses globals: WANDB_PROJECT, WANDB_ENTITY, CURRENT_WANDB_RUN_ID, CURRENT_WANDB_RUN_NAME
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

# Run all shards for one (model, dataset, batch_size) combination in parallel.
# Blocks until all shards finish, then merges them and deletes shard JSONLs.
#
# Args:
#   $1  model name
#   $2  dataset name (used in file names)
#   $3  input JSONL path
#   $4  batch size
#   $5  (optional) dataset-specific extra args, e.g. "--multilabel"
run_parallel() {
    local model="$1"
    local dataset_name="$2"
    local input_jsonl="$3"
    local batch_size="$4"
    local dataset_extra_args="${5:-}"
    local model_slug="${model//\//_}"
    local out_base="$OUTPUT_DIR/${model_slug}_${dataset_name}"
    local effective_dl="${MODEL_DATA_LOADING[$model]:-$DATA_LOADING}"

    echo ""
    echo "============================================================"
    echo "  Model   : $model"
    echo "  Dataset : $dataset_name"
    echo "  GPUs    : ${GPUS[*]}   |   batch_size: $batch_size   |   data_loading: $effective_dl"
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
            --data_loading    "$effective_dl" \
            $(maybe_max_samples) \
            $(compile_flag) \
            $(sampling_args) \
            $EXTRA_ARGS \
            $dataset_extra_args \
            &> "$LOG_DIR/${model_slug}_${dataset_name}_shard${shard}.log" &

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
        echo "[ERROR] $failed shard(s) failed for $model / $dataset_name — check logs in $LOG_DIR"
        return 1
    fi

    # ── Merge shards ─────────────────────────────────────────────────────────
    local merged_out="${out_base}_merged.jsonl"
    echo "  Merging shards → $merged_out"
    python "$INFERENCE" \
        --merge_shards "${out_base}_shard*.jsonl" \
        --output_jsonl "$merged_out" \
        --model        "$model" \
        $EXTRA_ARGS \
        $dataset_extra_args \
        $(build_wandb_args "$dataset_name")

    # ── Delete shard JSONLs (logs are kept) ──────────────────────────────────
    rm -f "${out_base}_shard"*.jsonl
    echo "  Shard files deleted."
}

# ── Dataset verification ──────────────────────────────────────────────────────
# Verify only the JSONLs that are actually needed
for ds in "${DATASETS[@]}"; do
    case "$ds" in
        eatd)     [[ -f "$EATD_JSONL"     ]] || { echo "[ERROR] Missing: $EATD_JSONL";     exit 1; } ;;
        mvsa)     [[ -f "$MVSA_JSONL"     ]] || { echo "[ERROR] Missing: $MVSA_JSONL";     exit 1; } ;;
        av-asd)   [[ -f "$AVASD_JSONL"   ]] || { echo "[ERROR] Missing: $AVASD_JSONL";   exit 1; } ;;
        iemocap)  [[ -f "$IEMOCAP_JSONL"  ]] || { echo "[ERROR] Missing: $IEMOCAP_JSONL";  exit 1; } ;;
        dreaddit) [[ -f "$DREADDIT_JSONL" ]] || { echo "[ERROR] Missing: $DREADDIT_JSONL"; exit 1; } ;;
        sarcnet)  [[ -f "$SARCNET_JSONL"  ]] || { echo "[ERROR] Missing: $SARCNET_JSONL";  exit 1; } ;;
        *) echo "[ERROR] Unknown dataset '$ds'. Valid: eatd mvsa av-asd iemocap dreaddit sarcnet"; exit 1 ;;
    esac
done

# ── Inference ─────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
    # One W&B run ID per model — all datasets log into the same run via resume="allow"
    _slug="${MODEL//\//_}"
    _default_name="${WANDB_TAG:+${WANDB_TAG}_}${_slug}_${RUN_TIMESTAMP}"
    CURRENT_WANDB_RUN_ID="${_default_name}"
    CURRENT_WANDB_RUN_NAME="${WANDB_RUN_NAME:-${_default_name}}"

    for DATASET in "${DATASETS[@]}"; do
        case "$DATASET" in
            eatd)     run_parallel "$MODEL" "eatd"     "$EATD_JSONL"     "$BATCH_SIZE_AUDIO" ;;
            mvsa)     run_parallel "$MODEL" "mvsa"     "$MVSA_JSONL"     "$BATCH_SIZE_IMAGE" ;;
            av-asd)   run_parallel "$MODEL" "av-asd"   "$AVASD_JSONL"    "$BATCH_SIZE_VIDEO" "--multilabel" ;;
            iemocap)  run_parallel "$MODEL" "iemocap"  "$IEMOCAP_JSONL"  "$BATCH_SIZE_VIDEO" ;;
            dreaddit) run_parallel "$MODEL" "dreaddit" "$DREADDIT_JSONL" "$BATCH_SIZE_IMAGE" ;;
            sarcnet)  run_parallel "$MODEL" "sarcnet"  "$SARCNET_JSONL"  "$BATCH_SIZE_IMAGE" ;;
        esac
    done
done

echo ""
echo "All done. Results in: $OUTPUT_DIR"
echo "Logs in:   $LOG_DIR"
