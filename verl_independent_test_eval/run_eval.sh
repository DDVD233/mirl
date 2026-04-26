#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Test-set evaluation pipeline — single combined JSONL across all HB datasets.
#
# Speed strategy:
#   • Each GPU runs one shard of the JSONL in parallel (data parallelism)
#   • Shards are merged after all GPUs finish; shard files are then deleted
#   • After merge, unified per-dataset metrics are computed automatically
#
# Usage:
#   bash run_eval.sh
#
# Edit the CONFIG section below before running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME="google/gemma-4-e4b-it"
# MODEL_NAME="keentomato/harpo_hier_step400"
# MODEL_NAME="PhilipC/HumanOmniV2"
# MODEL_NAME="ddvd233/OmniSapiens-7B-RL"
# MODEL_NAME="Qwen/Qwen2.5-Omni-7B"

# Single JSONL combining all test datasets.
# Each line must have: problem, answer, dataset, images/audios/videos (paths
# relative to the directory containing this JSONL file).
INPUT_JSONL="/scratch/keane/human_behaviour_data/final_v8_test_cleaned.jsonl"

# Directory where prediction JSONLs, metrics, and logs are written
OUTPUT_DIR="/home/keaneong/human-behavior/verl/verl_independent_test_eval/results"

# Path to the unified label map (used for metric computation at merge time)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL_MAP_PATH="$SCRIPT_DIR/../sft/label_maps/unified_label_map.json"

# GPUs to use. Leave empty to auto-detect all available GPUs.
# Example: GPUS=(0 1)  or  GPUS=(2 3 4 5)
GPUS=(2 3 4 5)

# Number of concurrent inference jobs per GPU.
# TOTAL_SHARDS = NUM_GPUS × JOBS_PER_GPU — increase when VRAM allows multiple processes.
JOBS_PER_GPU=4

# Batch size (samples per forward pass).
# Increase for image/text-only entries to amortise GPU overhead.
# Keep at 1–4 for entries with audio or video.
BATCH_SIZE=4

# Max tokens the model may generate per sample
MAX_NEW_TOKENS=512

# Sampling parameters for generation.
# Set TEMPERATURE="" or "0" to use greedy decoding (recommended for evaluation).
# TEMPERATURE=""
# TOP_P=0.95
# TOP_K=64
# MIN_P=0
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=64
MIN_P=0

# Data loading mode:
#   "default"    — decord + soundfile. Compatible with HumanOmniV2 and Gemma.
#   "verl_style" — qwen_vl_utils + torchaudio. Matches harpo_hier / omnisapiens training.
DATA_LOADING="default"

# Optional: num_frames override for video processing.
# Gemma4's processor defaults to 32; reduce to cut memory on longer clips.
# Leave empty to use the processor's default with automatic reduction for short clips.
NUM_FRAMES="4"

# Smoke test: take N random samples from every dataset (empty = disabled).
# Ensures all dataset code paths are exercised. Takes priority over MAX_SAMPLES.
SMOKE_N_PER_DATASET="1"   # e.g. "2" for a quick smoke test across all datasets

# Optional: cap samples by absolute count (empty = full run).
# Use SMOKE_N_PER_DATASET instead for a balanced smoke test.
MAX_SAMPLES=""

# Set to "1" to enable torch.compile (first batch will be slow)
TORCH_COMPILE=0

# Extra flags forwarded to eval.py for all shards (e.g. "--no_thinking")
EXTRA_ARGS=""

# Wandb logging (leave empty to disable)
WANDB_PROJECT=""      # e.g. "hb-eval"
WANDB_RUN_NAME=""     # e.g. "gemma4_test"
WANDB_ENTITY=""       # e.g. "my-team"

# Gemma thinking mode:
#   0 = native Gemma thinking: <|think|> system prompt + Gemma instruction (default)
#   1 = legacy thinking: shared THINKING_INSTRUCTION with <think></think> tags
GEMMA_LEGACY_THINKING=1

# ── END CONFIG ────────────────────────────────────────────────────────────────

EVAL="$SCRIPT_DIR/eval.py"

RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
MODEL_SLUG="${MODEL_NAME//\//_}"
OUT_BASE="$OUTPUT_DIR/${MODEL_SLUG}"
LOG_DIR="$OUTPUT_DIR/logs/$RUN_TIMESTAMP"

# ── Validation ────────────────────────────────────────────────────────────────
[[ -f "$INPUT_JSONL"    ]] || { echo "[ERROR] Missing input JSONL: $INPUT_JSONL"; exit 1; }
[[ -f "$LABEL_MAP_PATH" ]] || { echo "[ERROR] Missing label map: $LABEL_MAP_PATH"; exit 1; }

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

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

TOTAL_SHARDS=$(( NUM_GPUS * JOBS_PER_GPU ))
echo "Total shards: $TOTAL_SHARDS  ($NUM_GPUS GPU(s) × $JOBS_PER_GPU job(s)/GPU)"
echo "Logs → $LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────

maybe_max_samples() {
    [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]] && echo "--max_samples $MAX_SAMPLES" || echo ""
}

smoke_n_flag() {
    if [[ -n "$SMOKE_N_PER_DATASET" && "$SMOKE_N_PER_DATASET" != "0" ]]; then
        echo "--smoke_n_per_dataset $SMOKE_N_PER_DATASET"
    else
        maybe_max_samples
    fi
}

wandb_args() {
    local a=""
    [[ -n "$WANDB_PROJECT"   ]] && a+=" --wandb_project $WANDB_PROJECT"
    [[ -n "$WANDB_RUN_NAME"  ]] && a+=" --wandb_run_name $WANDB_RUN_NAME"
    [[ -n "$WANDB_ENTITY"    ]] && a+=" --wandb_entity $WANDB_ENTITY"
    echo "$a"
}

compile_flag() {
    [[ "$TORCH_COMPILE" == "1" ]] && echo "--torch_compile" || echo ""
}

num_frames_flag() {
    [[ -n "$1" && "$1" != "0" ]] && echo "--num_frames $1" || echo ""
}

gemma_legacy_flag() {
    [[ "$GEMMA_LEGACY_THINKING" == "1" ]] && echo "--gemma_legacy_thinking" || echo ""
}

sampling_args() {
    if [[ -n "$TEMPERATURE" && "$TEMPERATURE" != "0" ]]; then
        local args="--temperature $TEMPERATURE"
        [[ -n "${TOP_P:-}"  ]] && args+=" --top_p $TOP_P"
        [[ -n "${TOP_K:-}"  ]] && args+=" --top_k $TOP_K"
        [[ -n "${MIN_P:-}"  ]] && args+=" --min_p $MIN_P"
        echo "$args"
    fi
}

# ── Inference ─────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Model   : $MODEL_NAME"
echo "  Input   : $INPUT_JSONL"
echo "  GPUs    : ${GPUS[*]}  |  jobs/GPU: $JOBS_PER_GPU  |  shards: $TOTAL_SHARDS"
echo "  bs      : $BATCH_SIZE  |  data_loading: $DATA_LOADING  |  num_frames: ${NUM_FRAMES:-auto}"
echo "============================================================"

pids=()

for (( shard=0; shard<TOTAL_SHARDS; shard++ )); do
    gpu="${GPUS[$(( shard % NUM_GPUS ))]}"
    shard_out="${OUT_BASE}_shard${shard}.jsonl"
    echo "  [GPU $gpu] shard $shard/$TOTAL_SHARDS → $shard_out"

    CUDA_VISIBLE_DEVICES=$gpu python "$EVAL" \
        --model_name      "$MODEL_NAME" \
        --input_jsonl     "$INPUT_JSONL" \
        --output_jsonl    "$shard_out" \
        --label_map_path  "$LABEL_MAP_PATH" \
        --batch_size      "$BATCH_SIZE" \
        --max_new_tokens  "$MAX_NEW_TOKENS" \
        --num_shards      "$TOTAL_SHARDS" \
        --shard_idx       "$shard" \
        --data_loading    "$DATA_LOADING" \
        $(smoke_n_flag) \
        $(compile_flag) \
        $(sampling_args) \
        $(num_frames_flag "$NUM_FRAMES") \
        $(gemma_legacy_flag) \
        $EXTRA_ARGS \
        &> "$LOG_DIR/${MODEL_SLUG}_shard${shard}.log" &

    pids+=($!)
done

# ── Combined progress monitor ─────────────────────────────────────────────────
# Polls each shard's .progress sidecar file (written by eval.py) every 5 s and
# renders a single aggregated bar across all shards on one terminal line.
_progress_bar() {
    while true; do
        local done_total=0 samples_total=0
        for (( s=0; s<TOTAL_SHARDS; s++ )); do
            local pf="${OUT_BASE}_shard${s}.jsonl.progress"
            [[ -f "$pf" ]] || continue
            local d=0 t=0
            { read -r d && read -r t; } < "$pf" 2>/dev/null || true
            done_total=$(( done_total + ${d:-0} ))
            samples_total=$(( samples_total + ${t:-0} ))
        done
        if [[ "$samples_total" -gt 0 ]]; then
            local pct=$(( done_total * 100 / samples_total ))
            local filled=$(( pct * 40 / 100 )) bar="" space=""
            for (( i=0; i<filled;      i++ )); do bar+="=";  done
            for (( i=filled; i<40;     i++ )); do space+="."; done
            printf "\r  [%s%s] %d/%d (%d%%)" "$bar" "$space" \
                   "$done_total" "$samples_total" "$pct"
        fi
        sleep 5
    done
}

_progress_bar &
_pbar_pid=$!

# ── Wait for all shards ───────────────────────────────────────────────────────
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=$(( failed + 1 ))
    fi
done

kill "$_pbar_pid" 2>/dev/null
wait "$_pbar_pid" 2>/dev/null || true
printf "\n"

if [[ "$failed" -gt 0 ]]; then
    echo "[ERROR] $failed shard(s) failed — check logs in $LOG_DIR"
    exit 1
fi

echo "All shards done."

# ── Merge + compute unified metrics ──────────────────────────────────────────
MERGED_OUT="${OUT_BASE}_merged.jsonl"
METRICS_OUT="${OUT_BASE}_metrics.json"
JUDGE_OUT="${OUT_BASE}_judge.json"

echo "Merging shards → $MERGED_OUT"
python "$EVAL" \
    --merge_shards   "${OUT_BASE}_shard*.jsonl" \
    --output_jsonl   "$MERGED_OUT" \
    --model_name     "$MODEL_NAME" \
    --label_map_path "$LABEL_MAP_PATH" \
    --metrics_output "$METRICS_OUT" \
    --judge_output   "$JUDGE_OUT" \
    $(wandb_args)

# ── Clean up shard files ──────────────────────────────────────────────────────
rm -f "${OUT_BASE}_shard"*.jsonl "${OUT_BASE}_shard"*.jsonl.progress
echo "Shard files deleted."

echo ""
echo "Done."
echo "  Merged JSONL  → $MERGED_OUT"
echo "  Metrics       → $METRICS_OUT"
echo "  Judge format  → $JUDGE_OUT"
echo "  Logs          → $LOG_DIR"
