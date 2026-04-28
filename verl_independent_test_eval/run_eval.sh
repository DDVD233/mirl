#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Test-set evaluation pipeline — single combined JSONL across all HB datasets.
#
# Speed strategy:
#   • Datasets are processed sequentially (one at a time)
#   • Within each dataset, all shards run in parallel across GPUs
#   • Shards are merged after all shards for that dataset finish
#   • Per-dataset metrics + LLM-judge JSON are written immediately after merge
#   • A completed dataset (_merged.jsonl present) is skipped on re-run
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
GPUS=(1 3 7)

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
SMOKE_N_PER_DATASET=""   # e.g. "2" for a quick smoke test across all datasets

# Optional: cap samples by absolute count (empty = full run).
# Use SMOKE_N_PER_DATASET instead for a balanced smoke test.
MAX_SAMPLES=""

# Set to "1" to enable torch.compile (first batch will be slow)
TORCH_COMPILE=0

# Extra flags forwarded to eval.py for all shards (e.g. "--no_thinking")
EXTRA_ARGS=""

# Wandb logging (leave empty to disable)
WANDB_PROJECT="hb_eval"      # e.g. "hb-eval"
WANDB_RUN_NAME="gemma4_test"     # e.g. "gemma4_test"
WANDB_ENTITY=""       # e.g. "my-team"

# Gemma thinking mode:
#   0 = native Gemma thinking: <|think|> system prompt + Gemma instruction (default)
#   1 = legacy thinking: shared THINKING_INSTRUCTION with <think></think> tags
GEMMA_LEGACY_THINKING=1

# Resume: set to a directory of prior shard output JSONLs to skip already-done entries.
# The script scans every *.jsonl in that dir, collects valid predictions, and runs
# inference only on what remains. Leave empty for a fresh run (if shards from a previous
# run exist at the same output paths, they will be auto-resumed automatically).
RESUME_DIR="/home/keaneong/human-behavior/verl/verl_independent_test_eval/results"

# Set to 1 to skip inference entirely and jump straight to merge + metrics.
# Merges per-dataset shard files from RESUME_DIR (if set) or OUTPUT_DIR.
MERGE_ONLY=0

# ── END CONFIG ────────────────────────────────────────────────────────────────

EVAL="$SCRIPT_DIR/eval.py"

RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
MODEL_SLUG="${MODEL_NAME//\//_}"
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

# ── Dataset discovery ─────────────────────────────────────────────────────────
mapfile -t DATASETS < <(
    python3 -c "
import json
dsets = sorted({json.loads(l)['dataset'] for l in open('${INPUT_JSONL}') if l.strip()})
print('\n'.join(dsets))
"
)
echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"

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

resume_dir_flag() {
    [[ -n "$RESUME_DIR" ]] && echo "--resume_dir $RESUME_DIR" || echo ""
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

# ── Combined progress monitor ─────────────────────────────────────────────────
# Polls each shard's .progress sidecar file (written by eval.py) every 5 s and
# renders a single aggregated bar across all shards on one terminal line.
# $1 = DS_OUT_BASE (the per-dataset path prefix for shard files)
_progress_bar() {
    local base="$1"
    while true; do
        local done_total=0 samples_total=0
        for (( s=0; s<TOTAL_SHARDS; s++ )); do
            local pf="${base}_shard${s}.jsonl.progress"
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

# ── Per-dataset loop ──────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Model   : $MODEL_NAME"
echo "  Input   : $INPUT_JSONL"
echo "  GPUs    : ${GPUS[*]}  |  jobs/GPU: $JOBS_PER_GPU  |  shards: $TOTAL_SHARDS"
echo "  bs      : $BATCH_SIZE  |  data_loading: $DATA_LOADING  |  num_frames: ${NUM_FRAMES:-auto}"
[[ -n "$RESUME_DIR" ]] && echo "  Resume  : $RESUME_DIR"
echo "============================================================"

ds_idx=0
for dataset in "${DATASETS[@]}"; do
    ds_idx=$(( ds_idx + 1 ))
    dataset_slug="${dataset//[^a-zA-Z0-9_]/_}"
    DS_OUT_BASE="${OUTPUT_DIR}/${MODEL_SLUG}_${dataset_slug}"
    MERGED_DS="${DS_OUT_BASE}_merged.jsonl"
    METRICS_DS="${DS_OUT_BASE}_metrics.json"
    JUDGE_DS="${DS_OUT_BASE}_judge.json"

    echo ""
    echo "── [$ds_idx/${#DATASETS[@]}] Dataset: $dataset ──"

    # ── MERGE_ONLY: merge existing shard files and skip inference ─────────────
    if [[ "$MERGE_ONLY" == "1" ]]; then
        merge_base="${RESUME_DIR:-$OUTPUT_DIR}/${MODEL_SLUG}_${dataset_slug}"
        if compgen -G "${merge_base}_shard*.jsonl" > /dev/null 2>&1; then
            python "$EVAL" \
                --merge_shards   "${merge_base}_shard*.jsonl" \
                --output_jsonl   "$MERGED_DS" \
                --model_name     "$MODEL_NAME" \
                --label_map_path "$LABEL_MAP_PATH" \
                --metrics_output "$METRICS_DS" \
                --judge_output   "$JUDGE_DS" \
                $(wandb_args) && echo "  Merged → $MERGED_DS" \
                             || echo "[WARN] Merge failed for $dataset"
        else
            echo "  [SKIP] no shard files found for $dataset in ${merge_base%/*}"
        fi
        continue
    fi

    # ── Skip if this dataset was already fully merged ─────────────────────────
    if [[ -f "$MERGED_DS" ]]; then
        echo "  [SKIP] merged output already exists: $MERGED_DS"
        continue
    fi

    # ── Launch shards in parallel ─────────────────────────────────────────────
    pids=()
    for (( shard=0; shard<TOTAL_SHARDS; shard++ )); do
        gpu="${GPUS[$(( shard % NUM_GPUS ))]}"
        shard_out="${DS_OUT_BASE}_shard${shard}.jsonl"

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
            --dataset_filter  "$dataset" \
            $(smoke_n_flag) \
            $(compile_flag) \
            $(sampling_args) \
            $(num_frames_flag "$NUM_FRAMES") \
            $(gemma_legacy_flag) \
            $(resume_dir_flag) \
            $EXTRA_ARGS \
            &> "$LOG_DIR/${MODEL_SLUG}_${dataset_slug}_shard${shard}.log" &

        pids+=($!)
    done

    # ── Monitor progress ──────────────────────────────────────────────────────
    _progress_bar "$DS_OUT_BASE" &
    _pbar_pid=$!

    # ── Wait for all shards ───────────────────────────────────────────────────
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$(( failed + 1 ))
    done

    kill "$_pbar_pid" 2>/dev/null
    wait "$_pbar_pid" 2>/dev/null || true
    printf "\n"

    if [[ "$failed" -gt 0 ]]; then
        echo "[ERROR] $failed shard(s) failed for '$dataset' — skipping merge."
        echo "        Partial shard files kept for resume. Check logs in $LOG_DIR"
        continue
    fi

    # ── Merge + per-dataset metrics + judge output ────────────────────────────
    echo "  Merging shards → $MERGED_DS"
    python "$EVAL" \
        --merge_shards   "${DS_OUT_BASE}_shard*.jsonl" \
        --output_jsonl   "$MERGED_DS" \
        --model_name     "$MODEL_NAME" \
        --label_map_path "$LABEL_MAP_PATH" \
        --metrics_output "$METRICS_DS" \
        --judge_output   "$JUDGE_DS" \
        $(wandb_args)

    rm -f "${DS_OUT_BASE}_shard"*.jsonl "${DS_OUT_BASE}_shard"*.jsonl.progress
    echo "  Done."
    echo "    Merged  → $MERGED_DS"
    echo "    Metrics → $METRICS_DS"
    echo "    Judge   → $JUDGE_DS"

done

echo ""
echo "All datasets done."
echo "Results in: $OUTPUT_DIR"
echo "Logs      : $LOG_DIR"
