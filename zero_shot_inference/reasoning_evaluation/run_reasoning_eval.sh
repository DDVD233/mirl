#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Reasoning evaluation pipeline — av-asd, iemocap, dreaddit, sarcnet.
#
# For each model × dataset, runs two steps:
#   1. Four-mode reasoning inference  (direct / reasoning / stochastic / para)
#   2. Post-hoc metric computation + W&B logging
#
# Paraphrased JSONLs are pre-generated (by Claude Code) and stored alongside
# the originals as *_paraphrased.jsonl — no paraphrase step needed at runtime.
#
# Single-GPU per run (no sharding). The model is loaded once per dataset.
# Edit the CONFIG section before running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODELS=(
    "keentomato/harpo_hier_step400"
    "PhilipC/HumanOmniV2"
    "Qwen/Qwen2.5-Omni-7B"
)

# GPU to use for all inference
GPU=0

# Output root — prediction JSONLs, metrics, and logs land here
OUTPUT_DIR="/home/keaneong/human-behavior/verl/zero_shot_inference/results/reasoning_eval"

# Max tokens the model may generate per sample
MAX_NEW_TOKENS=1024

# Number of stochastic reasoning samples per entry
N_STOCHASTIC=3

# Sampling parameters for stochastic reasoning mode (Mode 3)
STOCHASTIC_TEMPERATURE=0.6
STOCHASTIC_TOP_P=0.95
STOCHASTIC_TOP_K=20

# Sampling parameters for direct (no-thinking) mode (Mode 1)
DIRECT_TEMPERATURE=0.7
DIRECT_TOP_P=0.8
DIRECT_TOP_K=20
DIRECT_MIN_P=0

# Data loading mode (verl_style matches harpo / omnisapiens training)
DATA_LOADING="verl_style"

# Optional: cap samples per dataset for a quick smoke-test (empty = full run)
MAX_SAMPLES="5"   # e.g. "5"

# Inference modes to run — valid values: direct reasoning stochastic para
# "direct" omitted: no-thinking mode showed no accuracy improvement in prior runs
MODES="reasoning stochastic para"

# Datasets to run — remove any you want to skip
# Available: eatd  mvsa  av-asd  iemocap  dreaddit  sarcnet
DATASETS=(
    "eatd"
    "mvsa"
    "av-asd"
    "iemocap"
    # "dreaddit"
    "sarcnet"
)

# ── DATASET JSONL PATHS ───────────────────────────────────────────────────────
# Adjust these paths for your cluster mount point.

EATD_JSONL="/scratch/keane/hb_generalization_data/MVSA_EATD_zeroshot/test_eatd_prompts.jsonl"
MVSA_JSONL="/scratch/keane/hb_generalization_data/MVSA_EATD_zeroshot/test_mvsa_prompts.jsonl"
AVASD_JSONL="/scratch/keane/hb_generalization_data/avasd/test_av_asd_promptsmultilabel.jsonl"
IEMOCAP_JSONL="/scratch/keane/hb_generalization_data/iemocap/latest_iemocap_test.jsonl"
DREADDIT_JSONL="/scratch/keane/hb_generalization_data/zero_shot_data_v2/test_dreaddit_prompts.jsonl"
SARCNET_JSONL="/scratch/keane/hb_generalization_data/zero_shot_data_v2/test_sarcnet_prompts.jsonl"

# Pre-generated paraphrased JSONLs (created by Claude Code; stored next to originals)
EATD_PARA_JSONL="/scratch/keane/hb_generalization_data/paraphrased_prompts/test_eatd_prompts_paraphrased.jsonl"
MVSA_PARA_JSONL="/scratch/keane/hb_generalization_data/paraphrased_prompts/test_mvsa_prompts_paraphrased.jsonl"
AVASD_PARA_JSONL="/scratch/keane/hb_generalization_data/paraphrased_prompts/test_av_asd_promptsmultilabel_paraphrased.jsonl"
IEMOCAP_PARA_JSONL="/scratch/keane/hb_generalization_data/paraphrased_prompts/latest_iemocap_test_paraphrased.jsonl"
DREADDIT_PARA_JSONL="/scratch/keane/hb_generalization_data/paraphrased_prompts/test_dreaddit_prompts_paraphrased.jsonl"
SARCNET_PARA_JSONL="/scratch/keane/hb_generalization_data/paraphrased_prompts/test_sarcnet_prompts_paraphrased.jsonl"

# ── W&B CONFIG ────────────────────────────────────────────────────────────────
WANDB_PROJECT="reasoning-evaluation"  # empty = disable W&B logging
WANDB_TAG=""                          # optional prefix: "{tag}_{model_slug}_{timestamp}"
WANDB_RUN_NAME=""                     # override full run name
WANDB_ENTITY=""                       # W&B entity (org/team); empty = default

# ── END CONFIG ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REASONING_EVAL_PY="$SCRIPT_DIR/reasoning_eval.py"
METRICS_PY="$SCRIPT_DIR/compute_reasoning_metrics.py"

RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="$OUTPUT_DIR/logs/$RUN_TIMESTAMP"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "Logs → $LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────

maybe_max_samples() {
    [[ -n "$MAX_SAMPLES" && "$MAX_SAMPLES" != "0" ]] && echo "--max_samples $MAX_SAMPLES" || echo ""
}

modes_need_para() { [[ "$MODES" == *"para"* ]]; }

build_wandb_args() {
    if [[ -z "$WANDB_PROJECT" ]]; then echo ""; return; fi
    local args="--wandb_project $WANDB_PROJECT --wandb_run_id $CURRENT_WANDB_RUN_ID --wandb_run_name $CURRENT_WANDB_RUN_NAME"
    [[ -n "$WANDB_ENTITY" ]] && args+=" --wandb_entity $WANDB_ENTITY"
    echo "$args"
}

# Step 1: Four-mode reasoning inference (single GPU, single model load)
run_reasoning_eval() {
    local dataset_name="$1"
    local input_jsonl="$2"
    local para_jsonl="$3"
    local dataset_extra_args="${4:-}"
    local out_jsonl="$OUTPUT_DIR/${CURRENT_MODEL_SLUG}_${dataset_name}_reasoning.jsonl"

    echo ""
    echo "============================================================"
    echo "  Model   : $CURRENT_MODEL"
    echo "  Dataset : $dataset_name  [reasoning eval, GPU $GPU]"
    echo "  Input   : $(basename "$input_jsonl")"
    echo "  Para    : $(basename "$para_jsonl")"
    echo "  Output  : $out_jsonl"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU python "$REASONING_EVAL_PY" \
        --model                  "$CURRENT_MODEL" \
        --input_jsonl            "$input_jsonl" \
        --para_input_jsonl       "$para_jsonl" \
        --output_jsonl           "$out_jsonl" \
        --max_new_tokens         "$MAX_NEW_TOKENS" \
        --n_stochastic           "$N_STOCHASTIC" \
        --stochastic_temperature "$STOCHASTIC_TEMPERATURE" \
        --stochastic_top_p       "$STOCHASTIC_TOP_P" \
        --stochastic_top_k       "$STOCHASTIC_TOP_K" \
        --direct_temperature     "$DIRECT_TEMPERATURE" \
        --direct_top_p           "$DIRECT_TOP_P" \
        --direct_top_k           "$DIRECT_TOP_K" \
        --direct_min_p           "$DIRECT_MIN_P" \
        --data_loading           "$DATA_LOADING" \
        --modes                  $MODES \
        $(maybe_max_samples) \
        $dataset_extra_args \
        &> "$LOG_DIR/${CURRENT_MODEL_SLUG}_${dataset_name}_reasoning_eval.log"

    echo "$out_jsonl"
}

# Step 2: Post-hoc metric computation + W&B logging
run_metrics() {
    local dataset_name="$1"
    local reasoning_jsonl="$2"

    echo "  [METRICS] Computing reasoning metrics for $dataset_name …"
    python "$METRICS_PY" \
        --input_jsonl  "$reasoning_jsonl" \
        --model_name   "$CURRENT_MODEL" \
        $(build_wandb_args) \
        &>> "$LOG_DIR/${CURRENT_MODEL_SLUG}_${dataset_name}_metrics.log"
    echo "  [METRICS] Done."
}

# ── Dataset verification ───────────────────────────────────────────────────────

for ds in "${DATASETS[@]}"; do
    case "$ds" in
        eatd)
            [[ -f "$EATD_JSONL" ]] || { echo "[ERROR] Missing: $EATD_JSONL"; exit 1; }
            modes_need_para && { [[ -f "$EATD_PARA_JSONL" ]] || { echo "[ERROR] Missing: $EATD_PARA_JSONL"; exit 1; }; }
            ;;
        mvsa)
            [[ -f "$MVSA_JSONL" ]] || { echo "[ERROR] Missing: $MVSA_JSONL"; exit 1; }
            modes_need_para && { [[ -f "$MVSA_PARA_JSONL" ]] || { echo "[ERROR] Missing: $MVSA_PARA_JSONL"; exit 1; }; }
            ;;
        av-asd)
            [[ -f "$AVASD_JSONL" ]] || { echo "[ERROR] Missing: $AVASD_JSONL"; exit 1; }
            modes_need_para && { [[ -f "$AVASD_PARA_JSONL" ]] || { echo "[ERROR] Missing: $AVASD_PARA_JSONL"; exit 1; }; }
            ;;
        iemocap)
            [[ -f "$IEMOCAP_JSONL" ]] || { echo "[ERROR] Missing: $IEMOCAP_JSONL"; exit 1; }
            modes_need_para && { [[ -f "$IEMOCAP_PARA_JSONL" ]] || { echo "[ERROR] Missing: $IEMOCAP_PARA_JSONL"; exit 1; }; }
            ;;
        dreaddit)
            [[ -f "$DREADDIT_JSONL" ]] || { echo "[ERROR] Missing: $DREADDIT_JSONL"; exit 1; }
            modes_need_para && { [[ -f "$DREADDIT_PARA_JSONL" ]] || { echo "[ERROR] Missing: $DREADDIT_PARA_JSONL"; exit 1; }; }
            ;;
        sarcnet)
            [[ -f "$SARCNET_JSONL" ]] || { echo "[ERROR] Missing: $SARCNET_JSONL"; exit 1; }
            modes_need_para && { [[ -f "$SARCNET_PARA_JSONL" ]] || { echo "[ERROR] Missing: $SARCNET_PARA_JSONL"; exit 1; }; }
            ;;
        *) echo "[ERROR] Unknown dataset '$ds'. Valid: eatd mvsa av-asd iemocap dreaddit sarcnet"; exit 1 ;;
    esac
done

# ── Main loop ─────────────────────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
    CURRENT_MODEL="$MODEL"
    CURRENT_MODEL_SLUG="${MODEL//\//_}"

    # One W&B run per model — all datasets resume into the same run
    _default_name="${WANDB_TAG:+${WANDB_TAG}_}${CURRENT_MODEL_SLUG}_${RUN_TIMESTAMP}"
    CURRENT_WANDB_RUN_ID="${_default_name}"
    CURRENT_WANDB_RUN_NAME="${WANDB_RUN_NAME:-${_default_name}}"

    for DATASET in "${DATASETS[@]}"; do
        case "$DATASET" in
            eatd)
                INPUT_JSONL="$EATD_JSONL"
                PARA_JSONL="$EATD_PARA_JSONL"
                DATASET_EXTRA=""
                ;;
            mvsa)
                INPUT_JSONL="$MVSA_JSONL"
                PARA_JSONL="$MVSA_PARA_JSONL"
                DATASET_EXTRA=""
                ;;
            av-asd)
                INPUT_JSONL="$AVASD_JSONL"
                PARA_JSONL="$AVASD_PARA_JSONL"
                DATASET_EXTRA="--multilabel"
                ;;
            iemocap)
                INPUT_JSONL="$IEMOCAP_JSONL"
                PARA_JSONL="$IEMOCAP_PARA_JSONL"
                DATASET_EXTRA=""
                ;;
            dreaddit)
                INPUT_JSONL="$DREADDIT_JSONL"
                PARA_JSONL="$DREADDIT_PARA_JSONL"
                DATASET_EXTRA=""
                ;;
            sarcnet)
                INPUT_JSONL="$SARCNET_JSONL"
                PARA_JSONL="$SARCNET_PARA_JSONL"
                DATASET_EXTRA=""
                ;;
        esac

        REASONING_JSONL=$(run_reasoning_eval "$DATASET" "$INPUT_JSONL" "$PARA_JSONL" "$DATASET_EXTRA")
        run_metrics "$DATASET" "$REASONING_JSONL"
    done
done

echo ""
echo "All done. Results in: $OUTPUT_DIR"
echo "Logs in:   $LOG_DIR"
