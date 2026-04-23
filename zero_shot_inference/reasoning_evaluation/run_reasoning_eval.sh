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
# Parallel dispatch across GPUs: JOBS_PER_GPU concurrent jobs per GPU,
# each job is one (model, dataset) pair. Edit the CONFIG section before running.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODELS=(
    "keentomato/harpo_hier_step400"
    "PhilipC/HumanOmniV2"
    "ddvd233/OmniSapiens-7B-RL"
    "Qwen/Qwen2.5-Omni-7B"
)

# GPUs to use. Leave empty to auto-detect all available GPUs.
# Example: GPUS=(0 1)  or  GPUS=(2 3 4 5)
GPUS=(0 1 7)

# Number of concurrent (model, dataset) jobs per GPU.
# Total parallel slots = NUM_GPUS × JOBS_PER_GPU.
JOBS_PER_GPU=2

# Output root — prediction JSONLs, metrics, and logs land here
OUTPUT_DIR="/home/keaneong/human-behavior/verl/zero_shot_inference/results/reasoning_eval"

# Max tokens the model may generate per sample
MAX_NEW_TOKENS=512

# Number of stochastic reasoning samples per entry
N_STOCHASTIC=5

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
MAX_SAMPLES=""   # e.g. "5"

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

# ── GPU detection ─────────────────────────────────────────────────────────────
if [[ "${#GPUS[@]}" -gt 0 ]]; then
    NUM_GPUS="${#GPUS[@]}"
    echo "Using specified GPU(s): ${GPUS[*]}"
elif command -v nvidia-smi &>/dev/null; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
    NUM_GPUS="${#GPUS[@]}"
    echo "Detected $NUM_GPUS GPU(s): ${GPUS[*]}"
else
    GPUS=(0); NUM_GPUS=1
    echo "No nvidia-smi found, defaulting to GPU 0"
fi
NUM_SLOTS=$(( NUM_GPUS * JOBS_PER_GPU ))
echo "Parallel slots: $NUM_SLOTS  ($NUM_GPUS GPU(s) × $JOBS_PER_GPU job(s)/GPU)"

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
# Models run sequentially. Datasets for each model are sharded across GPU slots
# (up to NUM_SLOTS concurrent jobs). After all datasets for a model finish,
# overall metrics are computed and logged to W&B before moving to the next model.

for MODEL in "${MODELS[@]}"; do
    CURRENT_MODEL="$MODEL"
    CURRENT_MODEL_SLUG="${MODEL//\//_}"
    _default_name="${WANDB_TAG:+${WANDB_TAG}_}${CURRENT_MODEL_SLUG}_${RUN_TIMESTAMP}"
    CURRENT_WANDB_RUN_ID="${_default_name}"
    CURRENT_WANDB_RUN_NAME="${WANDB_RUN_NAME:-${_default_name}}"

    echo ""
    echo "############################################################"
    echo "  Model: $MODEL"
    echo "############################################################"

    declare -a _slot_pids=()
    _model_jsonls=()

    for i in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$i]}"
        case "$DATASET" in
            eatd)     _in="$EATD_JSONL";     _para="$EATD_PARA_JSONL";     _ex="" ;;
            mvsa)     _in="$MVSA_JSONL";     _para="$MVSA_PARA_JSONL";     _ex="" ;;
            av-asd)   _in="$AVASD_JSONL";    _para="$AVASD_PARA_JSONL";    _ex="--multilabel" ;;
            iemocap)  _in="$IEMOCAP_JSONL";  _para="$IEMOCAP_PARA_JSONL";  _ex="" ;;
            dreaddit) _in="$DREADDIT_JSONL"; _para="$DREADDIT_PARA_JSONL"; _ex="" ;;
            sarcnet)  _in="$SARCNET_JSONL";  _para="$SARCNET_PARA_JSONL";  _ex="" ;;
        esac

        slot=$(( i % NUM_SLOTS ))
        gpu="${GPUS[$(( slot % NUM_GPUS ))]}"

        # Wait for any job currently occupying this slot before reusing it
        if [[ -n "${_slot_pids[$slot]:-}" ]]; then
            wait "${_slot_pids[$slot]}" || echo "[WARN] A job in slot $slot failed — continuing"
        fi

        _out_jsonl="$OUTPUT_DIR/${CURRENT_MODEL_SLUG}_${DATASET}_reasoning.jsonl"
        _model_jsonls+=("$_out_jsonl")

        (
            GPU="$gpu"
            run_reasoning_eval "$DATASET" "$_in" "$_para" "$_ex"
            run_metrics "$DATASET" "$_out_jsonl"
        ) &
        _slot_pids[$slot]=$!
    done

    # Drain all in-flight slots for this model before computing overall metrics
    for slot in "${!_slot_pids[@]}"; do
        [[ -n "${_slot_pids[$slot]:-}" ]] && { wait "${_slot_pids[$slot]}" || echo "[WARN] A job in slot $slot failed — continuing"; }
    done
    unset _slot_pids

    # ── Overall metrics for this model (all datasets combined) ────────────────
    if [[ "${#_model_jsonls[@]}" -gt 1 ]]; then
        echo ""
        echo "  [OVERALL METRICS] Computing cross-dataset summary for $MODEL …"
        python "$METRICS_PY" \
            --input_jsonl  "${_model_jsonls[@]}" \
            --model_name   "$MODEL" \
            $(build_wandb_args) \
            &>> "$LOG_DIR/${CURRENT_MODEL_SLUG}_overall_metrics.log"
        echo "  [OVERALL METRICS] Done."
    fi
done

echo ""
echo "All done. Results in: $OUTPUT_DIR"
echo "Logs in:   $LOG_DIR"
