#!/usr/bin/env bash
# One-click ChildPlay ADOS-2 video training pipeline.
#
# Usage: run_pipeline.sh <stage>
#   prep     — copy raw dataset + build train/val jsonl + SFT parquet
#   sft      — SFT stage only
#   rl       — RL (GRPO) from the merged SFT checkpoint
#   rl_only  — RL (GRPO) directly from the base HF model (prep first if needed)
#   all      — prep -> sft -> merge to HF -> rl
#
# SMOKE=1 runs a tiny few-step configuration on the smoke subsets.
# Other env knobs are forwarded to the stage scripts (NUM_GPUS, DATA_DIR, ...).
set -euo pipefail

STAGE=${1:-all}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR=${DATA_DIR:-/scratch/dvdai/childplay_dataset}
RUN_DIR=${RUN_DIR:-/scratch/dvdai/childplay_ados}
SFT_HF_DIR=${SFT_HF_DIR:-${RUN_DIR}/sft_hf}
PROJECT_NAME=${PROJECT_NAME:-childplay_ados}
export NUM_GPUS=${NUM_GPUS:-4}
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/logs_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "[pipeline] stage=${STAGE} smoke=${SMOKE:-0} log=${LOG}"

if [ "${SMOKE:-0}" = "1" ]; then
    export TRAIN_FILE=childplay_ados_smoke_train.jsonl
    export VAL_FILE=childplay_ados_smoke_val.jsonl
    export SFT_TRAIN_FILE=${SFT_TRAIN_FILE:-childplay_ados_sft_train.parquet}
    export TRAIN_BS=${TRAIN_BS:-8}
    export GRPO_N=${GRPO_N:-4}
    export VAL_BEFORE=False
    export SAVE_FREQ=2
    export TEST_FREQ=2
    export TOTAL_STEPS=${TOTAL_STEPS:-10}
    export LOGGER=${LOGGER:-'["console"]'}
    SMOKE_RL_ARGS=(+trainer.total_training_steps=${RL_STEPS:-4})
    EXP_SUFFIX="_smoke"
else
    SMOKE_RL_ARGS=()
    EXP_SUFFIX=""
fi
export SFT_EXP=${SFT_EXP:-qwen3vl8b_sft${EXP_SUFFIX}}
export RL_EXP=${RL_EXP:-qwen3vl8b_grpo${EXP_SUFFIX}}

run_prep() {
    echo "[pipeline] === prep ==="
    python3 "${SCRIPT_DIR}/prepare_childplay.py" --out "${DATA_DIR}" ${PREP_ARGS:-}
}

prep_if_needed() {
    if [ ! -f "${DATA_DIR}/childplay_ados_train.jsonl" ]; then
        run_prep
    else
        echo "[pipeline] prep outputs already present, skipping (rm ${DATA_DIR}/childplay_ados_train.jsonl to force)"
    fi
}

run_sft() {
    echo "[pipeline] === sft ==="
    bash "${SCRIPT_DIR}/run_childplay_sft.sh"
}

merge_sft() {
    echo "[pipeline] === merge SFT ckpt -> HF ==="
    local ckpt_dir="${RUN_DIR}/checkpoints/${PROJECT_NAME}/${SFT_EXP}"
    local latest
    latest=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt" 2>/dev/null || true)
    if [ -z "${latest}" ]; then
        echo "[pipeline] FATAL: no SFT checkpoint under ${ckpt_dir}" >&2
        exit 1
    fi
    local ckpt="${ckpt_dir}/global_step_${latest}"
    [ -d "${ckpt}/actor" ] && ckpt="${ckpt}/actor"
    echo "[pipeline] merging ${ckpt} -> ${SFT_HF_DIR}"
    python3 -m verl.model_merger merge --backend fsdp \
        --local_dir "${ckpt}" --target_dir "${SFT_HF_DIR}"
    if [ ! -f "${SFT_HF_DIR}/config.json" ] || ! ls "${SFT_HF_DIR}"/*.safetensors >/dev/null 2>&1; then
        echo "[pipeline] FATAL: merged HF dir ${SFT_HF_DIR} is incomplete" >&2
        exit 1
    fi
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
}

run_rl() {
    echo "[pipeline] === rl (model=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}) ==="
    bash "${SCRIPT_DIR}/run_childplay_grpo.sh" "${SMOKE_RL_ARGS[@]}"
}

case "${STAGE}" in
    prep)
        run_prep
        ;;
    sft)
        prep_if_needed
        run_sft
        ;;
    rl)
        prep_if_needed
        if [ ! -f "${SFT_HF_DIR}/config.json" ]; then
            echo "[pipeline] FATAL: ${SFT_HF_DIR} not found — run 'sft' + merge first (or use rl_only)" >&2
            exit 1
        fi
        MODEL_PATH="${SFT_HF_DIR}" run_rl
        ;;
    rl_only)
        prep_if_needed
        run_rl
        ;;
    all)
        prep_if_needed
        run_sft
        merge_sft
        MODEL_PATH="${SFT_HF_DIR}" run_rl
        ;;
    *)
        echo "Usage: $0 {prep|sft|rl|rl_only|all}" >&2
        exit 1
        ;;
esac
echo "[pipeline] stage=${STAGE} finished OK"
