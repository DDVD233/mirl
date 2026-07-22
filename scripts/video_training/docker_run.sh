#!/usr/bin/env bash
# One-click docker wrapper for the ChildPlay ADOS-2 pipeline.
#
# Usage:
#   [SMOKE=1] [CUDA_VISIBLE_DEVICES=0,1,2,3] bash scripts/video_training/docker_run.sh <stage>
# where <stage> is prep|sft|rl|rl_only|all (see run_pipeline.sh).
#
# Builds the image if missing, then runs the pipeline with /scratch/dvdai and the
# HF cache mounted. The live repo is bind-mounted over /workspace/mirl so code
# edits do not require a rebuild.
set -euo pipefail

STAGE=${1:-all}
IMAGE=${IMAGE:-mirl-video-training:latest}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-$(awk -F, '{print NF}' <<< "${GPUS}")}

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[docker_run] building ${IMAGE}"
    docker build -f "${REPO_DIR}/docker/Dockerfile.video_training" -t "${IMAGE}" "${REPO_DIR}"
fi

mkdir -p "${HOME}/.cache/huggingface"

# Forward tuning knobs that are set in the caller's environment.
ENV_ARGS=()
for var in GPU_UTIL ACTOR_OFFLOAD ROLLOUT_TP GRPO_N TRAIN_BS VAL_BS MODEL_PATH \
           SFT_LR RL_LR TOTAL_EPOCHS TOTAL_STEPS RL_STEPS LOGGER VAL_BEFORE \
           SAVE_FREQ TEST_FREQ MAX_PROMPT_LEN MAX_RESP_LEN PROJECT_NAME \
           SFT_EXP RL_EXP SFT_TRAIN_FILE SFT_VAL_FILE TRAIN_FILE VAL_FILE; do
    if [ -n "${!var:-}" ]; then
        ENV_ARGS+=(-e "${var}=${!var}")
    fi
done

exec docker run --rm ${DOCKER_TTY:--i} --shm-size=32g \
    --gpus "\"device=${GPUS}\"" \
    -e NUM_GPUS="${NUM_GPUS}" \
    -e SMOKE="${SMOKE:-0}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e DATA_DIR="${DATA_DIR:-/scratch/dvdai/childplay_dataset}" \
    -e RUN_DIR="${RUN_DIR:-/scratch/dvdai/childplay_ados}" \
    ${ENV_ARGS[@]:+"${ENV_ARGS[@]}"} \
    -v /scratch/dvdai:/scratch/dvdai \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${REPO_DIR}:/workspace/mirl" \
    -w /workspace/mirl \
    "${IMAGE}" bash scripts/video_training/run_pipeline.sh "${STAGE}"
