#!/usr/bin/env bash
set -x

# Copy HF model from SFT checkpoint huggingface/ dir for all 4 splits (skip participant).

PROJECT_NAME="tactile-sft"

for SPLIT_NAME in date glove question task; do
    echo "============================================"
    echo "Copying SFT model for split: ${SPLIT_NAME}"
    echo "============================================"

    EXPERIMENT_NAME="qwen3vl_sft_split_${SPLIT_NAME}_official"
    CKPT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
    OUTPUT_DIR="./outputs/${EXPERIMENT_NAME}"

    LATEST_STEP=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST_STEP" ]; then
        echo "WARNING: No checkpoint found for split '${SPLIT_NAME}' in ${CKPT_DIR}, skipping."
        continue
    fi

    HF_PATH="${CKPT_DIR}/global_step_${LATEST_STEP}/huggingface"
    if [ ! -d "$HF_PATH" ]; then
        echo "WARNING: HF model not found at ${HF_PATH}, skipping."
        continue
    fi

    echo "Found HF model at: ${HF_PATH}"
    mkdir -p "${OUTPUT_DIR}"
    cp -r "${HF_PATH}/"* "${OUTPUT_DIR}/"

    echo "SFT model for split '${SPLIT_NAME}' saved to: ${OUTPUT_DIR}"
done

echo ""
echo "All copies done!"
