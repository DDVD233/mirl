#!/usr/bin/env bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ──────────────────────────────────────────────────────────────────────
# Config (shared across all splits)
# ──────────────────────────────────────────────────────────────────────
ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}
backend=${BACKEND:-megatron}

TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-2}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

DATA_DIR="$HOME/scratch/raofu/3DHaptic"
PROJECT_NAME="tactile-sft"

MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=1e-6 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-7 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=True \
    engine.vanilla_mbridge=True"

# ──────────────────────────────────────────────────────────────────────
# Loop over 4 splits (skip participant)
# ──────────────────────────────────────────────────────────────────────
for SPLIT_NAME in date glove question task; do
    echo "============================================"
    echo "Starting SFT for split: ${SPLIT_NAME}"
    echo "============================================"

    train_path="${DATA_DIR}/annotation_verl_reasoning_sft_split_${SPLIT_NAME}_train.parquet"
    test_path="${DATA_DIR}/annotation_verl_reasoning_sft_split_${SPLIT_NAME}_test_mini.parquet"
    EXPERIMENT_NAME="qwen3vl_sft_split_${SPLIT_NAME}_official"
    CKPT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
    OUTPUT_DIR="./outputs/${EXPERIMENT_NAME}"

    # ── Step 1: SFT Training (1 epoch) ──
    torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS:-4} \
        ${ENTRYPOINT} \
        data.train_files="$train_path" \
        data.val_files="$test_path" \
        data.train_batch_size=32 \
        data.micro_batch_size_per_gpu=1 \
        data.max_length=8192 \
        data.pad_mode=no_padding \
        data.truncation=right \
        data.use_dynamic_bsz=True \
        data.max_token_len_per_gpu=8192 \
        data.ignore_input_ids_mismatch=True \
        model.path=Qwen/Qwen3-VL-8B-Instruct \
        model.use_remove_padding=True \
        ${MEGATRON_ENGINE_CONFIG} \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=100 \
        trainer.total_epochs=1 \
        trainer.resume_mode=disable \
        trainer.max_ckpt_to_keep=2 \
        checkpoint.save_contents=[model,optimizer,extra] $@

    # ── Step 2: Find the latest checkpoint ──
    LATEST_STEP=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST_STEP" ]; then
        echo "ERROR: No checkpoint found for split '${SPLIT_NAME}' in ${CKPT_DIR}, skipping merge."
        continue
    fi
    CKPT_PATH="${CKPT_DIR}/global_step_${LATEST_STEP}"
    echo "Found checkpoint at: ${CKPT_PATH}"

    # ── Step 3: Merge model to HuggingFace format ──
    mkdir -p "${OUTPUT_DIR}"

    python -m verl.model_merger merge \
        --backend megatron \
        --local_dir "${CKPT_PATH}" \
        --target_dir "${OUTPUT_DIR}"

    echo "============================================"
    echo "SFT model for split '${SPLIT_NAME}' saved to: ${OUTPUT_DIR}"
    echo "============================================"
done

echo ""
echo "All splits done!"
