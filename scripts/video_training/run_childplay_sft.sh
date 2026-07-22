#!/usr/bin/env bash
# SFT stage: Qwen3-VL-8B on ChildPlay ADOS justification traces (FSDP engine).
# Env knobs: NUM_GPUS, DATA_DIR, SFT_LR, SFT_EXP, TOTAL_EPOCHS, LOGGER, RESUME_MODE,
#            TOTAL_STEPS (optional cap for smoke runs)
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-4}
DATA_DIR=${DATA_DIR:-/scratch/dvdai/childplay_dataset}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
PROJECT_NAME=${PROJECT_NAME:-childplay_ados}
SFT_EXP=${SFT_EXP:-qwen3vl8b_sft}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_STEPS=${TOTAL_STEPS:-null}
SFT_TRAIN_FILE=${SFT_TRAIN_FILE:-childplay_ados_sft_train.parquet}
SFT_VAL_FILE=${SFT_VAL_FILE:-childplay_ados_sft_val.parquet}
RUN_DIR=${RUN_DIR:-/scratch/dvdai/childplay_ados}

torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/${SFT_TRAIN_FILE}" \
    data.val_files="${DATA_DIR}/${SFT_VAL_FILE}" \
    data.train_batch_size=${SFT_BS:-32} \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=16384 \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=16384 \
    data.ignore_input_ids_mismatch=True \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    engine=fsdp \
    optim=fsdp \
    optim.lr=${SFT_LR:-1e-5} \
    optim.weight_decay=0.1 \
    optim.clip_grad=1.0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${SFT_EXP}" \
    trainer.default_local_dir="${RUN_DIR}/checkpoints/${PROJECT_NAME}/${SFT_EXP}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-200} \
    trainer.test_freq=${TEST_FREQ:-200} \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.resume_mode=${RESUME_MODE:-auto} \
    trainer.max_ckpt_to_keep=2 \
    checkpoint.save_contents=[model,optimizer,extra] "$@"
