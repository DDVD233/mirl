#!/usr/bin/env bash
# Retrieval warm-start SFT: teach STOCK Qwen3.6-27B the concise search->answer
# format (RetrievalToolAgentLoop) from teacher-generated, rubric-filtered traces,
# BEFORE RL. Fixes the cold-start problems: retrieval starts a drag below the
# no-retrieval baseline (stock+ret 0.395) and RL rollouts are slow/verbose.
#
# Traces (scripts/self_evolving/make_retrieval_traces.py) are multi-turn message
# lists {messages:[user, assistant(query), tool(passages), assistant(answer)], tools}
# consumed by verl MultiTurnSFTDataset (loss only on assistant turns; the tool turn
# is masked). 1 epoch, save the final checkpoint for RL init.
#
# Run on SERVER 1 (4 GPUs) AFTER tearing down the local teacher (frees the GPUs).
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl_healthbench}
cd "$REPO"

TRACES_JSONL="${TRACES_JSONL:-/scratch/sheng/self_evolving/retrieval_sft_traces.jsonl}"
TRAIN_PARQUET="${TRAIN_PARQUET:-/scratch/sheng/self_evolving/retrieval_sft_traces.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-27B}"          # STOCK
EXP="${EXP:-retrieval_sft_qwen36_27b}"
CKPT_DIR="${CKPT_DIR:-/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$EXP}"

export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"

# jsonl -> parquet (MultiTurnSFTDataset reads parquet). messages/tools stay as lists.
/usr/local/bin/python - "$TRACES_JSONL" "$TRAIN_PARQUET" <<'PY'
import sys, json, pandas as pd
rows=[json.loads(l) for l in open(sys.argv[1])]
df=pd.DataFrame(rows)
# keep only what the SFT dataset needs
df=df[["messages","tools"]]
df.to_parquet(sys.argv[2])
print(f"wrote {len(df)} traces -> {sys.argv[2]}")
PY

torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_TRAINERS:-4}" \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_PARQUET" \
    data.messages_key=messages \
    data.tools_key=tools \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.max_length="${SFT_MAX_LENGTH:-12288}" \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU:-24576}" \
    model.path="$MODEL_PATH" \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    engine=fsdp \
    optim=fsdp \
    optim.lr="${LR:-1e-5}" \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.warmup_style=cosine \
    optim.min_lr_ratio=0.1 \
    engine.ulysses_sequence_parallel_size="${SP_SIZE:-2}" \
    engine.strategy=fsdp2 \
    engine.fsdp_size=-1 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs="${EPOCHS:-1}" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.project_name=self_evolving_medical \
    trainer.experiment_name="$EXP" \
    'trainer.logger=[console,wandb]' \
    trainer.resume_mode=disable \
    trainer.max_ckpt_to_keep=2 \
    checkpoint.save_contents=[model,optimizer,extra] \
    "$@"
