#!/usr/bin/env bash
# Warm-start SFT on GPT-5.6 traces (scripts/self_evolving/make_gpt56_traces.py).
#
# Replaces the selective warm start, which actively HURT: its SFT model's thinking
# collapsed to ~110 chars (stock ~10k) and step-0 val fell to 0.488 vs the 0.559
# no-tool baseline. What changed in the data (see make_gpt56_traces.py header):
#   * traces are emitted in the TWO renderings the rollout really samples — the
#     tool-free graded answer turn, and the phase-1 retrieve-or-not decision —
#     instead of one 4-message shape RL never sees;
#   * <think> is substantive teacher reasoning (gated >=900 chars), not a canned
#     one-liner, so SFT cannot teach think-suppression;
#   * the teacher never sees the rubric (no answer-key leakage), and traces are
#     kept only if they pass the VERBATIM official HealthBench grader.
#
# Conservative LR: the previous warm start ran 1e-5 and damaged the base model's
# reasoning. These traces are better but the failure mode was catastrophic, so
# default to 2e-6 and 1 epoch. The decisive number is the step-0 val of the RL
# run started from the merged checkpoint: it must beat 0.563 (stock + fixed
# retrieval loop) and 0.559 (no-tool), not just the 0.488 the old warm start hit.
#
# Run on a node with 4 free GPUs, AFTER tearing down anything holding them.
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl_healthbench}
cd "$REPO"

TRACES_JSONL="${TRACES_JSONL:-/scratch/sheng/self_evolving/gpt56_sft_traces.jsonl}"
TRAIN_PARQUET="${TRAIN_PARQUET:-/scratch/sheng/self_evolving/gpt56_sft_traces.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-27B}"          # STOCK, like v9/v10
EXP="${EXP:-gpt56_sft_qwen36_27b}"
CKPT_DIR="${CKPT_DIR:-/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/$EXP}"

export HF_HOME=/scratch/sheng/self_evolving/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"

# jsonl -> parquet (MultiTurnSFTDataset reads parquet). `tools` must exist on every
# row; answer traces carry [] because the graded answer turn is rendered tool-free.
/usr/local/bin/python - "$TRACES_JSONL" "$TRAIN_PARQUET" <<'PY'
import json
import sys

import pandas as pd

rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    rows.append({"messages": d["messages"], "tools": d.get("tools") or []})
df = pd.DataFrame(rows)
df.to_parquet(sys.argv[2])
from collections import Counter
kinds = Counter(json.loads(l)["extra_info"]["type"]
                for l in open(sys.argv[1]) if l.strip())
print(f"wrote {len(df)} traces -> {sys.argv[2]}")
print(f"  mix: {dict(kinds)}")
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
    optim.lr="${LR:-2e-6}" \
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
