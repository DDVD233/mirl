#!/usr/bin/env bash
# Build the distillation SFT annotations (reasoning trace + boxed answer) for
# mimiciv_rare, using the base Qwen3.6-27B served on SERVER 5
# (point.dd.works:18184). Run this ON server5 itself so the teacher endpoint is
# local (low latency, no external rate limit).
#
#   ssh -p 2338 root@point.dd.works      # server5 (serving node)
#   tmux new -s distill                  # one tmux session, multiple windows
#   bash /scratch/sheng/self_evolving/repro/run_make_distill_traces.sh   # or repo copy
#
# Produces, for every train question, up to N_PER distinct teacher traces that
# (a) box the EXACT ground truth, (b) never reveal the answer was given, and
# (c) land ~1000 tokens. The whole file is shuffled at the end. Output feeds
# StaticTraceSFTDataset (scripts/self_evolving/static_trace_sft_dataset.py) for
# the first (SFT) stage; the same train.jsonl drives the second (RL) stage.
set -xeuo pipefail

REPO="${REPO:-/scratch/sheng/self_evolving/repro}"
DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
OUT="${OUT:-$DATA_DIR/distill_sft_train.jsonl}"
# The model on point.dd.works:18184 is a SHARED tp2 deployment (also the
# production judge) reachable only via the external ingress — NOT localhost on
# any one pod. Keep CONCURRENCY low (~3) to avoid saturating it / starving other
# runs' judge calls. The run is crash-safe + resumable, so slow is fine.
API_BASE="${API_BASE:-http://point.dd.works:18184/v1}"
API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.climb_teacher_key)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
N_PER="${N_PER:-3}"
CONCURRENCY="${CONCURRENCY:-3}"
OVERSAMPLE="${OVERSAMPLE:-4}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

cd "$REPO"
exec /usr/local/bin/python -u scripts/self_evolving/make_distill_traces.py \
    --train_file "$DATA_DIR/train.jsonl" \
    --out "$OUT" \
    --api_base "$API_BASE" \
    --api_key "$API_KEY" \
    --model_name "$MODEL_NAME" \
    --n_per_question "$N_PER" \
    --oversample "$OVERSAMPLE" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY" \
    --max_samples "$MAX_SAMPLES" \
    "$@"
