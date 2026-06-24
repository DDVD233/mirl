#!/usr/bin/env bash
# Evaluate gpt-5.3-chat as the ANSWERER on the MIMIC-rare test set using the SAME
# protocol/scoring as the local-model sweep: same test.jsonl (multimodal), scored
# by self_evolving.compute_score with the lenient gpt-5.3 judge, per ICD category.
# A frontier baseline to contextualize the trained-model numbers. Runs via the
# TRAPI proxy (no GPU), so it can run alongside the local sweep.
#
#   LIMIT=24 bash run_eval_gpt53.sh        # quick slice
#   LIMIT=0  bash run_eval_gpt53.sh        # full 2452
set -xeuo pipefail
REPO=/scratch/sheng/self_evolving/verl
DATA=/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl
KEY=$(cat /scratch/sheng/self_evolving/.trapi_key)
TRAPI=http://point.dd.works:18890/v1
MODEL="${MODEL:-gpt-5.3-chat_2026-03-03}"
JUDGE="${JUDGE:-$MODEL}"                                 # judge model (default = answerer); vary to test judge effect
OUT="${OUT:-/scratch/sheng/self_evolving/eval_hf/gpt53_baseline}"

export CHAT_PROVIDER=trapi
export REWARD_EVAL_LENIENT_ONLY=1                       # same lenient-only as the sweep
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-8}"  # share the rate limit w/ the sweep
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-high}" # let gpt-5.3 think (match thinking-ON)
export OPENAI_MAX_COMPLETION_TOKENS="${OPENAI_MAX_COMPLETION_TOKENS:-24576}"
export HF_HOME=/scratch/sheng/self_evolving/hf_cache
mkdir -p "$OUT"
cd "$REPO"

/usr/local/bin/python scripts/self_evolving/eval_sota.py \
    --provider openai \
    --model_name "$MODEL" \
    --openai_base_url "$TRAPI" \
    --openai_api_key "$KEY" \
    --judge_model_name "$JUDGE" \
    --api_base "$TRAPI" \
    --judge_api_key "$KEY" \
    --embed_api_base http://mib.media.mit.edu:18001/v1 \
    --embed_model Qwen/Qwen3-VL-Embedding-2B \
    --val_file "${DATA_OVERRIDE:-$DATA}" \
    --concurrency "${CONC:-24}" \
    --sample_n "${SAMPLE_N:-0}" \
    --sample_seed "${SAMPLE_SEED:-0}" \
    --limit "${LIMIT:-0}" \
    --output_jsonl "$OUT/gpt53.jsonl" \
    --summary_json "$OUT/gpt53.summary.json" \
    "$@"
