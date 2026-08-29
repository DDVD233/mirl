#!/usr/bin/env bash
# MedXpertQA baseline: the frozen Qwen3.5-9B (the model every stage-2 9B arm starts
# from) on Text then MM, against the dedicated inference box.
#
# Sequential, not parallel, and at modest concurrency ON PURPOSE: that box is also
# ARM=16's retrieval summarizer (SUMM_BASE), and a queued brief blocks a search turn,
# which blocks a rollout. Eval throughput is worth less than a live arm's step time.
#
# BUDGET. The 9B thinks before it answers and vLLM leaves `content` EMPTY until
# </think> closes, so a response that runs out of tokens scores zero with no visible
# answer. Median reasoning on MM measured ~27k chars (~7k tokens), and the served
# max_model_len is 16384 TOTAL, so the budget here is set as close to that ceiling as
# the prompt allows -- images eat context, so MM gets less than Text. The reported
# `truncated` count is the honest caveat on any number this produces.
#
# Usage (on an MSR pod, from the repo root):
#   nohup bash scripts/self_evolving/eval/run_medxpertqa_baseline.sh > /root/medx_baseline.log 2>&1 &
set -u

S=/scratch/sheng/self_evolving
OUT="$S/eval_out"
BASE_URL="${BASE_URL:-http://point.dd.works:18186/v1}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
TAG="${TAG:-base9b}"
mkdir -p "$OUT"

run () {
    local subset="$1" maxtok="$2" conc="$3"
    echo "=== $(date -Is) medxpertqa $subset ($MODEL) ==="
    python3 scripts/self_evolving/eval/medxpertqa_eval.py \
        --subset "$subset" \
        --base_url "$BASE_URL" \
        --model "$MODEL" \
        --max_tokens "$maxtok" \
        --concurrency "$conc" \
        --temperature 0 \
        --out "$OUT/medxpertqa_${subset}_${TAG}.json"
    echo "=== $(date -Is) $subset done (rc=$?) ==="
}

run text 14000 24
run mm 11000 12
echo "=== $(date -Is) all done ==="
