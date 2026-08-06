#!/usr/bin/env bash
# Chain the two frontier-model evaluations that feed the paper's scaling figure.
#
#   1. wait for the full-benchmark zero-shot run to finish (it is already running)
#   2. re-judge its dumps with the SAME judge used inside the training loop, so the
#      frontier line and the training curve are comparable
#   3. restart the few-shot sweep at full concurrency (it resumes from its jsonl)
#
# Both evaluations share one rate-limited proxy, so the sweep runs starved until the
# zero-shot run is out of the way.
set -uo pipefail

OUT=/scratch/self_evolving_datasets/eval_gpt56
TRAPI=http://point.dd.works:18890/v1
JUDGE=http://point.dd.works:18184/v1
KEY="${TRAPI_KEY:?set TRAPI_KEY}"
REPO=/home/dvd/mirl
TEST=/scratch/self_evolving_datasets/mimiciv_rare/test.jsonl

cd "$REPO"

echo "[chain] waiting for zero-shot run to finish..."
while pgrep -f "model_name gpt-5.6-sol" >/dev/null; do sleep 60; done
echo "[chain] zero-shot done: $(wc -l < $OUT/eval_gpt56sol.jsonl) rows"

echo "[chain] re-judging with the in-loop judge"
mkdir -p "$OUT/dumps_selfjudge"
cp "$OUT/eval_gpt56sol.jsonl" "$OUT/dumps_selfjudge/gpt56sol.jsonl"
API_BASE="$JUDGE" API_KEY=EMPTY MODEL_NAME=Qwen/Qwen3.6-27B \
CONC=32 MAX_TOK=2048 REASONING=omit \
REWARD_FILE="$REPO/verl/utils/reward_score/self_evolving.py" \
  python3 scripts/self_evolving/eval/rejudge_dumps_dir.py \
    "$OUT/dumps_selfjudge" "$TEST" "$OUT/rejudge_selfjudge"

echo "[chain] restarting the few-shot sweep at full concurrency"
pkill -f eval_icl_sweep.py; sleep 5
CHAT_PROVIDER=vllm REWARD_EVAL_LENIENT_ONLY=1 \
  python3 scripts/self_evolving/eval_icl_sweep.py \
    --k 0 1 2 4 8 16 32 64 128 256 --n_eval 400 --api_key "$KEY" \
    --concurrency 32 --judge_concurrency 24 --out_dir "$OUT/icl"

echo "[chain] ALL DONE"
