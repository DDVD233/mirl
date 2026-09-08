#!/usr/bin/env bash
# Resume only unfinished arms, retry transient failures, and finish full grading.
set -euo pipefail
S=${S:-/scratch/sheng/self_evolving}
ROOT=$S/stage1_method_baselines_v2
REPO=$S/verl
RUNNER=$ROOT/code/failure_repair/mimic_method_baselines.py
TAG=${TAG:-qwen35_9b}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
CONCURRENCY=${CONCURRENCY:-64}
test "$(git -C "$REPO" rev-parse --short=8 HEAD)" = 01c43aa6
IFS= read -r TRAPI_API_KEY < "$S/.trapi_key" || true
export TRAPI_API_KEY
args=(--val-file "$S/mimiciv_rare/test.jsonl" --out-dir "$ROOT/$TAG"
      --model "$MODEL" --base-url http://127.0.0.1:8188/v1
      --eval-module "$REPO/scripts/self_evolving/eval_sota.py"
      --reward-file "$REPO/verl/utils/reward_score/self_evolving.py"
      --resume-failure-repair)
if [[ $TAG == qwen35_9b ]]; then
    methods=(cove imedrag)
else
    methods=(rag_fusion imedrag)
fi
while true; do
    remaining=0
    for method in "${methods[@]}"; do
        [[ -f $ROOT/$TAG/$method.summary.json ]] && continue
        if python "$RUNNER" generate "${args[@]}" --methods "$method" --concurrency "$CONCURRENCY"; then
            if python "$RUNNER" grade "${args[@]}" --methods "$method" --concurrency 32; then
                continue
            fi
        fi
        remaining=$((remaining + 1))
        echo "Retry required: $TAG/$method. Successful cases are retained."
    done
    if [[ $remaining == 0 ]]; then
        echo "ALL METHODS COMPLETE: $TAG"
        exit 0
    fi
    sleep 60
done
