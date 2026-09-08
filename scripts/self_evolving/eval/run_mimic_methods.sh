#!/usr/bin/env bash
# Run on an existing SSH-accessible node. Does not allocate or stop services.
set -euo pipefail
S=${S:-/scratch/sheng/self_evolving}
REPO=${REPO:-$S/verl}
RUNNER=${RUNNER:-$S/stage1_method_baselines/code/mimic_method_baselines.py}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
BASE_URL=${BASE_URL:-http://127.0.0.1:8188/v1}
TAG=${TAG:-qwen35_9b}
CONCURRENCY=${CONCURRENCY:-16}
OUT=${OUT:-$S/stage1_method_baselines/$TAG}
PYTHON=${PYTHON:-/usr/local/bin/python}
if [[ -z ${TRAPI_API_KEY:-} && -r "$S/.trapi_key" ]]; then
    IFS= read -r TRAPI_API_KEY < "$S/.trapi_key" || true
    export TRAPI_API_KEY
fi

test "$(git -C "$REPO" rev-parse --short=8 HEAD)" = 01c43aa6 || {
    echo "Expected the stage-1 checkout at 01c43aa6: $REPO" >&2; exit 1;
}
args=(--val-file "$S/mimiciv_rare/test.jsonl" --out-dir "$OUT"
      --model "$MODEL" --base-url "$BASE_URL" --concurrency "$CONCURRENCY"
      --eval-module "$REPO/scripts/self_evolving/eval_sota.py"
      --reward-file "$REPO/verl/utils/reward_score/self_evolving.py")
for attempt in $(seq 1 180); do
    if curl -fsS --max-time 5 "$BASE_URL/models" > /dev/null; then
        break
    fi
    sleep 5
done
curl -fsS --max-time 5 "$BASE_URL/models" > /dev/null
failed=0
for method in direct medrag self_consistency self_refine rag_fusion imedrag cove; do
    success=0
    for attempt in 1 2 3; do
        if "$PYTHON" "$RUNNER" generate "${args[@]}" --methods "$method"; then
            if [[ -z ${TRAPI_API_KEY:-} ]]; then
                echo "Generation complete for $method; grading needs TRAPI_API_KEY." >&2
                break
            fi
            if "$PYTHON" "$RUNNER" grade "${args[@]}" --concurrency "${JUDGE_CONCURRENCY:-32}" --methods "$method"; then
                success=1
                break
            fi
        fi
        sleep 10
    done
    if [[ $success == 0 ]]; then
        echo "INCOMPLETE $method; inspect its error artifact and resume." >&2
        failed=1
    fi
done
exit "$failed"
