#!/usr/bin/env bash
# Run only after stopping the corresponding original evaluation window.
set -euo pipefail
S=${S:-/scratch/sheng/self_evolving}
REVISION=$S/stage1_method_baselines_v2
TAG=${TAG:-qwen35_9b}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
BASE_URL=${BASE_URL:-http://127.0.0.1:8188/v1}
OUT=$REVISION/$TAG
if [[ ! -e $OUT/manifest.json ]]; then
    python "$REVISION/code/migrate_mimic_methods_v2.py" \
        --source "$S/stage1_method_baselines/$TAG" \
        generate --val-file "$S/mimiciv_rare/test.jsonl" --out-dir "$OUT" \
        --model "$MODEL" --base-url "$BASE_URL" \
        --eval-module "$S/verl/scripts/self_evolving/eval_sota.py" \
        --reward-file "$S/verl/verl/utils/reward_score/self_evolving.py"
fi
export TAG MODEL BASE_URL OUT
export RUNNER=$REVISION/code/mimic_method_baselines.py
exec bash "$REVISION/code/run_mimic_methods.sh"
