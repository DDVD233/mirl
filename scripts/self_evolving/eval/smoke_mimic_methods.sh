#!/usr/bin/env bash
# Exercise every live method on a synthetic fixture, never a validation subset.
set -euo pipefail
S=${S:-/scratch/sheng/self_evolving}
CODE=$S/stage1_method_baselines/code
REPO=$S/verl
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
OUT=${OUT:-$S/stage1_method_baselines/synthetic9b}
args=(--val-file "$CODE/mimic_methods_synthetic.jsonl" --expected-cases 1
      --out-dir "$OUT" --model "$MODEL" --base-url http://127.0.0.1:8188/v1
      --eval-module "$REPO/scripts/self_evolving/eval_sota.py"
      --reward-file "$REPO/verl/utils/reward_score/self_evolving.py"
      --max-tokens 512 --aux-tokens 256)
IFS= read -r TRAPI_API_KEY < "$S/.trapi_key" || true
export TRAPI_API_KEY
python "$CODE/mimic_method_baselines.py" generate "${args[@]}"
python "$CODE/mimic_method_baselines.py" grade "${args[@]}"
