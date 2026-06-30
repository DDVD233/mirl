#!/usr/bin/env bash
# Re-judge the baseline generation dumps with gpt-chat-latest via the gcr proxy
# (identical judge call as the 28 trained-checkpoint dumps) -> directly
# comparable overall + per-ICD-category accuracy.
set -uo pipefail
cd /scratch/sheng/self_evolving/verl
export API_BASE=http://point.dd.works:18890/v1
export API_KEY=$(cat /scratch/sheng/self_evolving/.trapi_key)
export MODEL_NAME=gpt-chat-latest_2026-05-28
export CONC=64 MAX_TOK=2048 REASONING=omit
/usr/local/bin/python scripts/self_evolving/eval/rejudge_dumps_dir.py \
  /scratch/sheng/self_evolving/eval_baselines/out \
  /scratch/sheng/self_evolving/mimiciv_rare/test.jsonl \
  /scratch/sheng/self_evolving/eval_baselines/rejudge_chatlatest \
  2>&1 | tee -a /scratch/sheng/self_evolving/eval_baselines/rejudge_baselines.log
