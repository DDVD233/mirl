#!/usr/bin/env bash
# Grade the HB-Pro inference-method dumps (chat-latest, 3 votes), two at a time. Args: dump tags.
set -u
S=/scratch/sheng/self_evolving; KEY=$(cat $S/.trapi_key); cd $S/paper_refresh
grade_one () { python3 regrade_hbpro_dumps.py --dump $S/logs_hb9b/val_generations/hbpro_methods_$1/0.jsonl --api-key "$KEY" \
  --model gpt-chat-latest_2026-05-28 --effort omit --votes 3 --concurrency 16 --out regrade/methods/$1.json 2>&1 | grep -vE "^\s+[0-9]+/" | tail -3; }
for t in "$@"; do grade_one "$t" & while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 20; done; done; wait
echo "GRADING DONE $(date -u +%FT%TZ)"
