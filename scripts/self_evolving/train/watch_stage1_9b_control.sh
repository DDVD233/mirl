#!/usr/bin/env bash
set -euo pipefail
S=${S:-/scratch/sheng/self_evolving}
IFS= read -r TRAPI_API_KEY < "$S/.trapi_key" || true
export TRAPI_API_KEY
exec python "$S/stage1_component_audit/code/watch_stage1_control_grades.py" \
  --test "$S/logs_stage1_9b_control/data_images_fixed/test.jsonl" \
  --traces "$S/logs_stage1_9b_control/val_generations/mimiciv_rare_qwen35_9b_trainset_selfjudge" \
  --output "$S/logs_stage1_9b_control/fixed_judge"
