#!/usr/bin/env bash
# Grade the GEPA / ACE HealthBench Professional dumps exactly like the other frozen-method rows:
#   regrade/chatlatest_std + regrade/gpt54low  (one vote; the two-grader "no training data" table)
#   regrade/methods                           (gpt-chat-latest, three votes; the methods table)
# Waits until every dump is complete (525 rows and the eval summary written). CPU only.
set -u
S=/scratch/sheng/self_evolving; V=$S/logs_hb9b/val_generations; O=$S/paper_refresh/context_baselines
TAGS="${TAGS:-qwen35_9b_gepa qwen35_9b_ace qwen36_27b_gepa qwen36_27b_ace}"
cd $S/paper_refresh
for t in $TAGS; do
  m=${t##*_}; model=${t%_*}
  until [ -s "$O/${m}_hbpro_${model}/eval_summary.json" ] && [ "$(wc -l < "$V/hbpro_methods_$t/0.jsonl" 2>/dev/null || echo 0)" -ge 525 ]; do sleep 120; done
  echo "=== $(date -u +%FT%TZ) dump ready: $t"
done
python3 - <<PY
import json
json.dump([{"run": "hbpro_methods_%s" % t, "step": 0, "dump": "$V/hbpro_methods_%s/0.jsonl" % t} for t in "$TAGS".split()],
          open("regrade_rows_context_baselines.json", "w"), indent=1)
PY
MODEL=gpt-chat-latest_2026-05-28 EFFORT=omit TAG=chatlatest_std ROWS=regrade_rows_context_baselines.json CONC=16 bash run_regrade_sweep_v2.sh
MODEL=gpt-5.4_2026-03-05 EFFORT=low TAG=gpt54low ROWS=regrade_rows_context_baselines.json CONC=16 bash run_regrade_sweep_v2.sh
bash $S/verl_specgap/scripts/self_evolving/eval/grade_hbpro_methods.sh $TAGS
echo "=== $(date -u +%FT%TZ) CONTEXT BASELINE GRADING DONE"
