#!/usr/bin/env bash
# Frontier sweep, MedXpertQA (text 2450 + image 2000), ONE model, three efforts, with the
# solver's two tools (search_medical_kb via the pod's gen server /retrieve, web_search via
# its serper cache) and the medical arm's search budget of 2. Runs on a training pod whose
# gen server is up (2334), at low concurrency so the pod's summarizer keeps serving the
# arm. Grading = the same one-criterion rubric regrade (mean = exact-match accuracy).
#   MODEL=gpt-5.6-luna_2026-07-09 bash scripts/self_evolving/eval/frontier_sweep_medx.sh
set -uo pipefail
S=/scratch/sheng/self_evolving
MODEL="${MODEL:?}"
SHORT=$(echo "$MODEL" | sed 's/gpt-5.6-//; s/_2026-07-09//')
OUT=$S/paper_refresh/frontier; mkdir -p "$OUT/grade"
KEY=$(cat $S/.trapi_key)
cd $S/verl_specgap
for effort in ${EFFORTS:-low medium high}; do
  for split in ${SPLITS:-text mm}; do
    PQ=$S/medxpertqa_${split}_val.parquet
    name=medx${split}_${SHORT}_${effort}
    dump=$OUT/$name.jsonl
    if [ ! -f "$dump" ]; then
      echo "=== $(date -u +%FT%TZ) GEN $name"
      SE_DOMAIN=medxpert python3 scripts/self_evolving/eval/frontier_tool_eval.py --parquet "$PQ" --tools kb+web \
          --max-searches 2 --model "$MODEL" --effort "$effort" --search-url http://localhost:8056/search \
          --retrieve-url http://localhost:8041/retrieve --concurrency "${CONC:-4}" --out "$dump" 2>&1 | grep -v -i "warn" | tail -3
    fi
    if [ -f "$dump" ] && [ ! -f "$OUT/grade/$name.json" ]; then
      echo "=== $(date -u +%FT%TZ) GRADE $name"
      python3 scripts/self_evolving/analysis/regrade_hbpro_dumps.py --dump "$dump" --val-parquet "$PQ" \
          --api-key "$KEY" --model gpt-chat-latest_2026-05-28 --effort omit --votes 1 --concurrency "${GRADE_CONC:-6}" \
          --out "$OUT/grade/$name.json" 2>&1 | grep -vE "^\s+[0-9]+/[0-9]+ calls" | tail -1
      python3 -c "import json; d=json.load(open('$OUT/grade/$name.json')); o=d['overall']; print('RESULT $name acc=%.3f join=%s' % (o['acc_raw'], d.get('join_verified')))"
    fi
  done
done
echo "SWEEP_DONE medx $SHORT $(date -u +%FT%TZ)"
