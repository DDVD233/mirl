#!/usr/bin/env bash
# Frontier sweep for ONE model (dvd 2026-09-14): PRBench Hard and ProfBench at three
# reasoning efforts with the solver's web tool, each dump graded right away with the
# stage-2 regrade script under gpt-chat-latest (one vote), the instrument every
# comparison row of the paper uses. Run one instance per model on pod 2333:
#   MODEL=gpt-5.6-luna_2026-07-09 bash scripts/self_evolving/eval/frontier_sweep.sh
# The web tool is the dedicated cache server on :8057 (started with an ABSOLUTE script
# path so the cross-domain queue's cleanup pattern does not kill it).
set -uo pipefail
S=/scratch/sheng/self_evolving
MODEL="${MODEL:?}"
SHORT=$(echo "$MODEL" | sed 's/gpt-5.6-//; s/_2026-07-09//')
OUT=$S/paper_refresh/frontier; mkdir -p "$OUT/grade"
KEY=$(cat $S/.trapi_key)
cd $S/verl_specgap
for effort in ${EFFORTS:-low medium high}; do
  for bench in ${BENCHES:-profbench prbench}; do
    case $bench in
      profbench) PQ=$S/profbench_val.parquet; MS=2 ;;
      prbench)   PQ=$S/prbench_hard_val.parquet; MS=4 ;;
    esac
    name=${bench}_${SHORT}_${effort}
    dump=$OUT/$name.jsonl
    if [ ! -f "$dump" ]; then
      echo "=== $(date -u +%FT%TZ) GEN $name"
      SE_DOMAIN=$bench python3 scripts/self_evolving/eval/frontier_tool_eval.py --parquet "$PQ" --tools web \
          --max-searches $MS --model "$MODEL" --effort "$effort" --search-url http://localhost:8057/search \
          --concurrency "${CONC:-16}" --out "$dump" 2>&1 | grep -v -i "warn" | tail -3
    fi
    if [ -f "$dump" ] && [ ! -f "$OUT/grade/$name.json" ]; then
      echo "=== $(date -u +%FT%TZ) GRADE $name"
      python3 scripts/self_evolving/analysis/regrade_hbpro_dumps.py --dump "$dump" --val-parquet "$PQ" \
          --api-key "$KEY" --model gpt-chat-latest_2026-05-28 --effort omit --votes 1 --concurrency 24 \
          --out "$OUT/grade/$name.json" 2>&1 | grep -vE "^\s+[0-9]+/[0-9]+ calls" | tail -2
      python3 -c "import json; d=json.load(open('$OUT/grade/$name.json')); o=d['overall']; print('RESULT $name raw_signed=%.3f raw=%.3f join=%s' % (o['acc_raw_signed'], o['acc_raw'], d.get('join_verified')))"
    fi
  done
done
echo "SWEEP_DONE $SHORT $(date -u +%FT%TZ)"
