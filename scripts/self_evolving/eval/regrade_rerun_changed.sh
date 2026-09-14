#!/usr/bin/env bash
# Re-run every existing regrade output with the current regrade_hbpro_dumps.py. The
# script keeps cached verdicts wherever the graded text is unchanged, so this costs
# only the rows whose answer construction changed (2026-09-14: budget-exhausted
# tool-loop rollouts were graded as empty answers). Grader settings come from each
# json; the val parquet from the dump's benchmark.
#
#   bash scripts/self_evolving/eval/regrade_rerun_changed.sh [CONC=6] [DIRS="..."]
set -uo pipefail
S=/scratch/sheng/self_evolving
cd $S/verl_specgap
KEY=$(cat $S/.trapi_key)
CONC="${CONC:-6}"
DIRS="${DIRS:-$S/paper_refresh/regrade/chatlatest_std $S/paper_refresh/regrade/gpt54low $S/paper_refresh/regrade/chatlatest_strict $S/paper_refresh/frontier/grade}"
for d in $DIRS; do
  for j in $d/*.json; do
    [ -f "$j" ] || continue
    read -r dump model effort votes strict < <(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(d['dump'], d['grader'], d['effort'], d['votes'], int(bool(d.get('strict'))))" "$j")
    [ -f "$dump" ] || { echo "MISSING dump for $j: $dump"; continue; }
    case "$dump" in
      *profbench*) PQ=$S/profbench_val.parquet ;;
      *prbench*)   PQ=$S/prbench_hard_val.parquet ;;
      *medxpert*mm*)   PQ=$S/medxpertqa_mm_val.parquet ;;
      *medxpert*)  PQ=$S/medxpertqa_text_val.parquet ;;
      *)           PQ=$S/healthbench_pro_val.parquet ;;
    esac
    extra=(); [ "$strict" = 1 ] && extra+=(--strict)
    echo "=== $(date -u +%FT%TZ) RERUN $(basename $d)/$(basename $j) model=$model effort=$effort votes=$votes strict=$strict"
    python3 scripts/self_evolving/analysis/regrade_hbpro_dumps.py --dump "$dump" --val-parquet "$PQ" \
        --api-key "$KEY" --model "$model" --effort "$effort" --votes "$votes" --concurrency "$CONC" "${extra[@]}" \
        --out "$j" 2>&1 | grep -E "answer source|cached verdicts|grader calls|acc_raw_signed|acc_len_adj_signed\"|Error|error" | head -6
    python3 -c "
import json,sys; d=json.load(open(sys.argv[1])); o=d['overall']
print('RESULT %s raw_signed=%.3f len_adj_signed=%.3f invalidated=%s sources=%s' % (sys.argv[2], o['acc_raw_signed'], o['acc_len_adj_signed'], d.get('verdicts_invalidated'), d.get('answer_source')))" "$j" "$(basename $d)/$(basename $j .json)"
  done
done
echo "RERUN_DONE $(date -u +%FT%TZ)"
