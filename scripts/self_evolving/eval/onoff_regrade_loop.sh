#!/usr/bin/env bash
# Keep regrading, under both paper graders, every dump the retrieval on/off table
# needs -- the in-loop dumps that exist already, the inference-time method dumps,
# and the val-only dumps as the on/off queue produces them. Re-runs the sweep
# driver every ten minutes over the dumps that exist and stops when every expected
# output is present. Outputs land in paper_refresh/regrade/<tag>/<run>_<step>.json.
#
#   bash scripts/self_evolving/eval/onoff_regrade_loop.sh   (on pod 2333, any time)
set -uo pipefail
S=/scratch/sheng/self_evolving
P=$S/paper_refresh
V=$S/logs_hb9b/val_generations
A=${V}_aicr
cd "$P"

# run step dump  (the on/off queue writes step 0 for every val-only item)
EXPECTED="
hb27b_specgap_ship_retrieval_websearch 200 $V/hb27b_specgap_ship_retrieval_websearch/200.jsonl
hb27b_specgap_simple_retrieval_aicr 60 $A/hb27b_specgap_simple_retrieval_aicr/60.jsonl
hb9b_specgap_ship_retrieval_self9b_websearch 0 $V/hb9b_specgap_ship_retrieval_self9b_websearch/0.jsonl
hb9b_specgap_ship_retrieval_self9b_websearch 145 $V/hb9b_specgap_ship_retrieval_self9b_websearch/145.jsonl
hb9b_specgap_ship_retrieval_self9b_websearch 200 $V/hb9b_specgap_ship_retrieval_self9b_websearch/200.jsonl
hb9b_specgap_simple_retrieval_self9b_websearch 75 $V/hb9b_specgap_simple_retrieval_self9b_websearch/75.jsonl
hb9b_specgap_simple_retrieval_self9b_websearch 80 $V/hb9b_specgap_simple_retrieval_self9b_websearch/80.jsonl
hb9b_specgap_ship_noretrieval_sj_aicr 0 $A/hb9b_specgap_ship_noretrieval_sj_aicr/0.jsonl
hb9b_specgap_ship_noretrieval_sj_aicr 460 $A/hb9b_specgap_ship_noretrieval_sj_aicr/460.jsonl
hb9b_specgap_ship_noretrieval_sj_aicr 480 $A/hb9b_specgap_ship_noretrieval_sj_aicr/480.jsonl
hb9b_specgap_simple_noretrieval_sj_aicr 345 $A/hb9b_specgap_simple_noretrieval_sj_aicr/345.jsonl
hb9b_specgap_simple_noretrieval_sj_aicr 860 $A/hb9b_specgap_simple_noretrieval_sj_aicr/860.jsonl
hbpro_methods_qwen35_9b_direct 0 $V/hbpro_methods_qwen35_9b_direct/0.jsonl
hbpro_methods_qwen35_9b_medrag 0 $V/hbpro_methods_qwen35_9b_medrag/0.jsonl
hbpro_methods_qwen35_9b_rag_fusion 0 $V/hbpro_methods_qwen35_9b_rag_fusion/0.jsonl
hbpro_methods_qwen35_9b_imedrag 0 $V/hbpro_methods_qwen35_9b_imedrag/0.jsonl
hbpro_methods_qwen36_27b_direct 0 $V/hbpro_methods_qwen36_27b_direct/0.jsonl
hbpro_methods_qwen36_27b_medrag 0 $V/hbpro_methods_qwen36_27b_medrag/0.jsonl
hbpro_methods_qwen36_27b_rag_fusion 0 $V/hbpro_methods_qwen36_27b_rag_fusion/0.jsonl
hbpro_methods_qwen36_27b_imedrag 0 $V/hbpro_methods_qwen36_27b_imedrag/0.jsonl
hbpro_27b_ser200_notools 0 $V/hbpro_27b_ser200_notools/0.jsonl
hbpro_27b_fixed60_notools 0 $V/hbpro_27b_fixed60_notools/0.jsonl
hbpro_27b_base_notools 0 $V/hbpro_27b_base_notools/0.jsonl
hbpro_9bD_ser480_tools 0 $V/hbpro_9bD_ser480_tools/0.jsonl
hbpro_9bD_fixed860_tools 0 $V/hbpro_9bD_fixed860_tools/0.jsonl
hbpro_9bC_ser200_notools 0 $V/hbpro_9bC_ser200_notools/0.jsonl
hbpro_9bC_fixed80_notools 0 $V/hbpro_9bC_fixed80_notools/0.jsonl
"
# The method dumps have their own row counts; every HB-Pro dump must be complete
# (525 rows) before it is graded, or a half-written val-only dump gets a partial score.
while :; do
    python3 - <<EOF
import json, os
rows = []
for line in """$EXPECTED""".strip().splitlines():
    run, step, dump = line.split()
    if os.path.exists(dump) and sum(1 for _ in open(dump)) >= 525:
        rows.append({"run": run, "step": int(step), "dump": dump})
json.dump(rows, open("regrade_rows_onoff_live.json", "w"), indent=1)
print(len(rows), "gradable dumps")
EOF
    MODEL=gpt-chat-latest_2026-05-28 EFFORT=omit TAG=chatlatest_std ROWS=regrade_rows_onoff_live.json CONC=24 bash run_regrade_sweep_v2.sh
    MODEL=gpt-5.4_2026-03-05 EFFORT=low TAG=gpt54low ROWS=regrade_rows_onoff_live.json CONC=24 bash run_regrade_sweep_v2.sh
    missing=0
    while read -r run step dump; do
        [ -z "$run" ] && continue
        for tag in chatlatest_std gpt54low; do
            [ -f "regrade/$tag/${run}_${step}.json" ] || missing=$((missing+1))
        done
    done <<< "$EXPECTED"
    echo "=== $(date -u +%FT%TZ) outputs still missing: $missing"
    [ "$missing" -eq 0 ] && { echo "ONOFF_REGRADE_LOOP_DONE $(date -u +%FT%TZ)"; break; }
    sleep 600
done
