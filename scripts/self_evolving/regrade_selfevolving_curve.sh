#!/usr/bin/env bash
# Re-grade the long self-evolving run's validation curve with the pinned judge.
#
# The run logged 197 full validation passes (every 5 steps, 0-980, 2452 rows each)
# across two directories -- steps 0-500 under logs_evolve_reward and 500-980 under
# logs_selfimprove_from_sft, same experiment name, one continuous run.
#
# Its in-run numbers are not comparable to anything else: measured on one step-0 dump,
# swapping the judge prompt moves lenient accuracy by 15 points and swapping the judge
# model by 7.5, which is larger than the training effect being reported. So every curve
# in the paper is rebuilt here from the raw generations with one pinned configuration:
# the STRICT judge prompt (the frozen checkout's working tree) and gpt-chat-latest,
# the judge behind every MIMIC-rare number in the main table.
#
# STRIDE picks every Nth validation point. At stride 4 that is ~50 points x 2452 rows,
# which is plenty dense for a scaling curve and roughly a quarter of the judge calls
# that re-grading all 197 would take.
#
#   TRAPI_KEY=... bash scripts/self_evolving/regrade_selfevolving_curve.sh
set -uo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl}
KEY=${TRAPI_KEY:?set TRAPI_KEY}
JUDGE=${JUDGE:-gpt-chat-latest_2026-05-28}
API_BASE_=${API_BASE_:-http://point.dd.works:18890/v1}
CONC_=${CONC_:-12}      # shared proxy: the few-shot sweep is running too
STRIDE=${STRIDE:-4}
OUT_DIR=${OUT_DIR:-/scratch/sheng/self_evolving/rejudge/selfevolving_curve}
TEST=${TEST:-/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl}

EARLY=/scratch/sheng/self_evolving/logs_evolve_reward/val_generations/mimiciv_rare_qwen36_27b_evolve_from_sft
LATE=/scratch/sheng/self_evolving/logs_selfimprove_from_sft/val_generations/mimiciv_rare_qwen36_27b_evolve_from_sft

mkdir -p "$OUT_DIR"
cd "$REPO"

# Collect step -> file across both legs, dropping the duplicated step 500, then keep
# every STRIDE-th point. Sorted numerically so the stride is even across the run.
mapfile -t picks < <(
  { for f in "$EARLY"/*.jsonl; do echo "$(basename "$f" .jsonl) $f"; done
    for f in "$LATE"/*.jsonl; do s=$(basename "$f" .jsonl); [ "$s" = "500" ] || echo "$s $f"; done
  } | sort -n -k1,1 | awk -v st="$STRIDE" '(NR-1) % st == 0 {print $2}'
)

echo "[regrade] ${#picks[@]} validation points selected (stride $STRIDE), judge=$JUDGE"
API_BASE="$API_BASE_" API_KEY="$KEY" MODEL_NAME="$JUDGE" \
CONC="$CONC_" MAX_TOK=2048 REASONING=omit \
REWARD_FILE="$REPO/verl/utils/reward_score/self_evolving.py" \
  python3 scripts/self_evolving/eval/rejudge_val_traces.py "$TEST" "$OUT_DIR" "${picks[@]}"

echo "[regrade] done -> $OUT_DIR"
