#!/usr/bin/env bash
# Evaluate a GPT model on MedThinkVQA (720-case multi-image radiology DDx) via a
# TRAPI / OpenAI-compatible proxy. No GPU needed (frontier API model).
#
# Faithful reconstruction of the held-out MedThinkVQA final-answer eval; see the
# header of medthinkvqa_eval.py for the faithfulness caveats.
#
#   MODEL=gpt-5.1_2025-11-13 EFFORT=medium bash run_medthinkvqa_trapi.sh
#   MODEL=gpt-5.1_2025-11-13 LIMIT=5 bash run_medthinkvqa_trapi.sh          # smoke test
set -xeuo pipefail

MODEL="${MODEL:-gpt-5.1_2025-11-13}"
BASE="${TRAPI_BASE:-http://point.dd.works:18890/v1}"
KEY="${TRAPI_KEY:-sk-xMByFeWLKB87wZ}"
EFFORT="${EFFORT:-medium}"
DATA_DIR="${DATA_DIR:-/scratch/dvd/medthinkvqa}"
CONC="${CONC:-8}"
MAXTOK="${MAXTOK:-16000}"
LIMIT="${LIMIT:-0}"
PY="${PY:-/home/dvd/miniconda3/envs/new2/bin/python}"

cd "$(dirname "$0")"
OUT="${OUT:-$(pwd)/results_medthinkvqa}"

exec "$PY" medthinkvqa_eval.py \
    --data-dir "$DATA_DIR" \
    --model-name "$MODEL" --base-url "$BASE" --api-key "$KEY" \
    --reasoning-effort "$EFFORT" --max-output-tokens "$MAXTOK" \
    --concurrency "$CONC" --limit "$LIMIT" --output-dir "$OUT" "$@"
