#!/usr/bin/env bash
# Re-judge a run's validation dumps with the external judge of record, as they land.
#
# The frozen training checkout has no separate validation-judge hook: compute_score
# takes one judge endpoint and uses it for both rollout rewards and validation. So the
# training run keeps judging with the model itself, verl dumps every validation
# generation, and this re-scores those dumps with gpt-chat-latest -- the judge every
# other MIMIC-rare number in the paper was produced with.
#
# Doing it offline rather than live is deliberate: the same treatment then applies to
# the already-finished self-evolving run, whose dumps still exist, so both curves end
# up on one judge instead of one curve being live-judged and the other re-judged.
#
# Resumable and idempotent: rejudge_val_traces.py skips any dump already scored.
#
#   RUN_DIR=/scratch/sheng/self_evolving/logs_baseline/val_generations/<exp> \
#   OUT_DIR=/scratch/sheng/self_evolving/rejudge/<exp> \
#   TRAPI_KEY=... bash scripts/self_evolving/rejudge_val_watch.sh
set -uo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl}
RUN_DIR=${RUN_DIR:?set RUN_DIR to the validation_data_dir of the run}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
TEST=${TEST:-/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl}
KEY=${TRAPI_KEY:?set TRAPI_KEY}
JUDGE=${JUDGE:-gpt-chat-latest_2026-05-28}
API_BASE_=${API_BASE_:-http://point.dd.works:18890/v1}
CONC_=${CONC_:-16}          # modest: the proxy is shared with the few-shot sweep
INTERVAL=${INTERVAL:-900}
ONCE=${ONCE:-0}

mkdir -p "$OUT_DIR"
cd "$REPO"

while :; do
    mapfile -t dumps < <(find "$RUN_DIR" -name '*.jsonl' -type f 2>/dev/null | sort)
    if [ "${#dumps[@]}" -gt 0 ]; then
        echo "[rejudge] $(date -Is) ${#dumps[@]} dumps present; scoring new ones with $JUDGE"
        API_BASE="$API_BASE_" API_KEY="$KEY" MODEL_NAME="$JUDGE" \
        CONC="$CONC_" MAX_TOK=2048 REASONING=omit \
        REWARD_FILE="$REPO/verl/utils/reward_score/self_evolving.py" \
          python3 scripts/self_evolving/eval/rejudge_val_traces.py \
            "$TEST" "$OUT_DIR" "${dumps[@]}"
    else
        echo "[rejudge] $(date -Is) no dumps yet under $RUN_DIR"
    fi
    [ "$ONCE" = "1" ] && break
    sleep "$INTERVAL"
done
