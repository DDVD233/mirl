#!/usr/bin/env bash
set -euo pipefail
S=${S:-/scratch/sheng/self_evolving}
ROOT=$S/stage1_component_audit
IFS= read -r TRAPI_API_KEY < "$S/.trapi_key" || true
export TRAPI_API_KEY
if [[ ${1:-} == smoke ]]; then
    exec python "$ROOT/code/stage1_component_audit.py" --output "$ROOT/smoke_v2" \
        --audit-per-stratum 1 --seeds 2 --samples 1 --concurrency 4
fi
while ! python "$ROOT/code/stage1_component_audit.py" --output "$ROOT/results" --concurrency 8; do
    echo "Retrying only missing component-audit items in 60 seconds."
    sleep 60
done
echo "STAGE-1 COMPONENT AUDIT COMPLETE"
