#!/usr/bin/env bash
# Launch the TRAPI reverse proxy (scripts/self_evolving/trapi_proxy.py) in a
# detached tmux session on point.dd.works so it survives SSH disconnects.
#
# Downstream (OPD teacher, reward/judge) then point at this instead of the
# local vLLM teacher:
#     export TEACHER_URL=http://point.dd.works:18890/v1
#     export API_BASE=http://point.dd.works:18890/v1
#     export TEACHER_MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
#     export MODEL_NAME=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
#
# Prereq on this host: az login --scope api://trapi/.default
set -euo pipefail

SESSION="${SESSION:-trapi_proxy}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/svl/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
export PROXY_PORT="${PROXY_PORT:-18890}"
export TRAPI_UPSTREAM="${TRAPI_UPSTREAM:-https://trapi.research.microsoft.com/redmond/interactive/openai/v1}"
export TRAPI_SCOPE="${TRAPI_SCOPE:-api://trapi/.default}"
# Shared secret clients must present (Authorization: Bearer <key>). Strongly
# recommended once the port is reachable externally. Downstream sets this as
# TEACHER_API_KEY / API_KEY. Empty = open mode (LAN/localhost only).
export PROXY_API_KEY="${PROXY_API_KEY:-}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already running. Attach: tmux attach -t $SESSION"
    echo "Stop it first with: tmux kill-session -t $SESSION"
    exit 1
fi

# Verify we can mint a token before launching, so failures are obvious.
if ! az account get-access-token --scope "$TRAPI_SCOPE" --query expiresOn -o tsv >/dev/null 2>&1; then
    echo "ERROR: cannot acquire a TRAPI token. Run: az login --scope $TRAPI_SCOPE" >&2
    exit 1
fi

tmux new-session -d -s "$SESSION" \
    "PROXY_HOST='$PROXY_HOST' PROXY_PORT='$PROXY_PORT' \
     TRAPI_UPSTREAM='$TRAPI_UPSTREAM' TRAPI_SCOPE='$TRAPI_SCOPE' \
     PROXY_API_KEY='$PROXY_API_KEY' \
     '$PYTHON_BIN' '$SCRIPT_DIR/trapi_proxy.py' 2>&1 | tee -a /tmp/trapi_proxy.log"

echo "Started TRAPI proxy in tmux session '$SESSION' on ${PROXY_HOST}:${PROXY_PORT}"
echo "  upstream : $TRAPI_UPSTREAM"
if [ -n "$PROXY_API_KEY" ]; then
    echo "  auth     : client key REQUIRED (set TEACHER_API_KEY/API_KEY to it downstream)"
else
    echo "  auth     : OPEN — set PROXY_API_KEY before exposing this port externally"
fi
echo "  logs     : tmux attach -t $SESSION   (or tail -f /tmp/trapi_proxy.log)"
echo "  health   : curl -s http://localhost:${PROXY_PORT}/health"
echo "  stop     : tmux kill-session -t $SESSION"
