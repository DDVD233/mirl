#!/usr/bin/env bash
# Self-evolving generation server with chat/proposer/judge on TRAPI Kimi-K2.6
# served through the local TRAPI proxy (scripts/self_evolving/trapi_proxy.py).
#
# Parameterized by env so multiple hosts share one git-tracked script. The
# sensitive proxy key is NOT stored here — pass it at launch:
#
#   API_KEY="$(cat /root/.trapi_proxy_key)" bash scripts/self_evolving/launch_kimi_gen.sh
#
# Per-host overrides (defaults match the sheng /scratch layout):
#   DATA_DIR, GEN_SERVER_PORT, GEN_WORKERS, MAX_POOL_SIZE, REPO_ROOT, PYTHON_BIN
set -e

export API_BASE="${API_BASE:-http://point.dd.works:18890/v1}"
export CHAT_PROVIDER="${CHAT_PROVIDER:-trapi}"
export MODEL_NAME="${MODEL_NAME:-Kimi-K2.6_2026-04-20}"

export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
export MILVUS_TOKEN="${MILVUS_TOKEN:-root:Milvus}"
export MILVUS_COLLECTION="${MILVUS_COLLECTION:-medical_knowledge}"
export LOG_DIR="${LOG_DIR:-/scratch/sheng/self_evolving/logs}"
export GEN_SERVER_PORT="${GEN_SERVER_PORT:-8004}"
export GEN_SERVER_HOST="${GEN_SERVER_HOST:-0.0.0.0}"
export GEN_WORKERS="${GEN_WORKERS:-32}"
export MAX_POOL_SIZE="${MAX_POOL_SIZE:-1200}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
export REPO_ROOT="${REPO_ROOT:-/scratch/sheng/self_evolving/verl}"

cd "$REPO_ROOT"
exec bash scripts/self_evolving/start_generation_server.sh "$@"
