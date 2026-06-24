#!/usr/bin/env bash
# Self-evolving question-generation server on SERVER 4 for the gpt-5.3-judge exp.
#   - question generator (teacher) = gpt-5.3 via the TRAPI proxy (point.dd.works:18890)
#   - embedding server + Milvus knowledge DB (medical_knowledge_v2) on mib
#   - serves on :8007 (matches the trainer's gen_server_url=localhost:8007)
# Run in a tmux window BEFORE the trainer; the trainer blocks on /healthz.
set -xeuo pipefail

export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
export API_BASE="${API_BASE:-http://point.dd.works:18890/v1}"
export API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key)}"
export MODEL_NAME="${MODEL_NAME:-gpt-5.3-chat_2026-03-03}"
export CHAT_PROVIDER="${CHAT_PROVIDER:-trapi}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
export MILVUS_TOKEN="${MILVUS_TOKEN:-root:Milvus}"
export MILVUS_COLLECTION="${MILVUS_COLLECTION:-medical_knowledge_v2}"
export MILVUS_TOP_K="${MILVUS_TOP_K:-16}"
export GEN_SERVER_PORT="${GEN_SERVER_PORT:-8007}"
export LOG_DIR="${LOG_DIR:-/scratch/sheng/self_evolving/logs_gptjudge_s4}"
mkdir -p "$LOG_DIR"

exec bash "$(dirname "$0")/start_generation_server.sh"
