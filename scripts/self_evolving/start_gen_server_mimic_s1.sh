#!/usr/bin/env bash
# Launch the self-evolving question-generation server on SERVER 1 for the
# mimiciv_rare 27B self-improve RETRAIN. Wires the server-1 environment:
#   - teacher (Qwen3.6-27B) = server 5, external http://point.dd.works:18184
#   - embedding server + Milvus knowledge DB = mib
#   - serves on :8006 (matches the trainer's gen_server_url=localhost:8006)
# Run in a tmux window BEFORE the trainer; the trainer's SelfEvolvingDataset
# blocks on /healthz until this is reachable.
set -xeuo pipefail

export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
export API_BASE="${API_BASE:-http://point.dd.works:18184/v1}"
export API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.climb_teacher_key)}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
export MILVUS_TOKEN="${MILVUS_TOKEN:-root:Milvus}"
export MILVUS_COLLECTION="${MILVUS_COLLECTION:-medical_knowledge}"
export MILVUS_TOP_K="${MILVUS_TOP_K:-16}"
export GEN_SERVER_PORT="${GEN_SERVER_PORT:-8006}"
export LOG_DIR="${LOG_DIR:-/scratch/sheng/self_evolving/logs_si_qwen36_retrain}"
mkdir -p "$LOG_DIR"

exec bash "$(dirname "$0")/start_generation_server.sh"
