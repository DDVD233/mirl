#!/usr/bin/env bash
# Launch the self-evolving generation server in RUBRIC MODE (HealthBench-
# Professional task + co-generated rubric) for the server-4 run. Wires:
#   - teacher/self-judge (Qwen3.6-27B) = server 5, http://point.dd.works:18184
#   - embedding server + Milvus knowledge DB = mib (grounding is best-effort)
#   - serves on :8006 (matches the trainer's gen_server_url=localhost:8006)
#   - evolvable, file-backed prompts under $PROMPT_DIR
# In rubric mode the server synthesizes its use_case x specialty seeds (no
# train.jsonl needed) and exposes POST /evolve for end-of-step prompt evolution.
# Run in a tmux window BEFORE the trainer; SelfEvolvingDataset blocks on /healthz.
set -xeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"

API_BASE="${API_BASE:-http://point.dd.works:18184/v1}"
API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.climb_teacher_key)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
MILVUS_TOKEN="${MILVUS_TOKEN:-root:Milvus}"
MILVUS_COLLECTION="${MILVUS_COLLECTION:-medical_knowledge_v2}"
MILVUS_TOP_K="${MILVUS_TOP_K:-8}"

# Generation knobs. n_queries = clinician requests proposed per (use_case x
# specialty) seed; each is grounded + turned into one task+rubric example.
N_QUERIES="${N_QUERIES:-6}"
QUESTIONS_PER_QUERY="${QUESTIONS_PER_QUERY:-1}"
ACCURACY_WINDOW="${ACCURACY_WINDOW:-64}"
MAX_POOL_SIZE="${MAX_POOL_SIZE:-200}"
GEN_WORKERS="${GEN_WORKERS:-8}"

GEN_SERVER_PORT="${GEN_SERVER_PORT:-8006}"
GEN_SERVER_HOST="${GEN_SERVER_HOST:-0.0.0.0}"

EXP="${EXP:-healthbench_rubric_qwen36_27b_v5}"
LOG_DIR="${LOG_DIR:-/scratch/sheng/self_evolving/logs_healthbench_rubric/$EXP}"
# Evolvable prompt files live here (query_proposer.txt, task_rubric_generator.txt,
# *_guidance.txt, history/step_NNN/). Keep under the experiment dir so the trainer
# and gen server (colocated on server 4) agree on the path.
PROMPT_DIR="${PROMPT_DIR:-$LOG_DIR/prompts}"
mkdir -p "$LOG_DIR" "$PROMPT_DIR"

export CHAT_PROVIDER="${CHAT_PROVIDER:-vllm}"

cd "$(dirname "$0")/../../.."

exec "$PYTHON_BIN" scripts/self_evolving/generation_server.py \
    --rubric_mode \
    --prompt_dir "$PROMPT_DIR" \
    --api_base "$API_BASE" \
    --api_key "$API_KEY" \
    --model_name "$MODEL_NAME" \
    --embed_api_base "$EMBED_API_BASE" \
    --embed_model "$EMBED_MODEL" \
    --milvus_uri "$MILVUS_URI" \
    --milvus_token "$MILVUS_TOKEN" \
    --milvus_collection "$MILVUS_COLLECTION" \
    --milvus_top_k "$MILVUS_TOP_K" \
    --n_queries "$N_QUERIES" \
    --questions_per_query "$QUESTIONS_PER_QUERY" \
    --accuracy_window "$ACCURACY_WINDOW" \
    --max_pool_size "$MAX_POOL_SIZE" \
    --workers "$GEN_WORKERS" \
    --log_dir "$LOG_DIR" \
    --host "$GEN_SERVER_HOST" \
    --port "$GEN_SERVER_PORT" \
    "$@"
