#!/usr/bin/env bash
# TRAPI-backed generation server for the gemma-4-31B RL job (run_gemma4_31b_rl.sh).
# Mirrors the server1 /root/gen_server_rl.sh: questions via gpt-5.5 through the TRAPI
# proxy (point.dd.works:18890), :8004. Used when the RL job runs on a node (e.g. server4).
set -xeuo pipefail
export CHAT_PROVIDER=trapi
export GEN_CHAT_TIMEOUT="${GEN_CHAT_TIMEOUT:-1800}"
API_BASE="${API_BASE:-http://point.dd.works:18890/v1}"
API_KEY="${API_KEY:-$(cat /scratch/sheng/self_evolving/.trapi_key 2>/dev/null || echo sk-xMByFeWLKB87wZ)}"
LOG_DIR=/scratch/sheng/self_evolving/logs_rl_gemma31b_trapi; mkdir -p "$LOG_DIR"
cd /scratch/sheng/self_evolving/verl
/usr/local/bin/python scripts/self_evolving/generation_server.py \
  --seeds_path /scratch/sheng/self_evolving/mimiciv_rare/train.jsonl \
  --test_seeds_path /scratch/sheng/self_evolving/mimiciv_rare/test.jsonl \
  --direct_target 0 --gen_train_target 0.5 --gen_test_target 0.5 \
  --api_base "$API_BASE" --api_key "$API_KEY" \
  --model_name "${GEN_MODEL:-gpt-5.5_2026-04-24}" \
  --embed_api_base http://mib.media.mit.edu:18001/v1 --embed_model Qwen/Qwen3-VL-Embedding-2B \
  --milvus_uri http://mib.media.mit.edu:19531 --milvus_token root:Milvus \
  --milvus_collection medical_knowledge --milvus_top_k 16 \
  --n_queries 10 --questions_per_query 1 --accuracy_window 64 --max_pool_size 2400 --workers 16 \
  --log_dir "$LOG_DIR" --host 0.0.0.0 --port "${GEN_PORT:-8004}" \
  2>&1 | tee "$LOG_DIR/gen.log"
