#!/usr/bin/env bash
# Self-evolving generation server (CPU) for the SPLIT gemma-4-31B self-improve run.
# Runs on the TRAINER node (server4); its inference backend is the REMOTE key-gated
# vLLM on server2 (point.dd.works:18187). Proposes/validates medical questions; the
# trainer reads them from localhost:8005.
set -xeuo pipefail
export CHAT_PROVIDER=vllm
export GEN_CHAT_TIMEOUT="${GEN_CHAT_TIMEOUT:-1800}"
API_BASE="${API_BASE:-http://point.dd.works:18187/v1}"
GEN_HOST_KEY="${GEN_HOST_KEY:-$(cat /scratch/sheng/self_evolving/.gen_host_key)}"
LOG_DIR=/scratch/sheng/self_evolving/logs_si_gemma31b_split; mkdir -p "$LOG_DIR"
cd /scratch/sheng/self_evolving/verl
/usr/local/bin/python scripts/self_evolving/generation_server.py \
  --seeds_path /scratch/sheng/self_evolving/mimiciv_rare/train.jsonl \
  --test_seeds_path /scratch/sheng/self_evolving/mimiciv_rare/test.jsonl \
  --direct_target "${DIRECT_TARGET:-0.30}" --gen_train_target "${GEN_TRAIN_TARGET:-0.35}" --gen_test_target "${GEN_TEST_TARGET:-0.35}" \
  --api_base "$API_BASE" --api_key "$GEN_HOST_KEY" \
  --model_name google/gemma-4-31B-it \
  --embed_api_base http://mib.media.mit.edu:18001/v1 --embed_model Qwen/Qwen3-VL-Embedding-2B \
  --milvus_uri http://mib.media.mit.edu:19531 --milvus_token root:Milvus \
  --milvus_collection medical_knowledge --milvus_top_k 16 \
  --n_queries 10 --questions_per_query 1 --accuracy_window 64 --max_pool_size 512 --workers 16 \
  --log_dir "$LOG_DIR" --host 0.0.0.0 --port "${GEN_PORT:-8005}" \
  --teacher_retries 2 --teacher_max_tokens 4096 --milvus_history_collection gen_history_gemma31b_split \
  2>&1 | tee "$LOG_DIR/gen.log"
