#!/usr/bin/env bash
# Self-evolving generation server for the Gemma-4-31B self-improve run. CPU-only;
# proposes/validates medical questions via the local gemma-4-31B vLLM (:8100).
set -xeuo pipefail
export CHAT_PROVIDER=vllm
export GEN_CHAT_TIMEOUT="${GEN_CHAT_TIMEOUT:-1800}"
LOG_DIR=/scratch/sheng/self_evolving/logs_si_gemma31b; mkdir -p "$LOG_DIR"
cd /scratch/sheng/self_evolving/verl
/usr/local/bin/python scripts/self_evolving/generation_server.py \
  --seeds_path /scratch/sheng/self_evolving/mimiciv_rare/train.jsonl \
  --test_seeds_path /scratch/sheng/self_evolving/mimiciv_rare/test.jsonl \
  --direct_target 0.30 --gen_train_target 0.35 --gen_test_target 0.35 \
  --api_base "${API_BASE:-http://localhost:8100/v1}" --api_key EMPTY \
  --model_name google/gemma-4-31B-it \
  --embed_api_base http://mib.media.mit.edu:18001/v1 --embed_model Qwen/Qwen3-VL-Embedding-2B \
  --milvus_uri http://mib.media.mit.edu:19531 --milvus_token root:Milvus \
  --milvus_collection medical_knowledge --milvus_top_k 16 \
  --n_queries 10 --questions_per_query 1 --accuracy_window 64 --max_pool_size 512 --workers 16 \
  --log_dir "$LOG_DIR" --host 0.0.0.0 --port "${GEN_PORT:-8005}" \
  --teacher_retries 2 --teacher_max_tokens 4096 --milvus_history_collection gen_history_gemma31b_si \
  2>&1 | tee "$LOG_DIR/gen.log"
