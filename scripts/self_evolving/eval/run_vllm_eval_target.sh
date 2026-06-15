#!/usr/bin/env bash
# Generic vLLM launcher for a single eval target model on the vps3 MIT
# cluster node (4xH200, conda env cu130 already has vllm). frpc on the
# node already maps local 8188 -> vps3.dd.works:18189, so external
# clients (e.g. the trainer pod running eval_sota.py) hit
# http://vps3.dd.works:18189/v1.
#
# Usage:
#   MODEL=google/gemma-4-31B-it TP=4 bash run_vllm_eval_target.sh
#   MODEL=google/gemma-4-E4B-it TP=1 bash run_vllm_eval_target.sh
#   MODEL=google/medgemma-1.5-4b-it TP=1 bash run_vllm_eval_target.sh
#
# All knobs are env-overridable (PORT, MAX_MODEL_LEN, GPU_MEM_UTIL, etc).

set -eu

MODEL="${MODEL:?set MODEL=<hf-repo-id>}"
PORT="${PORT:-8188}"
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export HF_HOME="${HF_HOME:-$HOME/scratch/dvdai/huggingface}"
LOG_FILE="${LOG_FILE:-/tmp/vllm_eval_$(echo "$MODEL" | tr '/:' '__').log}"

# Faster HF download for first-time model pulls if hf_transfer is installed.
if python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

echo "=== launching vllm at $(date) ==="
echo "    MODEL=$MODEL  PORT=$PORT  TP=$TP"
echo "    MAX_NUM_SEQS=$MAX_NUM_SEQS  MAX_MODEL_LEN=$MAX_MODEL_LEN  GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "    HF_HOME=$HF_HOME  LOG=$LOG_FILE"

exec vllm serve "$MODEL" \
  -tp "$TP" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code 2>&1 | tee "$LOG_FILE"
