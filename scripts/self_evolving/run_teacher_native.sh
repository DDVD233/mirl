#!/usr/bin/env bash
# Serve the OPD teacher (Qwen3.5-397B-A17B-FP8) on a K8s pod whose image
# already includes vLLM (zjdavid/verl-selfevolving:cu130-vllm0.22.1). No apptainer —
# just `vllm serve` in the host environment.
#
# Defaults match the new sheng-evolving2 pod: frpc maps the container's
# local 8188 -> public point.dd.works:18187, so external clients should
# hit http://point.dd.works:18187/v1 (which matches the OPD config's
# TEACHER_URL default).
#
# Overridable env:
#   MODEL, PORT, DP, TP, MAX_NUM_SEQS, GPU_MEM_UTIL, HF_HOME, LOG_FILE
#
# gpu_memory_utilization defaults to 0.6 so requests with prompt_logprobs
# (the OPD client) have prefill activation headroom. 0.95 OOM'd inside
# dump_input.py on the prior teacher when the trainer's end-of-step
# burst of teacher calls landed on top of gen+judge traffic.

set -eu

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-8188}"
DP="${DP:-2}"
TP="${TP:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
# The model's config says max_position_embeddings=262144 (256K context),
# which makes vLLM size KV cache per request for a 256K sequence. Cap
# here to the largest window we actually use:
#   - OPD client: prompt 8192 + response 4096 = 12288 tokens
#   - gen-server proposer/judge: chat-completion with max_tokens=12288
#     and a long system+context prompt, so we need ~12288 + ~16K of
#     prompt budget = ~28K. 32768 leaves margin.
# 0.70 util keeps enough KV for one full-window request at 32K and still
# leaves prefill activation headroom for `prompt_logprobs` bursts.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Persist the model in the shared PVC so a teacher restart on either pod
# reuses the same 400GB FP8 weights.
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
LOG_FILE="${LOG_FILE:-/scratch/sheng/self_evolving/logs/teacher_vllm.log}"

# 5-10x faster HF download for the 400GB+ FP8 weights. Falls back silently
# to the standard downloader if hf_transfer isn't installed.
if python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

mkdir -p "$(dirname "$LOG_FILE")"

echo "=== launching vllm at $(date) ==="
echo "    MODEL=$MODEL  PORT=$PORT  DP=$DP TP=$TP"
echo "    MAX_NUM_SEQS=$MAX_NUM_SEQS  GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "    HF_HOME=$HF_HOME  LOG_FILE=$LOG_FILE"

exec vllm serve "$MODEL" \
  -dp "$DP" \
  -tp "$TP" \
  --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code 2>&1 | tee "$LOG_FILE"
