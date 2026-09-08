#!/usr/bin/env bash
# 2334: stage-1-style inference baselines on HB-Pro for the frozen 9B and 27B, then grade.
set -u
S=/scratch/sheng/self_evolving; cd $S/verl_specgap
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
LOGD=$S/logs/hbpro_methods; mkdir -p $LOGD
serve () { # tag model gpus tp port
  CUDA_VISIBLE_DEVICES="$3" nohup vllm serve "$2" --served-model-name "$1" --host 0.0.0.0 --port "$5" --trust-remote-code \
    --tensor-parallel-size "$4" --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-seqs 64 \
    --reasoning-parser qwen3 --mm-encoder-attn-backend TORCH_SDPA > $LOGD/serve_$1.log 2>&1 &
  for i in $(seq 1 240); do curl -sf -m 5 "http://localhost:$5/v1/models" >/dev/null 2>&1 && return 0; sleep 10; done; return 1
}
serve qwen35_9b Qwen/Qwen3.5-9B 0 1 8210 || { echo "FATAL 9B serve"; exit 1; }
serve qwen36_27b Qwen/Qwen3.6-27B 2,3 2 8211 || { echo "FATAL 27B serve"; exit 1; }
LIMIT="${LIMIT:-0}"; METHODS="${METHODS:-direct medrag rag_fusion imedrag}"
python3 scripts/self_evolving/eval/hbpro_method_baselines.py --model qwen35_9b --base-url http://localhost:8210/v1 \
   --out-prefix hbpro_methods_qwen35_9b --methods $METHODS --limit $LIMIT --concurrency 12 > $LOGD/gen_qwen35_9b.log 2>&1 &
P1=$!
python3 scripts/self_evolving/eval/hbpro_method_baselines.py --model qwen36_27b --base-url http://localhost:8211/v1 \
   --out-prefix hbpro_methods_qwen36_27b --methods $METHODS --limit $LIMIT --concurrency 12 > $LOGD/gen_qwen36_27b.log 2>&1 &
P2=$!
wait $P1 $P2
echo "=== $(date -u +%FT%TZ) generation done"
pkill -9 -f "^/usr/bin/python3 /usr/local/bin/vllm serve" 2>/dev/null; pkill -9 -f "^VLLM::EngineCore" 2>/dev/null; sleep 10
if [ "$LIMIT" = 0 ]; then
  KEY=$(cat $S/.trapi_key); mkdir -p $S/paper_refresh/regrade/methods
  for m in qwen35_9b qwen36_27b; do for meth in $METHODS; do
    d=$S/logs_hb9b/val_generations/hbpro_methods_${m}_${meth}/0.jsonl
    [ -f "$d" ] || continue
    python3 $S/paper_refresh/regrade_hbpro_dumps.py --dump "$d" --api-key "$KEY" --model gpt-chat-latest_2026-05-28 \
      --effort omit --votes 3 --concurrency 16 --out $S/paper_refresh/regrade/methods/${m}_${meth}.json 2>&1 | grep -vE "^\s+[0-9]+/" | tail -4
  done; done
fi
echo "=== $(date -u +%FT%TZ) METHODS DONE"
