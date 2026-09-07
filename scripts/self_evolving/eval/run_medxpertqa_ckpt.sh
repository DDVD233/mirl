#!/usr/bin/env bash
# MedXpertQA (text) for merged HF checkpoints and base models, one after another, on a
# subset of the pod's GPUs: serve with vLLM, wait for health, run the standalone runner
# (official exact-match on the boxed letter), tear the server down, next model.
#
# Same runner and prompt as run_medxpertqa_baseline.sh (the frozen-9B row). The budget
# is larger than that row's 14000 because we control max_model_len here: the 27B
# thinks for long, and a truncated reasoning scores zero, so the honest comparison
# between base and trained 27B needs the same generous budget for both.
#
# Usage (on an MSR pod, repo root = verl_specgap):
#   GPUS=2,3 TP=2 bash scripts/self_evolving/eval/run_medxpertqa_ckpt.sh \
#       hb27b_ser_step200=/scratch/sheng/self_evolving/checkpoints/hf_merged/hb27b_ser_step200 \
#       base27b=Qwen/Qwen3.6-27B
set -u
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
S=/scratch/sheng/self_evolving
OUT="${OUT:-$S/eval_out}"
GPUS="${GPUS:-2,3}"; TP="${TP:-2}"; PORT="${PORT:-8200}"
MAXLEN="${MAXLEN:-32768}"; MAXTOK="${MAXTOK:-24000}"; CONC="${CONC:-32}"
SUBSETS="${SUBSETS:-text}"
mkdir -p "$OUT" "$S/logs/medxpertqa_ckpt"

for spec in "$@"; do
  tag="${spec%%=*}"; model="${spec#*=}"
  if [ -f "$OUT/medxpertqa_text_${tag}.json" ] && [ "$SUBSETS" = text ]; then echo "skip $tag (done)"; continue; fi
  echo "=== $(date -Is) serving $tag ($model) on GPUs $GPUS"
  CUDA_VISIBLE_DEVICES="$GPUS" nohup vllm serve "$model" --served-model-name "$tag" \
      --host 0.0.0.0 --port "$PORT" --trust-remote-code --tensor-parallel-size "$TP" \
      --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len "$MAXLEN" \
      --max-num-seqs 64 --reasoning-parser qwen3 --mm-encoder-attn-backend TORCH_SDPA \
      --limit-mm-per-prompt '{"image":0}' > "$S/logs/medxpertqa_ckpt/serve_${tag}.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 240); do
    curl -sf -m 5 "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "FATAL: vllm died for $tag"; tail -30 "$S/logs/medxpertqa_ckpt/serve_${tag}.log"; break; }
    sleep 10
  done
  if curl -sf -m 5 "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    for subset in $SUBSETS; do
      echo "=== $(date -Is) medxpertqa $subset for $tag"
      python3 scripts/self_evolving/eval/medxpertqa_eval.py --subset "$subset" \
          --base_url "http://localhost:$PORT/v1" --model "$tag" --max_tokens "$MAXTOK" \
          --concurrency "$CONC" --temperature 0 --dump_responses \
          --out "$OUT/medxpertqa_${subset}_${tag}.json" 2>&1 | tail -5
    done
  fi
  echo "=== $(date -Is) stopping server for $tag"
  kill $SPID 2>/dev/null; sleep 5; pkill -9 -f "vllm serve $model --served-model-name $tag" 2>/dev/null
  sleep 15
done
echo "=== $(date -Is) ALL DONE"
