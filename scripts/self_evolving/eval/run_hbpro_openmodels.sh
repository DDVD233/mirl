#!/usr/bin/env bash
# Reference rows for the stage-2 paper: off-the-shelf models (and our merged SER checkpoint
# under the same protocol) on HealthBench Professional with the official simple-evals
# pipeline, graded by gpt-chat-latest (the in-loop validation grader). No tools, greedy,
# bare conversation, 8192-token answer budget (the in-loop response cap).
#
# One lane = one pair of GPUs: serve with vLLM, evaluate, tear down, next model.
#   LANE_GPUS=0,1 PORT=8210 bash run_hbpro_openmodels.sh hb27b_ser_step200=/scratch/.../hf_merged/hb27b_ser_step200 Qwen/Qwen3.6-27B
#   LANE_GPUS=2,3 PORT=8211 bash run_hbpro_openmodels.sh google/gemma-4-31B-it Qwen/Qwen3.5-9B
# A spec is either <hub id> or <tag>=<path>. LIMIT=20 for a smoke run.
set -u
S=/scratch/sheng/self_evolving
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=0 NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
LANE_GPUS="${LANE_GPUS:-0,1}"; PORT="${PORT:-8210}"; TP="${TP:-2}"
LIMIT="${LIMIT:-0}"; CONC="${CONC:-8}"; MAXTOK="${MAXTOK:-8192}"; MAXLEN="${MAXLEN:-32768}"
OUT="${OUT:-$S/logs/hbpro_eval_stage2}"; LOGD=$S/logs/hbpro_eval_stage2/serve
KEY=$(cat $S/.trapi_key)
mkdir -p "$OUT" "$LOGD"
cd $S/verl_specgap
MMFLAG=""
vllm serve --help 2>/dev/null | grep -q -- "--mm-encoder-attn-backend" && MMFLAG="--mm-encoder-attn-backend TORCH_SDPA"

for spec in "$@"; do
  if [[ "$spec" == *=* ]]; then tag="${spec%%=*}"; model="${spec#*=}"; else tag="$(echo "$spec" | tr '/' '_')"; model="$spec"; fi
  if ls "$OUT"/result__hbpro-"$tag"*.json >/dev/null 2>&1 && [ "$LIMIT" = 0 ]; then echo "skip $tag (done)"; continue; fi
  RP=""; case "$model" in *Qwen*|*qwen*|*hb27b*|*hb9b*) RP="--reasoning-parser qwen3";; esac
  echo "=== $(date -Is) serving $tag ($model) on GPUs $LANE_GPUS port $PORT"
  CUDA_VISIBLE_DEVICES="$LANE_GPUS" nohup vllm serve "$model" --served-model-name "$tag" \
      --host 0.0.0.0 --port "$PORT" --trust-remote-code --tensor-parallel-size "$TP" \
      --dtype bfloat16 --gpu-memory-utilization "${GPU_UTIL:-0.90}" --max-model-len "$MAXLEN" --max-num-seqs 64 \
      $RP $MMFLAG > "$LOGD/${tag}.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 360); do
    curl -sf -m 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { echo "FATAL: vllm died for $tag"; tail -20 "$LOGD/${tag}.log"; break; }
    sleep 10
  done
  if curl -sf -m 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    LIM=(); [ "$LIMIT" != 0 ] && LIM=(--limit "$LIMIT")
    echo "=== $(date -Is) evaluating $tag"
    /usr/local/bin/python scripts/self_evolving/eval/healthbench_professional_eval.py \
        --model-provider vllm --model-base "http://localhost:$PORT/v1" --model-name "$tag" \
        --max-tokens "$MAXTOK" --temperature 0 \
        --grader trapi --grader-base http://point.dd.works:18890/v1 --grader-key "$KEY" \
        --grader-model gpt-chat-latest_2026-05-28 --grader-effort "" \
        --concurrency "$CONC" --no-wandb --output-dir "$OUT" "${LIM[@]}" 2>&1 | tail -12
  fi
  echo "=== $(date -Is) stopping $tag"
  kill $SPID 2>/dev/null; sleep 5; pkill -9 -f "vllm serve $model --served-model-name $tag" 2>/dev/null; sleep 15
done
echo "=== $(date -Is) LANE DONE"
