#!/bin/bash
# GEPA + ACE (training-free self-evolving baselines) on a 4-GPU pod: serve the two frozen
# backbones, learn a context per (method, task, model), freeze it, evaluate it once.
#   hbpro : learns ONLY from generator-written tasks (HealthBench Pro description); val never seen
#   mimic : learns on the MIMIC-IV rare-diagnosis train split; evaluated on the test split
# Everything is resumable (state + answer caches on the NFS), so rerun after a reclaim.
#   TASKS="hbpro mimic" METHODS="gepa ace" MODELS="qwen35_9b qwen36_27b" bash run_context_baselines.sh
set -u
S=/scratch/sheng/self_evolving; REPO=${REPO:-$S/verl_specgap}; STAGE1=${STAGE1:-$S/verl}
cd "$REPO" || exit 1
export HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
IFS= read -r TRAPI_API_KEY < "$S/.trapi_key"; export TRAPI_API_KEY
OUT=$S/paper_refresh/context_baselines; LOGD=$S/logs/context_baselines; mkdir -p "$OUT" "$LOGD"
PYS=scripts/self_evolving/eval/context_evolving_baselines.py
TASKS="${TASKS:-hbpro mimic}"; METHODS="${METHODS:-gepa ace}"; MODELS="${MODELS:-qwen35_9b qwen36_27b}"
BUDGET="${BUDGET:-2000}"; TRAIN_SIZE="${TRAIN_SIZE:-300}"; EPOCHS="${EPOCHS:-2}"; LIMIT="${LIMIT:-0}"

serve () { # tag model gpus tp dp port
  curl -sf -m 5 "http://localhost:$6/v1/models" >/dev/null 2>&1 && return 0
  CUDA_VISIBLE_DEVICES="$3" nohup vllm serve "$2" --served-model-name "$1" --host 0.0.0.0 --port "$6" --trust-remote-code \
    --tensor-parallel-size "$4" --data-parallel-size "$5" --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --max-num-seqs 64 --reasoning-parser qwen3 --mm-encoder-attn-backend TORCH_SDPA \
    > "$LOGD/serve_$1.log" 2>&1 &
  for i in $(seq 1 240); do curl -sf -m 5 "http://localhost:$6/v1/models" >/dev/null 2>&1 && return 0; sleep 10; done; return 1
}
port_of () { case "$1" in qwen35_9b) echo 8210 ;; qwen36_27b) echo 8211 ;; esac; }
case " $MODELS " in *" qwen35_9b "*) serve qwen35_9b Qwen/Qwen3.5-9B 0,1 1 2 8210 || { echo "FATAL 9B serve"; exit 1; } ;; esac
case " $MODELS " in *" qwen36_27b "*) serve qwen36_27b Qwen/Qwen3.6-27B 2,3 2 1 8211 || { echo "FATAL 27B serve"; exit 1; } ;; esac
echo "=== $(date -u +%FT%TZ) servers up"

job () { # method task model
  local o="$OUT/${1}_${2}_${3}" url="http://localhost:$(port_of "$3")/v1" extra=()
  [ "$2" = mimic ] && extra=(--eval-module "$STAGE1/scripts/self_evolving/eval_sota.py"
                             --reward-file "$STAGE1/verl/utils/reward_score/self_evolving.py")
  if [ ! -s "$o/context.txt" ]; then
    python3 "$PYS" learn --method "$1" --task "$2" --model "$3" --base-url "$url" --out "$o" --budget "$BUDGET" \
      --train-size "$TRAIN_SIZE" --epochs "$EPOCHS" "${extra[@]}" >> "$LOGD/learn_${1}_${2}_${3}.log" 2>&1 \
      || { echo "LEARN FAILED $1 $2 $3"; return 1; }
  fi
  python3 "$PYS" eval --task "$2" --model "$3" --base-url "$url" --context-file "$o/context.txt" --out "$o" \
    --out-prefix "hbpro_ctx_${3}_${1}" --limit "$LIMIT" "${extra[@]}" >> "$LOGD/eval_${1}_${2}_${3}.log" 2>&1 \
    || { echo "EVAL FAILED $1 $2 $3"; return 1; }
  echo "=== $(date -u +%FT%TZ) done $1 $2 $3: $(tr -d '\n ' < "$o/eval_summary.json" | cut -c1-200)"
}
pids=()
for t in $TASKS; do for m in $METHODS; do for mod in $MODELS; do job "$m" "$t" "$mod" & pids+=($!); done; done; done
wait "${pids[@]}"
echo "=== $(date -u +%FT%TZ) CONTEXT BASELINES DONE"
