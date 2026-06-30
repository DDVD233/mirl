#!/usr/bin/env bash
# End-to-end tail for the baseline sweep. Run AFTER run_baselines.sh (the 7-model
# sweep) is already running in another tmux window. This:
#   1. waits for that first sweep to finish,
#   2. rejudges the finished dumps with gpt-chat-latest (in background, on the gcr
#      proxy — runs concurrently with the big-model GPU generation below),
#   3. generates the 2 huge MoE references (Qwen3.5-397B-A17B-FP8, Kimi-K2.6-NVFP4)
#      — resumable, skips the 7 already done,
#   4. rejudges again (skip-aware: only the 2 new dumps).
set -uo pipefail
BASE=/scratch/sheng/self_evolving/eval_baselines
cd /scratch/sheng/self_evolving/verl
export HF_HOME=/scratch/sheng/self_evolving/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1

echo "[orch $(date +%H:%M:%S)] waiting for first 7-model sweep to finish..."
while ! grep -q "DONE\. ok=" "$BASE/baseline_gen.log" 2>/dev/null; do sleep 60; done
echo "[orch $(date +%H:%M:%S)] first sweep done -> rejudging finished dumps (bg) + starting big-model gen"

# rejudge the already-finished dumps now, concurrently with big-model GPU work
bash scripts/self_evolving/eval/run_rejudge_baselines.sh >"$BASE/rejudge_batch1.log" 2>&1 &
REJ=$!

# generate the 2 big MoE references (GPU); long serve_timeout for the ~500GB Kimi download
/usr/local/bin/python scripts/self_evolving/eval/run_baseline_eval_sweep.py --serve_timeout 10800 \
  2>&1 | tee "$BASE/baseline_gen_big.log"
echo "[orch $(date +%H:%M:%S)] big-model gen done"

wait "$REJ" 2>/dev/null || true
echo "[orch $(date +%H:%M:%S)] batch-1 rejudge done -> rejudging the 2 new dumps"
bash scripts/self_evolving/eval/run_rejudge_baselines.sh
echo "[orch $(date +%H:%M:%S)] ALL BASELINES DONE"
