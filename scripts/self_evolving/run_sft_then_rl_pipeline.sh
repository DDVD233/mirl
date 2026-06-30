#!/usr/bin/env bash
# Unattended SFT -> RL pipeline driver (run in a tmux window on server 1).
#
#   1. SFT on the HALF distillation set (distill_sft_train_half.jsonl), lr 1e-6,
#      1 epoch over half the questions (~0.5 epoch of the full set), offload OFF.
#   2. locate the latest SFT actor/huggingface checkpoint.
#   3. free the GPUs, start the self-evolving generation server on :8006.
#   4. launch RL self-evolving with REWARD EVOLUTION ON, initialised from the SFT
#      checkpoint. Judge / teacher / gen-server LLM throughout = server 5
#      (point.dd.works:18184).
#
# Output streams to the tmux pane AND /scratch/.../logs_pipeline.log.
set -uo pipefail
REPO=/root/mirl_evolve
DATA_DIR=/scratch/sheng/self_evolving/mimiciv_rare
LOG=/scratch/sheng/self_evolving/logs_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "================ PIPELINE START $(date) ================"
cd "$REPO"

# ---------------------------------------------------------------- Stage 1: SFT
# Eval DISABLED (TEST_FREQ huge, val_before_train off): with FSDP offload off the
# training run fills GPU memory, so waking the validation vLLM engine OOMs (this
# crashed the first attempt at step 135). Resumes from the latest checkpoint
# (resume_mode=auto) if one exists.
echo "[pipeline] === SFT start $(date) ==="
EXP=mimiciv_rare_qwen36_27b_sft_distill \
  LR=1e-6 TOTAL_EPOCHS=1 TOTAL_STEPS=320 SAVE_FREQ=90 TEST_FREQ=99999 \
  VAL_BEFORE_TRAIN=False \
  DISTILL_FILE="$DATA_DIR/distill_sft_train_half.jsonl" \
  bash scripts/self_evolving/train/run_qwen36_27b_sft_distill.sh
echo "[pipeline] === SFT finished rc=$? $(date) ==="

# ----------------------------------- locate latest checkpoint + merge -> HF
# verl only writes HF safetensors at the FINAL checkpoint; intermediate saves are
# FSDP shards + config only. Merge the latest shards -> HF explicitly so RL has a
# loadable model regardless of how SFT ended.
CKROOT=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/mimiciv_rare_qwen36_27b_sft_distill
ACTOR_DIR=$(ls -d "$CKROOT"/global_step_*/actor 2>/dev/null | sort -t_ -k3 -n | tail -1)
echo "[pipeline] latest SFT actor dir = $ACTOR_DIR"
if [ -z "$ACTOR_DIR" ] || ! ls "$ACTOR_DIR"/model_world_size_*.pt >/dev/null 2>&1; then
  echo "[pipeline] FATAL: no SFT FSDP shards found; NOT starting RL."; exit 1
fi
SFT_CKPT="$ACTOR_DIR/huggingface"
if [ ! -f "$SFT_CKPT/model.safetensors" ] && ! ls "$SFT_CKPT"/model-*.safetensors >/dev/null 2>&1; then
  echo "[pipeline] merging FSDP shards -> HF safetensors at $SFT_CKPT"
  /usr/local/bin/python -m verl.model_merger merge --backend fsdp \
    --local_dir "$ACTOR_DIR" --target_dir "$SFT_CKPT"
fi
if [ ! -f "$SFT_CKPT/config.json" ] || { [ ! -f "$SFT_CKPT/model.safetensors" ] && ! ls "$SFT_CKPT"/model-*.safetensors >/dev/null 2>&1; }; then
  echo "[pipeline] FATAL: SFT HF checkpoint incomplete after merge; NOT starting RL."; exit 1
fi
echo "[pipeline] SFT checkpoint ready = $SFT_CKPT"

# ------------------------------------------------------------------ free GPUs
echo "[pipeline] freeing GPUs after SFT"
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f main_sft_evolving 2>/dev/null || true
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 8
echo "[pipeline] GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

# -------------------------------------------------------- Stage 2: gen server
# Reuse an already-healthy gen server on :8006 (the prior experiment left one
# running, configured for mimiciv_rare + server5); only start a fresh one if none.
if curl -s -m 3 http://localhost:8006/healthz 2>/dev/null | grep -q '"ok":true'; then
  echo "[pipeline] gen server already healthy on :8006 — reusing it"
else
  echo "[pipeline] starting gen server on :8006"
  tmux kill-window -t pipe:gensrv 2>/dev/null || true
  tmux new-window -t pipe -n gensrv \
    "bash $REPO/scripts/self_evolving/serve/start_gen_server_mimic_s1.sh 2>&1 | tee /scratch/sheng/self_evolving/logs_gensrv.log"
  echo "[pipeline] waiting for gen server /healthz ..."
  UP=0
  for i in $(seq 1 180); do
    if curl -s -m 3 http://localhost:8006/healthz 2>/dev/null | grep -q '"ok":true'; then UP=1; echo "[pipeline] gen server UP after ~$((i*5))s"; break; fi
    sleep 5
  done
  if [ "$UP" != "1" ]; then echo "[pipeline] FATAL: gen server not up after ~15m; NOT starting RL."; exit 1; fi
fi

# ------------------------------------------- Stage 3: RL (reward-evolution ON)
echo "[pipeline] === RL start $(date)  init=$SFT_CKPT ==="
cd "$REPO"
EXP=mimiciv_rare_qwen36_27b_evolve_from_sft REPO="$REPO" EVOLVE_ENABLE=True \
  bash scripts/self_evolving/train/run_qwen36_27b_evolve_reward.sh \
  actor_rollout_ref.model.path="$SFT_CKPT"
echo "================ PIPELINE END rc=$? $(date) ================"
