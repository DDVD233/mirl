#!/usr/bin/env bash
# Sequential scaling-law ablation queue for SERVER 4 (4x B200). Companion lines
# to the j75o3rrt main line (27B SFT -> data-gen RL) for the compute-vs-mimic-rare
# figure. Every stage: full val (test.jsonl, val_max_samples=-1) every 5 steps,
# judge = Qwen3.6-27B @ server 5 (must be up), wandb project self_evolving_medical.
#
#   Stage 1  27B "normal RL" control: GRPO on the REAL mimiciv_rare train set
#            (no gen server), initialized from the SAME SFT checkpoint as the
#            main line, use_kl_loss=False to match it. 500 steps.
#   Stage 2  Qwen3.5-9B SFT distill (same teacher traces, lr 1e-6, bs 32).
#   Stage 2b merge the step-90 SFT checkpoint to a HF dir (same data budget as
#            the 27B line's "half distill" init).
#   Stage 3  Qwen3.5-9B data-gen RL from its SFT init (gen server started here
#            on :8006, teacher/judge = 27B @ server 5). 500 steps.
#
# Failures stop the queue (set -e) so a broken stage can't waste GPU-days.
set -xeuo pipefail

REPO="${REPO:-/scratch/sheng/self_evolving/verl_healthbench}"
LOGROOT=/scratch/sheng/self_evolving/logs_ablation_s4
SFT27=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/mimiciv_rare_qwen36_27b_sft_distill/global_step_90/actor/huggingface
CKPT9B_SFT=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/mimiciv_rare_qwen35_9b_sft_distill
mkdir -p "$LOGROOT"
cd "$REPO"

echo "================ STAGE 1: 27B train-set RL from SFT ($(date)) ================"
MODEL_PATH="$SFT27" USE_KL_LOSS=False REPO="$REPO" \
  EXP=mimiciv_rare_qwen36_27b_trainset_rl_from_sft \
  bash scripts/self_evolving/train/run_qwen36_27b_baseline.sh \
  2>&1 | tee -a "$LOGROOT/stage1_27b_trainset_rl.log"

echo "================ STAGE 2: 9B SFT distill ($(date)) ================"
REPO="$REPO" bash scripts/self_evolving/train/run_qwen35_9b_sft_distill.sh \
  2>&1 | tee -a "$LOGROOT/stage2_9b_sft.log"

echo "================ STAGE 2b: merge 9B SFT step-90 -> HF ($(date)) ================"
/usr/local/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CKPT9B_SFT/global_step_90/actor" \
  --target_dir "$CKPT9B_SFT/global_step_90/actor/huggingface" \
  2>&1 | tee -a "$LOGROOT/stage2b_merge.log"

echo "================ STAGE 3: 9B data-gen RL from SFT ($(date)) ================"
# Gen server (mimic RL mode, :8006) runs alongside stage 3 only.
GEN_LOG="$LOGROOT/stage3_gen_server.log"
LOG_DIR="$LOGROOT/stage3_gen_logs" REPO="$REPO" \
  nohup bash scripts/self_evolving/serve/start_gen_server_mimic_s1.sh \
  > "$GEN_LOG" 2>&1 &
GEN_PID=$!
trap 'kill $GEN_PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  curl -sf localhost:8006/healthz > /dev/null && break
  sleep 5
done
curl -sf localhost:8006/healthz > /dev/null || { echo "gen server failed to start"; exit 1; }

ACTOR_MODEL_PATH="$CKPT9B_SFT/global_step_90/actor/huggingface" REPO="$REPO" \
  bash scripts/self_evolving/train/run_qwen35_9b_evolve_from_sft.sh \
  2>&1 | tee -a "$LOGROOT/stage3_9b_rl.log"

echo "================ QUEUE DONE ($(date)) ================"
