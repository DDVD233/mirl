#!/usr/bin/env bash
# 9B phase-1 ablation pipeline (server1, 2x B200): replicate the j75o3rrt main
# line (SFT-distill init -> data-gen RL, generator ON, reward-evolve OFF, KL off)
# with Qwen3.5-9B as base AND as its own generator/judge (served on server5,
# point.dd.works:18184, single B200 TP1).
#
# Deviations from run_ablation_queue_s4.sh stages 2-3, and why:
#   - TOTAL_STEPS=90 for SFT: RL inits from global_step_90 (the 27B main line
#     used its step-90 SFT ckpt = "half distill" data budget). lr warmup_style
#     is constant with no decay, so step-90-of-90 == step-90-of-500, and
#     max_ckpt_to_keep=2 would DELETE step_90 if SFT ran past step 150.
#   - trainer.n_gpus_per_node=2: this pod has 2 GPUs, not 4.
#   - judge/generator model_name = Qwen/Qwen3.5-9B (self-judge, not 27B teacher).
#   - EVOLVE_ENABLE=False explicit (phase 1: generator only, no reward evolve).
#   - val sampling == training sampling (do_sample=True, temp 1.0; rep-penalty 1.1
#     is rollout-level so it already applies to val): avoids train/val distribution
#     mismatch (deviation from j75o3rrt, which validated greedily; user 2026-08-04).
set -xuo pipefail

W=/scratch/sheng/self_evolving/verl_phase1_9b
LOGROOT=/scratch/sheng/self_evolving/logs_9b_phase1
CKPT9B_SFT=/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/mimiciv_rare_qwen35_9b_sft_distill
JUDGE_NAME=Qwen/Qwen3.5-9B
mkdir -p "$LOGROOT"
cd "$W"

echo "================ STAGE 0: wait for 9B teacher/judge on :18184 ($(date)) ================"
for i in $(seq 1 240); do
  curl -sf -m 5 http://point.dd.works:18184/v1/models >/dev/null && break
  sleep 15
done
curl -sf -m 5 http://point.dd.works:18184/v1/models >/dev/null || { echo "FATAL: teacher never came up"; exit 1; }

echo "================ STAGE 1: 9B SFT distill, 90 steps ($(date)) ================"
if [ ! -f "$CKPT9B_SFT/global_step_90/actor/huggingface/model.safetensors.index.json" ] \
   && [ ! -f "$CKPT9B_SFT/global_step_90/actor/huggingface/model.safetensors" ]; then
  REPO="$W" TOTAL_STEPS=90 TOTAL_EPOCHS=1 \
    bash scripts/self_evolving/train/run_qwen35_9b_sft_distill.sh \
    trainer.n_gpus_per_node=2 \
    ++reward.custom_reward_function.reward_kwargs.model_name="$JUDGE_NAME" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    2>&1 | tee -a "$LOGROOT/stage1_9b_sft.log"
  test -d "$CKPT9B_SFT/global_step_90/actor" || { echo "FATAL: no step-90 SFT checkpoint"; exit 1; }

  # GPU cleanup: verl leaves ray workers + vllm engines holding memory.
  ray stop --force || true
  pkill -9 -f main_sft_evolving || true
  sleep 10
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 || true
  sleep 20

  echo "================ STAGE 1b: merge step-90 -> HF ($(date)) ================"
  if [ ! -f "$CKPT9B_SFT/global_step_90/actor/huggingface/model.safetensors.index.json" ]; then
    /usr/local/bin/python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$CKPT9B_SFT/global_step_90/actor" \
      --target_dir "$CKPT9B_SFT/global_step_90/actor/huggingface" \
      2>&1 | tee -a "$LOGROOT/stage1b_merge.log"
  fi
fi
ls "$CKPT9B_SFT/global_step_90/actor/huggingface/" | grep -q safetensors || { echo "FATAL: merged HF ckpt missing weights"; exit 1; }
test -f "$CKPT9B_SFT/global_step_90/actor/huggingface/tokenizer_config.json" || { echo "FATAL: merged ckpt missing tokenizer"; exit 1; }

echo "================ STAGE 2: gen server :8006 ($(date)) ================"
if ! curl -sf localhost:8006/healthz >/dev/null 2>&1; then
  MODEL_NAME="$JUDGE_NAME" LOG_DIR="$LOGROOT/gen_logs" PYTHON_BIN=/usr/local/bin/python \
    nohup bash "$W/scripts/self_evolving/serve/start_gen_server_mimic_s1.sh" \
    > "$LOGROOT/gen_server.log" 2>&1 &
fi
for i in $(seq 1 60); do
  curl -sf localhost:8006/healthz >/dev/null && break
  sleep 5
done
curl -sf localhost:8006/healthz >/dev/null || { echo "FATAL: gen server failed to start"; exit 1; }

echo "================ STAGE 3: 9B phase-1 RL (gen on, evolve off, KL off) ($(date)) ================"
ACTOR_MODEL_PATH="$CKPT9B_SFT/global_step_90/actor/huggingface" \
  REPO="$W" EVOLVE_ENABLE=False \
  EXP=mimiciv_rare_qwen35_9b_evolve_from_sft \
  bash scripts/self_evolving/train/run_qwen35_9b_evolve_from_sft.sh \
  trainer.n_gpus_per_node=2 \
  ++reward.custom_reward_function.reward_kwargs.model_name="$JUDGE_NAME" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  2>&1 | tee -a "$LOGROOT/stage3_9b_rl.log"

echo "================ PIPELINE END ($(date)) ================"
