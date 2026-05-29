#!/usr/bin/env bash
# Non-OPD self-evolving GRPO run with chat/proposer/judge on TRAPI Kimi-K2.6
# (via the local TRAPI proxy). No distillation teacher.
#
# Parameterized by env so different students/hosts share one git-tracked
# script. The sensitive proxy key is NOT stored here — pass it at launch:
#
#   API_KEY="$(cat /root/.trapi_proxy_key)" \
#   ACTOR_MODEL_PATH=Qwen/Qwen3.6-27B \
#   EXPERIMENT_NAME=mimiciv_rare_qwen36_27b_full_kimi \
#   bash scripts/self_evolving/launch_kimi_run.sh
#
# Hyperparameters (clip_ratio_high=0.28, train_batch_size=256,
# ppo_mini_batch_size=128, lr=5e-7, test/save_freq=5) are applied here as
# overrides on top of run_qwen36_27b_full.sh; later positional args win.
set -e

# --- chat / judge: TRAPI Kimi via local proxy ---
export API_BASE="${API_BASE:-http://point.dd.works:18890/v1}"
export CHAT_PROVIDER="${CHAT_PROVIDER:-trapi}"
export MODEL_NAME="${MODEL_NAME:-Kimi-K2.6_2026-04-20}"

# --- student model + experiment ---
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen3.6-27B}"
export DATA_DIR="${DATA_DIR:-/scratch/sheng/self_evolving/mimiciv_rare}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-mimiciv_rare_qwen36_27b_full_kimi}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://mib.media.mit.edu:18001/v1}"
export EMBED_API_KEY="${EMBED_API_KEY:-EMPTY}"
export EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
export GEN_SERVER_URL="${GEN_SERVER_URL:-http://localhost:8004}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-/scratch/sheng/self_evolving/hf_cache}"
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
export REPO_ROOT="${REPO_ROOT:-/scratch/sheng/self_evolving/verl}"
LOG_DIR="${LOG_DIR:-/scratch/sheng/self_evolving/logs}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$LOG_DIR/val_generations/$EXPERIMENT_NAME}"

cd "$REPO_ROOT"
bash scripts/self_evolving/run_qwen36_27b_full.sh \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    data.train_batch_size=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    "$@" 2>&1 | tee "$LOG_DIR/train_${EXPERIMENT_NAME}.log"
