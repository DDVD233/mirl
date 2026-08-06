#!/usr/bin/env bash
# v11 = the DECISION POINT for the GPT-5.6 warm start.
#
# Starts RL from the GPT-5.6 SFT checkpoint with the redesigned retrieval loop and
# the REBUILT knowledge base (StatPearls / DailyMed / MedlinePlus / ICD-10 added,
# general-Wikipedia and bare-title rows down-weighted or excluded — see
# scripts/self_evolving/kb/retrieval.py). `val_before_train=True` means the FIRST
# number this prints is the one that matters; it is a warm-start evaluation, not a
# commitment to a long run.
#
# READ THE RIGHT COLUMN. The official HealthBench-Pro metric is
# `acc_len_adj_signed` — length-adjusted AND signed (negative criteria subtract).
# Do NOT read `acc_raw`; it runs ~0.18 higher and makes a bad run look fine.
# Baselines, all judge=gpt-chat-latest_2026-05-28, from rejudge_results.jsonl:
#
#   acc_len_adj_signed   acc_raw     config
#   ------------------   -------     ------
#        0.3752          0.5137      stock, no tool
#        0.2922          0.3907      stock + raw retrieval (the broken v7 loop)
#        0.3827          0.5643      stock + fixed retrieval loop  <-- THE BAR
#        0.399             --        v9 in-loop step 0 (best observed start)
#
# So: step-0 must clear ~0.383 to justify the warm start at all, and ~0.399 to
# beat the best start we have seen. Below 0.383 the warm start is not paying for
# itself — kill it and report rather than spending GPU-days on RL.
# (The previous selective-SFT warm start was reported as 0.488 *raw*; there is no
# published signed number for it. Do not compare it to the column above.)
#
# Set VAL_ONLY=1 to stop after the step-0 validation instead of training on.
set -uo pipefail
cd /scratch/sheng/self_evolving/verl_healthbench

# Resolve the merged checkpoint. verl names the directory after the final step
# (global_step_28 for a 901-trace epoch), so there is no fixed "final" path to
# hardcode — auto-pick the highest step that actually has a merged model.
SFT_BASE="${SFT_BASE:-/scratch/sheng/self_evolving/checkpoints/self_evolving_medical/gpt56_sft_qwen36_27b}"
if [[ -z "${SFT_CKPT:-}" ]]; then
  for d in $(ls -d "$SFT_BASE"/global_step_* 2>/dev/null | sort -t_ -k3 -n -r); do
    if [[ -f "$d/hf_merged/config.json" ]]; then
      SFT_CKPT="$d/hf_merged"
      break
    fi
  done
fi
if [[ -z "${SFT_CKPT:-}" || ! -f "$SFT_CKPT/config.json" ]]; then
  echo "ERROR: no merged SFT checkpoint under $SFT_BASE/global_step_*/hf_merged" >&2
  echo "       (run scripts/self_evolving/train/run_gpt56_pipeline.sh first)" >&2
  exit 1
fi
echo "using SFT checkpoint: $SFT_CKPT"

export ACTOR_MODEL_PATH="$SFT_CKPT"
export EXP="${EXP:-healthbench_rubric_qwen36_27b_v11_gpt56sft}"
export RETRIEVAL_URL="${RETRIEVAL_URL:-http://localhost:8006/retrieve}"

# Same judge as v9/the rejudges, so step-0 is directly comparable to 0.383/0.399.
# v10 used gpt-5.4 instead and scored 0.364-0.370 on the same signed metric: that
# is a JUDGE difference, not a regression, and mixing the two invalidates the
# comparison. Keep this pinned unless you re-baseline everything.
export VAL_JUDGE_MODEL="${VAL_JUDGE_MODEL:-gpt-chat-latest_2026-05-28}"
export REWARD_JUDGE_CONCURRENCY="${REWARD_JUDGE_CONCURRENCY:-6}"

# Keep the v10 OOM fix: retrieval rollouts overflowed the 24576-token micro-batch.
export PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-12288}"

# Truncate the sampling tail. Rollouts were drawn with top_p=1.0 and no top_k over a
# 248k-token vocabulary, i.e. every rare token reachable at every step, while the
# model's own generation_config.json asks for top_k=20 / top_p=0.95. Measured cost:
# 47% of training rollouts contained CJK/Cyrillic/Hangul tokens mid-sentence and
# invented drug names ("Ikavaline", "Biojax"), with corruption rising the deeper into
# the answer it got (93 -> 383 foreign chars per 100k, first to last decile) — while
# greedy validation of the same checkpoint looked clean. Those rollouts averaged
# -0.013 vs +0.126 for clean ones, so most of the training signal was being spent
# teaching the model about its own sampling noise.
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"

EXTRA=()
if [[ "${VAL_ONLY:-0}" == "1" ]]; then
  EXTRA+=(trainer.val_only=True)
fi

# Dump TRAINING rollouts per step, not just validation. The graded prompt for a
# retrieval rollout is the phase-2 clean prompt with the retrieved passages inline,
# so one dump carries all four things an error post-mortem needs: the task, the
# passages actually returned, the model's thinking, and its answer — alongside the
# per-criterion reward info. Without this, only val generations are inspectable and
# training-side failures can only be guessed at from scalar curves.
ROLLOUT_DIR="${ROLLOUT_DIR:-/scratch/sheng/self_evolving/logs_healthbench_rubric/rollouts/$EXP}"
mkdir -p "$ROLLOUT_DIR"
EXTRA+=(+trainer.rollout_data_dir="$ROLLOUT_DIR")
EXTRA+=(actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P")
EXTRA+=(actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K")

LOG="${LOG:-/scratch/sheng/self_evolving/logs_healthbench_rubric/v11_gpt56sft_train.log}"
mkdir -p "$(dirname "$LOG")"

bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric_v7_retrieval.sh \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"
