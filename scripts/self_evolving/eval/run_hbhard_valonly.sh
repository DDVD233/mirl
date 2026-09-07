#!/usr/bin/env bash
# Held-out transfer eval for the stage-2 paper: validate a trained (HF-merged) or base
# policy on the original HealthBench Hard set (1,000 examples) with the SAME in-loop
# validation pipeline the paper uses for HealthBench Professional -- solver tools on
# (clinical KB retrieval + web search through the gen server), official per-criterion
# grader template, gpt-chat-latest grader with three votes -- via run_9b_hb_gen.sh in
# trainer.val_only mode. Nothing is trained; STEPS=1 and no checkpoint is written.
#
# Differences from the HB-Pro validation, both deliberate:
#   HB_VAL_LENGTH_PENALTY_PER_500=0   the original HealthBench has no length term; the
#                                     official score is acc_raw (per-example clipped
#                                     rubric fraction, averaged)
#   EVOLVE/SPEC_GAP/PROBE/PATCH=0     no adversary machinery is needed to validate
#
# Usage (on an MSR pod, from $S/verl_specgap):
#   MODEL=/scratch/sheng/self_evolving/checkpoints/hf_merged/hb27b_ser_step200 \
#   EXP=hb27b_ser200_hbhard N_GPUS=2 bash scripts/self_evolving/eval/run_hbhard_valonly.sh
#   MODEL=Qwen/Qwen3.6-27B EXP=base27b_hbhard N_GPUS=2 RETRIEVAL=1 bash ...
#   MODEL=<9B ckpt> EXP=... RETRIEVAL=0 WEB_SEARCH_TOOL=0 bash ...   # no-retrieval setting
set -euo pipefail
S=/scratch/sheng/self_evolving
REPO="${REPO:-$S/verl_specgap}"
MODEL="${MODEL:?set MODEL=<hf dir or hub id>}"
EXP="${EXP:?set EXP=<unique name>}"
VAL_PARQUET="${VAL_PARQUET:-$S/healthbench_hard_val.parquet}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.netrc" ]; then
    WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc")
fi
export WANDB_API_KEY="${WANDB_API_KEY:?no WANDB_API_KEY}"
cd "$REPO"
echo "=== $(date -u) val-only $EXP model=$MODEL val=$VAL_PARQUET retrieval=${RETRIEVAL:-1} ==="
exec env RETRIEVAL="${RETRIEVAL:-1}" WEB_SEARCH_TOOL="${WEB_SEARCH_TOOL:-1}" WEB_EVIDENCE=0 \
     EVOLVE=0 SPEC_GAP=0 SPEC_GAP_SHIP=0 PROBE=0 PATCH=0 HACK_MEMO=0 \
     VAL_ONLY=1 STEPS=1 N_GPUS="${N_GPUS:-2}" \
     ACTOR_MODEL_PATH="$MODEL" VAL_PARQUET="$VAL_PARQUET" \
     HB_VAL_LENGTH_PENALTY_PER_500=0 HB_JUDGE_VOTES="${HB_JUDGE_VOTES:-3}" \
     SUMM_BASE="${SUMM_BASE:-http://point.dd.works:18186/v1}" SUMM_FALLBACK_BASE="" \
     SUMMARY_CONCURRENCY="${SUMMARY_CONCURRENCY:-64}" \
     SEARCH_SNAPSHOT="${SEARCH_SNAPSHOT:-$S/kb/search_cache_arm16.sqlite}" \
     GEN_PORT="${GEN_PORT:-8061}" \
     EXP="$EXP" REPO="$REPO" WANDB_API_KEY="$WANDB_API_KEY" \
     bash scripts/self_evolving/train/run_9b_hb_gen.sh
