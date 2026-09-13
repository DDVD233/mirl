#!/usr/bin/env bash
# HealthBench Professional validation of ONE policy (HF-merged checkpoint or base model)
# under the paper's in-loop validation protocol, with the solver's tools switchable.
#
# This is the retrieval-on/off table: the same weights answer the 525 tasks once with
# the tool loop the arms trained with (clinical KB retrieval + web search through the
# gen server) and once with no tools at all, so the gain that survives without
# retrieval is separated from the gain that comes from using retrieval well. Identical
# to run_hbhard_valonly.sh except for the benchmark: HB-Pro keeps its length term
# (the launcher default, so acc_len_adj_signed is the same metric as Table~tab:main)
# and the val parquet is the HB-Pro one.
#
# Everything else is the arm's own validation path (run_9b_hb_gen.sh in
# trainer.val_only mode): sampled decoding at temperature 1, thinking on, the
# official per-criterion grader template, gpt-chat-latest with three votes. Nothing is
# trained; STEPS=1 and no checkpoint is written. The dump lands in
# $S/logs_hb9b/val_generations/$EXP/0.jsonl and can be regraded under any grader with
# scripts/self_evolving/analysis/regrade_hbpro_dumps.py.
#
# Tools ON needs a summarizer (the frozen Qwen3.5-9B). With the dedicated box gone,
# the launcher spawns one on FROZEN_GPU when SUMM_BASE is local -- so on a 4-GPU pod
# run the policy on N_GPUS=3 and the summarizer on FROZEN_GPU=3. Tools OFF needs no
# summarizer, no serper, no KB: N_GPUS=4.
#
# Usage (on an MSR pod, from $S/verl_specgap):
#   MODEL=$S/checkpoints/hf_merged/hb27b_ser_step200 EXP=hbpro_27b_ser200_notools \
#     RETRIEVAL=0 WEB_SEARCH_TOOL=0 N_GPUS=4 bash scripts/self_evolving/eval/run_hbpro_valonly.sh
#   MODEL=$S/checkpoints/hf_merged/hb9b_noretr_ser_step480 EXP=hbpro_9bD_ser480_tools \
#     RETRIEVAL=1 WEB_SEARCH_TOOL=1 N_GPUS=3 FROZEN_GPU=3 bash scripts/self_evolving/eval/run_hbpro_valonly.sh
set -euo pipefail
S=/scratch/sheng/self_evolving
REPO="${REPO:-$S/verl_specgap}"
MODEL="${MODEL:?set MODEL=<hf dir or hub id>}"
EXP="${EXP:?set EXP=<unique name>}"
VAL_PARQUET="${VAL_PARQUET:-$S/healthbench_pro_val.parquet}"
RETRIEVAL="${RETRIEVAL:-1}"
WEB_SEARCH_TOOL="${WEB_SEARCH_TOOL:-$RETRIEVAL}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.netrc" ]; then
    WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc")
fi
export WANDB_API_KEY="${WANDB_API_KEY:?no WANDB_API_KEY}"

# The serper cache restores from its snapshot and snapshots BACK to the same file, so
# an eval must not point at an arm's snapshot: a private copy seeded from arm16's
# keeps the paid-for queries and never clobbers the training arm's file.
SNAP="${SEARCH_SNAPSHOT:-$S/kb/search_cache_valonly.sqlite}"
if [ "$WEB_SEARCH_TOOL" = 1 ] && [ ! -f "$SNAP" ]; then
    cp "$S/kb/search_cache_arm16.sqlite" "$SNAP"
fi
# The summarizer is served locally (the dedicated box at :18186 is gone); the launcher
# spawns it on FROZEN_GPU when SUMM_BASE is local, and only when RETRIEVAL=1.
SUMM_BASE="${SUMM_BASE:-http://localhost:${SUMM_PORT:-8199}/v1}"
cd "$REPO"
echo "=== $(date -u) hbpro val-only $EXP model=$MODEL retrieval=$RETRIEVAL web=$WEB_SEARCH_TOOL n_gpus=${N_GPUS:-4} ==="
exec env RETRIEVAL="$RETRIEVAL" WEB_SEARCH_TOOL="$WEB_SEARCH_TOOL" WEB_EVIDENCE=0 \
     EVOLVE=0 SPEC_GAP=0 SPEC_GAP_SHIP=0 PROBE=0 PATCH=0 HACK_MEMO=0 \
     VAL_ONLY=1 STEPS=1 N_GPUS="${N_GPUS:-4}" TRAIN_BS="${TRAIN_BS:-4}" \
     ACTOR_MODEL_PATH="$MODEL" VAL_PARQUET="$VAL_PARQUET" \
     HB_JUDGE_VOTES="${HB_JUDGE_VOTES:-3}" \
     SUMM_BASE="$SUMM_BASE" SUMM_FALLBACK_BASE="" FROZEN_GPU="${FROZEN_GPU:-3}" \
     SUMMARY_CONCURRENCY="${SUMMARY_CONCURRENCY:-64}" \
     SEARCH_SNAPSHOT="$SNAP" \
     GEN_PORT="${GEN_PORT:-8061}" \
     EXP="$EXP" REPO="$REPO" WANDB_API_KEY="$WANDB_API_KEY" \
     bash scripts/self_evolving/train/run_9b_hb_gen.sh
