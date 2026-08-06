#!/usr/bin/env bash
# v7-SELECTIVE: retrieval-augmented RL from the SELECTIVE warm-start SFT init.
#
# Why: v7_wsfix (warm-start SFT that retrieved on EVERY trace) baked in an
# always-search habit -> val retrieval rate 1.00 on all 525 tasks and step-0 acc
# 0.496, BELOW the 0.559 no-retrieval baseline. The "retrieval is optional" prompt
# could not override the SFT habit. Fix: re-SFT on SELECTIVE traces
# (make_selective_traces.py) that teach the CHOICE -- direct-answer traces for what
# the model already knows + retrieve-only-when-a-direct-answer-fails traces -- so the
# policy retrieves only when it needs an external fact.
#
# Point the base v7 retrieval recipe at the selective-SFT hf_merged checkpoint.
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl_healthbench}
cd "$REPO"

# Selective-SFT init (merged HF). Overridable; pipeline passes the discovered step.
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:?set ACTOR_MODEL_PATH to the selective-SFT hf_merged dir}"
export EXP="${EXP:-healthbench_rubric_qwen36_27b_v7_selective}"
export RETRIEVAL_URL="${RETRIEVAL_URL:-http://localhost:8006/retrieve}"

exec bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric_v7_retrieval.sh "$@"
