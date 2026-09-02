#!/usr/bin/env bash
# Stage 1 of 2 for the MedXpertQA line: SFT the Qwen3.5-9B on teacher reasoning
# traces, then hand the checkpoint to the spec-gap adversary RL (stage 2, ARM=22).
#
# WHY A STAGE 1 AT ALL. The RL-only arm (ARM=21, wandb
# medxpert9b_specgap_ship_retrieval_websearch_0830_1047) saturated: overall val
# peaked 0.4265 at step 70 and decayed to 0.407 by 170, while think_chars DOUBLED
# (text 7.5k -> 14k chars) and think_closed stayed 1.000. The policy learned to
# reason LONGER and hit a capability ceiling -- RL can only elicit what the model
# already has. That is the same wall the MIMIC-Rare line hit, where SFT-then-RL
# cleared it, and this mirrors that recipe.
#
# ONE EPOCH, deliberately: the point is to install the reasoning FORMAT and a floor
# of clinical knowledge, then let RL do the shaping. More epochs memorise the teacher
# and shrink the exploration RL needs.
#
# The traces are built from PUBLIC sources only (MedQA + CLIMB), never MedXpertQA --
# that benchmark ships no train split, so its test set IS our held-out validation.
# See build_medxpert_sft_source.py / make_medxpert_traces.py.
set -xeuo pipefail

S=/scratch/sheng/self_evolving

export REPO="${REPO:-$S/verl_specgap}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
export EXP="${EXP:-medxpert9b_sft_distill}"
export DISTILL_FILE="${DISTILL_FILE:-$S/medxpert_sft/traces_train.jsonl}"
export SFT_VAL_FILE="${SFT_VAL_FILE:-$S/medxpert_sft/traces_val.jsonl}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
# -1 = derive the step count from the epochs. The shared distill script defaults
# TOTAL_STEPS to 500, and that OVERRIDES total_epochs: with 9,675 traces at batch 32
# (~302 steps/epoch) it would quietly run 1.65 epochs, not the one asked for. More
# epochs memorise the teacher and shrink the exploration the RL stage needs, which is
# the whole reason stage 1 is a single pass.
export TOTAL_STEPS="${TOTAL_STEPS:--1}"
export LR="${LR:-1e-6}"
# Every ~15% of an epoch, so stage 2 can start from a partial-SFT checkpoint if the
# full epoch turns out to overshoot (the MIMIC line used exactly that "half distill"
# option and it is cheap insurance).
export SAVE_FREQ="${SAVE_FREQ:-40}"

# The shared script's judge default is point.dd.works:18184 -- server5, RETIRED since
# 2026-07-31. Every val judge call failed with ClientConnectorError after 4 retries,
# so validation burned ~4.5 min a pass and produced nothing. The live 27B is on 2333.
# It is also 2337's reward judge, which is why TEST_FREQ is raised rather than left at
# 5: 197 generate+judge prompts every 5 steps over ~302 steps is ~60 validations and
# roughly five hours, most of a second run's worth of GPU time, for far more curve
# resolution than a single-epoch SFT needs.
export TEACHER_BASE="${TEACHER_BASE:-http://point.dd.works:18188/v1}"
export TEST_FREQ="${TEST_FREQ:-25}"

# Qwen3.5-9B quirks, same as run_qwen35_9b_sft_distill.sh: head_dim=256 breaks
# FlashAttention's varlen kernel, so sdpa + use_remove_padding=False.
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
export ATTN_SDPA="${ATTN_SDPA:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"

# Traces run long (teacher reasoning is 260-580 words) and the CLIMB half carries
# image tokens in the prompt, which count toward max_prompt_length.
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-16384}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6144}"

exec bash "$(dirname "$0")/run_qwen36_27b_sft_distill.sh" "$@"
