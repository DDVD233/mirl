#!/usr/bin/env bash
# HealthBench-Professional RUBRIC run, v7 = RETRIEVAL-AUGMENTED (tool-use) rollouts.
#
# Same self-evolving rubric co-generation as v6, but each rollout is now a
# MANDATORY 2-turn conversation:
#   turn 1 (assistant): reason + call `search_medical_kb`
#   tool  turn         : top-k passages from the medical KB (loss-masked)
#   turn 2 (assistant): reason + final answer  -> graded by the rubric
#
# Motivation: v6 plateaued at val acc ~0.55 because ~2/3 of the 187 hard-zero
# val failures need SPECIFIC clinical knowledge the 27B lacks (drug safety,
# dosing thresholds, contraindications, procedure indications). Self-generated
# RL can't create knowledge the policy doesn't have; retrieval injects it.
# A read-only KB coverage probe (2026-07-23) confirmed medical_knowledge_v2
# returns on-point passages for most of these failures (post-cutoff guideline
# updates and exact ICD codes remain out of reach).
#
# Retrieval path (all reused, no new services):
#   solver --POST /retrieve--> gen server :8006 --_milvus_search--> mib embed
#   :18001 (Qwen3-VL-Embedding-2B) + Milvus :19531 (medical_knowledge_v2).
# The `search_medical_kb` tool (verl/tools/medical_retrieval_tool.py) forwards to
# that endpoint; RetrievalToolAgentLoop enforces the mandatory 2-turn structure
# and ports v4-v6's thinking-budget forcing into the multi-turn path.
#
# Start start_gen_server_rubric_s4.sh (unchanged; now also serves /retrieve)
# BEFORE this. Trainer on SERVER 4 (4 GPUs); self judge (Qwen3.6-27B) on SERVER 5.
set -xeuo pipefail

REPO=${REPO:-/scratch/sheng/self_evolving/verl_healthbench}
cd "$REPO"

# v7 = fresh from the SFT init (the 2-turn input distribution differs from v6's
# single-turn, so this is a clean A/B, not a resume).
export EXP="${EXP:-healthbench_rubric_qwen36_27b_v7}"

# Retrieval wiring shared by the agent loop and the tool.
export RETRIEVAL_URL="${RETRIEVAL_URL:-http://localhost:8006/retrieve}"
TOOL_CONFIG="${TOOL_CONFIG:-scripts/self_evolving/train/config/medical_retrieval_tool.yaml}"

# RetrievalToolAgentLoop structure: up to MAX_SEARCHES retrieval rounds (each with
# a smaller think budget), then a GUARANTEED answer turn with an answer-budget
# reserve so the final answer is never starved (the empty-answer bug that sank the
# first v7 val to 0.18).
#
# SPEED-TUNED (2026-07-23): cold-RL with retrieval on the verbose stock model ran
# ~1hr/step (3 long gen passes/rollout; ~8-9k think chars). Both cost drivers cut:
#   - MAX_SEARCHES 2->1 : one search + one answer = 2 gen passes, not 3.
#   - search think 2048->1024, answer think 5120->3072 : curb stock's verbosity
#     (a search query needs little reasoning; 3072 is ample for a medical answer).
# Reduces tokens generated per rollout ~2x. Re-raise if throughput allows.
export VERL_MAX_SEARCHES="${VERL_MAX_SEARCHES:-1}"
export VERL_SEARCH_THINK_BUDGET="${VERL_SEARCH_THINK_BUDGET:-1024}"
export VERL_ANSWER_RESERVE_TOKENS="${VERL_ANSWER_RESERVE_TOKENS:-2500}"
export VERL_THINK_BUDGET_TOKENS="${VERL_THINK_BUDGET_TOKENS:-3072}"

# Reuse the entire v6 recipe (SFT init, lr 2e-7, think budget 5120, entropy
# 0.0005, zero-var filter, signed floor -0.5, gpt-5.1 val judge, KL off) and add
# the multi-turn + retrieval-agent-loop overrides on top.
exec bash scripts/self_evolving/train/run_qwen36_27b_healthbench_rubric.sh \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4000 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=retrieval_tool_agent \
    "$@"
