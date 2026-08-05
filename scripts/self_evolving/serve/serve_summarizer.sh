#!/usr/bin/env bash
# Retrieval summarizer: the FROZEN self-model that compresses retrieved passages
# into a question-conditioned evidence brief for /retrieve.
#
# WHY FROZEN, and why not the training actor: offline SFT-trace generation and RL
# rollouts must see byte-identical briefs. If this tracked the actor's weights the
# brief would change every optimizer step, so the warm start would teach a passage
# distribution that does not exist at train time — the exact invariant
# scripts/self_evolving/kb/retrieval.py exists to protect. Point MODEL at the same
# checkpoint the SFT traces were built from and leave it there for the whole run.
#
# WHY A SEPARATE PROCESS: summarization sits on the GENERATION critical path (every
# search turn waits on it), so co-locating it with the rollout engine makes the two
# fight for SMs exactly when both are busy.
#
# Sized for a merged plan of ~16-24 passages (<=24k chars ~ 6k tokens) + the request
# + a <=700-token brief. No tool-call parser: this endpoint never sees tools.
#
#   PORT=8199 MODEL=Qwen/Qwen3.5-9B bash scripts/self_evolving/serve/serve_summarizer.sh
#
# Single-box fallback (no spare GPU): pin to one device and take a small slice, and
# drop the rollout engine's fraction to match:
#   CUDA_VISIBLE_DEVICES=3 MEM=0.08 bash .../serve_summarizer.sh
#   ... trainer: actor_rollout_ref.rollout.gpu_memory_utilization=0.38
set -xeuo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8199}"
TP="${TP:-1}"
MEM="${MEM:-0.85}"
MAXLEN="${MAXLEN:-16384}"
MAXSEQS="${MAXSEQS:-128}"

S=${S:-/scratch/sheng/self_evolving}
export HF_HOME="${HF_HOME:-$S/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# B200 vllm image: nvcc/CCCL header mismatch aborts kernel compilation without this.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK}"

exec vllm serve "$MODEL" \
    --trust-remote-code \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$MAXSEQS"
