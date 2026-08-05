#!/usr/bin/env bash
# Box-specific launcher for the 9B retrieval RL run on the preemptible MSR pod
# reachable at point.dd.works:2335 (4x B200 183 GB, hand-managed, no slurm).
#
# GPU PLAN. The summarizer has to live on this box (the pods have no pod-to-pod
# networking, so the idle server3 cannot host it). It shares GPU 3 with the trainer:
#   summarizer  SUMM_MEM=0.14 -> ~26 GB   (9B bf16 weights are ~18.5 GB; the script
#                                          default of 0.08 = 14.6 GB CANNOT load it)
#   rollout     VLLM_GPU_UTIL=0.35 -> ~64 GB per GPU
#   -> GPU 3 carries ~90 GB, GPUs 0-2 ~64 GB, leaving >=93 GB everywhere for FSDP
#      activations. FSDP allocates symmetrically, so rank 3 is the one that would
#      OOM first; that is the number to watch if this ever dies in update_actor.
set -xeuo pipefail

S=/scratch/sheng/self_evolving
cd "$S/verl_healthbench"

export WANDB_API_KEY="$(cat $S/.wandb_key_dvd)"
export EXP="${EXP:-hb9b_retrieval_fast}"

export SUMM_GPU="${SUMM_GPU:-3}"
export SUMM_MEM="${SUMM_MEM:-0.14}"
export MAXSEQS="${MAXSEQS:-64}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.35}"

# Retrieval services live on mib (embeddings :18001, Milvus :19531); verified
# reachable from this pod.
export EMBED_BASE="${EMBED_BASE:-http://mib.media.mit.edu:18001/v1}"
export MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
export GEN_PORT="${GEN_PORT:-8031}"

exec bash scripts/self_evolving/train/run_9b_retrieval_fast.sh "$@"
