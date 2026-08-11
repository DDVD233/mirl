#!/usr/bin/env bash
# The FROZEN 9B as a dedicated multi-GPU inference node: 4 data-parallel replicas,
# tensor-parallel 1, one OpenAI endpoint in front of them.
#
# WHY A WHOLE BOX. Under RETRIEVAL=1 the retrieval summarizer sits on the GENERATION
# critical path: every search turn of every rollout waits on a brief before the policy
# can continue. At batch 32 x rollout.n 8 x ~1.6 searches that is ~256-800 calls per
# step, arriving in one burst because a training batch generates in parallel. Serving
# that from a single shared GPU means the summarizer, not the trainer, sets the step
# time -- and when it browns out, /retrieve degrades to raw passages, which changes what
# the policy learns to retrieve while every log still says retrieval is on.
#
# WHY DP=4 AND NOT TP=4. A 9B in bf16 is ~18 GB, so it fits on one B200 with room for a
# large KV cache; tensor-parallelism would split a model that does not need splitting and
# pay all-reduce latency per token for it. The workload is many short independent
# requests, i.e. throughput-bound, not latency-bound on a single long generation. Four
# independent replicas therefore give ~4x the tokens/s that TP=4 gives at ~1x, and each
# replica keeps a full-size KV cache instead of a quarter of one -- which is what
# actually caps concurrency here, since a merged plan of ~16-24 passages is a ~6k-token
# prompt. vLLM's DP front-end round-robins across the replicas, so callers see one URL.
#
#   PORT=8188 bash scripts/self_evolving/serve/serve_frozen9b_dp.sh
#
# Port 8188 is deliberate on the MSR pods: /etc/frp/frpc.toml already tunnels local 8188
# to a unique public port on point.dd.works (the "comfyui" proxy), so serving there
# publishes the endpoint with no frp change and no restart of a shared tunnel.
set -xeuo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8188}"
DP="${DP:-4}"
TP="${TP:-1}"
# Per REPLICA, and each replica owns one whole GPU on a box with no other tenant -- so it
# takes nearly all of it. The 0.30-0.45 numbers elsewhere in this project are for a vLLM
# sharing GPUs with FSDP training; nothing shares these.
MEM="${MEM:-0.90}"
MAXLEN="${MAXLEN:-16384}"
# The cap that actually binds. At MEM=0.90 each replica gets ~145 GiB of KV cache, about
# 4.2M tokens -- room for ~256 concurrent requests at FULL 16k context, and far more at the
# ~7k a real summarize call uses (a ~6k passage plan plus a 1k brief). Left at 128 the
# server would idle on half its cache while requests queued, which is the failure this
# whole box exists to remove: a queued brief blocks a search turn, which blocks a rollout.
MAXSEQS="${MAXSEQS:-256}"
# One API server process becomes the bottleneck before four replicas do: it does all the
# tokenization, JSON, and HTTP for every request. Scale it with DP.
ASC="${ASC:-$DP}"

S=${S:-/scratch/sheng/self_evolving}
export HF_HOME="${HF_HOME:-$S/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# B200 vllm image: nvcc/CCCL header mismatch aborts kernel compilation without this.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK}"

# Fail before loading 4 replicas if the box cannot supply them. Without this the run
# starts, three ranks come up, and the fourth dies with a CUDA error 900 lines into a log
# nobody reads -- while the endpoint answers, so callers see silent 3/4 capacity.
have=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
need=$(( DP * TP ))
if [ "$have" -lt "$need" ]; then
    echo "FATAL: DP=$DP x TP=$TP needs $need GPUs, box has $have" >&2; exit 1
fi
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${busy:-0}" -gt "${ALLOW_BUSY_MIB:-4000}" ]; then
    echo "FATAL: ${busy}MiB already held on a GPU (trainer or orphan vLLM still up)." >&2
    echo "       Serving on top of it would OOM one replica. Free the box first." >&2
    exit 1
fi

exec vllm serve "$MODEL" \
    --trust-remote-code \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --data-parallel-size "$DP" \
    --tensor-parallel-size "$TP" \
    --api-server-count "$ASC" \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$MEM" \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$MAXSEQS"
