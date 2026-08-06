#!/usr/bin/env bash
# Qwen3.6-27B teacher/self-judge on AICR — replaces the reclaimed server5
# (point.dd.works:18184, pod gone 2026-08-01). Same serving config as
# serve_qwen36_27b_teacher_s5.sh but TP2 on 2 GPUs (per dvd: matches what
# server5 actually used), qwen3 parsers, TORCH_SDPA mm-encoder (B200 ViT fix).
#
# Publishes its endpoint to $S/chain/TEACHER_ENDPOINT — training launchers
# resolve the teacher from that file at startup (it moves every 24h chain
# link). vllm restarts forever on crash; the 24h chain renews the node.
#
# Launch:
#   RUN_SCRIPT=$WORK/mirl/scripts/self_evolving/aicr/run_teacher.sh FRP_PORT=18189 \
#     sbatch --job-name=qwen-teacher --gres=gpu:b200:2 -c 24 --mem=200G \
#     $WORK/mirl/scripts/self_evolving/aicr/train_chain.sbatch
set -uo pipefail

S=/scratch/dvdai_mit/self_evolving
SIF=$S/sif/verl-selfevolving-cu130-vllm0.22.1.sif
MSR=/scratch/sheng/self_evolving
PORT=8188

# Publish at server5's ORIGINAL public address: frp local 8188 -> point.dd.works
# :18184 (open at the firewall, free since server5 was reclaimed). Clients keep
# the historical default URL and the teacher can move nodes (devel/batch/24h
# renewals) with no client reconfiguration — the exact path server5 used.
echo "http://point.dd.works:18184/v1" > "$S/chain/TEACHER_ENDPOINT"
FRPC_T=$S/chain/frpc-teacher-${SLURM_JOB_ID:-manual}.toml
cat > "$FRPC_T" <<FRPEOF
serverAddr = "point.dd.works"
serverPort = 7000

[[proxies]]
name = "aicr-teacher-${SLURM_JOB_ID:-manual}"
type = "tcp"
localIP = "127.0.0.1"
localPort = $PORT
remotePort = 18184
FRPEOF
( while true; do "$HOME/frp/frpc" -c "$FRPC_T" >> "$S/logs/frpc_teacher_${SLURM_JOB_ID:-manual}.log" 2>&1; sleep 15; done ) &
echo "teacher will publish at http://point.dd.works:18184/v1 (frp from $(hostname):$PORT)"

while true; do
    apptainer exec --nv --writable-tmpfs --bind "$S:$MSR" "$SIF" \
        env HF_HOME=$MSR/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
        vllm serve Qwen/Qwen3.6-27B \
            --trust-remote-code \
            --served-model-name Qwen/Qwen3.6-27B \
            --host 0.0.0.0 --port $PORT \
            --tensor-parallel-size "${TEACHER_TP:-2}" \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
            --reasoning-parser qwen3 \
            --mm-encoder-tp-mode data \
            --mm-encoder-attn-backend TORCH_SDPA \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.90 \
            --max-model-len 32768
    echo "vllm exited rc=$? at $(date) - restarting in 15s"
    sleep 15
done
