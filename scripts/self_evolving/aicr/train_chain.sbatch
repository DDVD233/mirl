#!/bin/bash
#SBATCH --job-name=hb-train
#SBATCH --partition=b200-batch
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/dvdai_mit/self_evolving/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dvdai@mit.edu
#
# Self-renewing training job for the AICR cluster (24h partition limit).
# See scripts/self_evolving/aicr/README.md for the full setup.
#
# What it does, in order:
#   1. Queues its own successor with --dependency=afterany:<this job>, so the
#      chain lives entirely inside slurm (no login-node babysitter process).
#      Touch $SCRATCH/chain/STOP.<jobname> to end the chain.
#   2. Starts frpc so the node is reachable at ssh -p $FRP_PORT dvdai_mit@point.dd.works.
#   3. Opens the Milvus tunnel to mib (localhost:19531 -> mib:19531, the
#      medvecdb host port) for the /retrieve endpoint.
#   4. Runs $RUN_SCRIPT. The workload MUST be resumable from its checkpoint
#      dir (verl trainer.resume_mode=auto) because the 24h wall clock kills it.
#      With no RUN_SCRIPT the job just holds the allocation (interactive use).
#
# Env knobs (propagated to successors via --export=ALL):
#   RUN_SCRIPT  path to the training launcher (e.g. .../train/run_v9_retrieval_fixed.sh)
#   FRP_PORT    remote ssh port on point.dd.works (default 2335)
#   CHAIN=0     disable self-resubmission (single 24h run)

set -uo pipefail

WORK=/work/mit/ppliang_mit/dvdai
SCRATCH=/scratch/dvdai_mit/self_evolving
mkdir -p "$SCRATCH/chain" "$SCRATCH/logs"

# A STOP file ends the chain: this (successor) job exits immediately without
# resubmitting or running anything.
if [[ -f "$SCRATCH/chain/STOP.$SLURM_JOB_NAME" ]]; then
    echo "STOP file present ($SCRATCH/chain/STOP.$SLURM_JOB_NAME) - chain ended, exiting"
    exit 0
fi

# --- 1. queue the successor first, before anything can crash -----------------
# The first link records its exact submit line (minus any dependency/export
# flags) in a chain state file; every later link replays that. Replaying our
# own sacct SubmitLine instead would ACCUMULATE --dependency/--export flags
# each generation until sacct truncates the line and the chain dies.
SUBFILE=$SCRATCH/chain/SUBMITLINE.$SLURM_JOB_NAME
if [[ ! -f "$SUBFILE" ]]; then
    subline=$(sacct -j "$SLURM_JOB_ID" -o SubmitLine%500 -n 2>/dev/null | head -1 | sed 's/^ *//;s/ *$//')
    subline=$(sed -E 's/--dependency=[^ ]+ //g; s/--export=[^ ]+ //g' <<<"$subline")
    workdir=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'WorkDir=\K.*')
    if [[ "$subline" == sbatch\ * ]]; then
        printf '%s\n%s\n' "${workdir:-$SCRATCH}" "$subline" > "$SUBFILE"
    fi
fi
if [[ "${CHAIN:-1}" == 1 ]]; then
    npend=$(squeue -h -u "$USER" --name="$SLURM_JOB_NAME" -t PD | wc -l)
    if (( npend == 0 )) && [[ -f "$SUBFILE" ]]; then
        { read -r cworkdir; read -r csubline; } < "$SUBFILE"
        ( cd "$cworkdir" && \
          eval "${csubline/sbatch /sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL }" ) \
            && echo "successor queued (afterany:$SLURM_JOB_ID): $csubline"
    elif (( npend == 0 )); then
        sbatch --dependency="afterany:$SLURM_JOB_ID" --export=ALL \
            --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" "$0" \
            && echo "successor queued via fallback (afterany:$SLURM_JOB_ID)"
    else
        echo "successor already pending ($npend), not stacking another"
    fi
else
    echo "chain disabled (CHAIN=${CHAIN:-1})"
fi

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi -L 2>/dev/null || true

# --- 2. frp: expose this node's sshd on point.dd.works:$FRP_PORT -------------
# Default derived PER CHAIN NAME, in a range disjoint from the MSR pods.
# Every chain job used to default to 2335 -- the SAME port the MSR pod
# david-evolving-opp-*-half-node-0 uses. Whenever that pod was preempted, whichever
# AICR job was retrying grabbed its port within seconds and the pod could never get
# it back (the retry loop below reclaims it every 15s), so the pod became
# unreachable and answered as an AICR node. Observed 2026-08-05. Stable per name so
# a chain keeps its port across links; 2400-2449 is clear of 2333-2338.
_p=$(printf '%s' "${SLURM_JOB_NAME:-chain}" | cksum | cut -d" " -f1)
FRP_PORT=${FRP_PORT:-$((2400 + _p % 50))}
FRPC_TOML=$SCRATCH/chain/frpc-$SLURM_JOB_ID.toml
cat > "$FRPC_TOML" <<EOF
serverAddr = "point.dd.works"
serverPort = 7000

[[proxies]]
name = "aicr-$SLURM_JOB_NAME-$SLURM_JOB_ID"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = $FRP_PORT
EOF
# retry loop: survives frps restarts and the brief window where the previous
# chain job still holds the remote port
( while true; do "$HOME/frp/frpc" -c "$FRPC_TOML" >> "$SCRATCH/logs/frpc_$SLURM_JOB_ID.log" 2>&1; sleep 15; done ) &
echo "frpc up: ssh -p $FRP_PORT dvdai_mit@point.dd.works"

# --- 3. Milvus tunnel: localhost:19531 -> mib medvecdb -----------------------
( while true; do
    ssh -N -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes -L 19531:localhost:19531 mib
    sleep 15
  done ) &
echo "milvus tunnel: localhost:19531 -> mib:19531"

# --- 4. workload -------------------------------------------------------------
# On FAILURE the job does NOT exit: it holds the allocation (frpc + tunnels
# stay up) so the node can be debugged over ssh -p $FRP_PORT. Touch
# $SCRATCH/chain/RERUN.<jobname> to re-run the workload inside the same
# allocation; touch the STOP file to end the hold (and the chain).
export AICR=1
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
# Belt-and-braces vs the 30-day mtime purge: every (daily) chain link re-touches
# old files in the background — the biweekly scrontab toucher alone proved
# unreliable (its Aug-1 occurrence never fired) and rsync -a imports stale mtimes.
( bash "$WORK/mirl/scripts/self_evolving/aicr/touch_scratch.sh" > /dev/null 2>&1 & )
if [[ -n "${RUN_SCRIPT:-}" ]]; then
    attempt=0
    while true; do
        attempt=$((attempt+1))
        echo "=== running (attempt $attempt): $RUN_SCRIPT ==="
        bash "$RUN_SCRIPT"
        rc=$?
        echo "RUN_SCRIPT exited rc=$rc at $(date)"
        # Per dvd (2026-08-03): NEVER self-terminate — a clean exit also holds
        # the allocation (a crash once masqueraded as rc=0 and killed the
        # chain). Only a human STOP/RERUN decides.
        if (( rc == 0 )); then
            echo "workload exited CLEANLY - HOLDING allocation (chain does not self-stop)."
        else
            echo "workload FAILED (rc=$rc) - HOLDING allocation for debugging."
        fi
        echo "  debug:  ssh -p ${FRP_PORT} dvdai_mit@point.dd.works   (node $(hostname))"
        echo "  rerun:  touch $SCRATCH/chain/RERUN.$SLURM_JOB_NAME"
        echo "  stop:   touch $SCRATCH/chain/STOP.$SLURM_JOB_NAME  (ends hold + chain)"
        date > "$SCRATCH/chain/FAILED.$SLURM_JOB_NAME.$SLURM_JOB_ID"
        while [[ ! -f "$SCRATCH/chain/RERUN.$SLURM_JOB_NAME" ]]; do
            if [[ -f "$SCRATCH/chain/STOP.$SLURM_JOB_NAME" ]]; then
                echo "STOP file appeared during hold - exiting"
                exit "$rc"
            fi
            sleep 30
        done
        rm -f "$SCRATCH/chain/RERUN.$SLURM_JOB_NAME"
        # Sweep GPU orphans before relaunch: a killed trainer can leave vLLM
        # engine workers holding ~100GB/GPU (cgroup-scoped nvidia-smi sees ONLY
        # this job's GPUs, so this cannot touch co-tenants).
        orphans=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
        if [[ -n "$orphans" ]]; then
            echo "killing GPU orphans: $orphans"
            kill $orphans 2>/dev/null; sleep 10
            kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) 2>/dev/null
        fi
        echo "RERUN requested - relaunching workload"
    done
else
    echo "no RUN_SCRIPT set - holding allocation for interactive use"
    sleep infinity
fi
