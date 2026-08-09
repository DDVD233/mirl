#!/usr/bin/env bash
# Revive a spec-gap arm after its POD dies, without ever changing its config.
#
# WHY THIS EXISTS. On 2026-08-08 the container behind port 2336 was recreated out
# from under a live run: PID 1 was 75 seconds old while the host kernel reported 22
# days of uptime. That kills the trainer, the vLLM workers, the generation server and
# the tmux server at once, leaves no traceback and nothing in dmesg, and frees all four
# GPUs. It cost ten minutes only because somebody was watching the dashboard; overnight
# it would have idled four B200s until morning.
#
# A watchdog INSIDE the pod cannot help -- the restart kills it too. So this runs on the
# local machine and reaches in over ssh.
#
# WHAT IT WILL AND WILL NOT DO. It only ever relaunches an arm exactly as it was
# launched, and the trainer's resume_mode=auto then continues from the newest
# checkpoint. It never edits a config, never changes an arm's env, and never restarts a
# HEALTHY run -- a deliberate restart to adopt a code change is a human decision,
# because it forfeits the steps since the last checkpoint and puts a config seam in the
# middle of a curve.
#
# Revival is not free either: save_freq=20, so up to 20 steps are lost. It is still
# strictly better than an idle box, which loses every step it would have run.
#
# Usage (from the repo root on the local machine):
#   nohup bash scripts/self_evolving/train/watchdog_specgap.sh >> <logfile> 2>&1 &
#   DRY=1 bash scripts/self_evolving/train/watchdog_specgap.sh    # detect, never act

HOST=root@point.dd.works
REPO=/scratch/sheng/self_evolving/verl_specgap
LOGDIR=/scratch/sheng/self_evolving/logs_hb9b
POLL_S="${POLL_S:-300}"
DRY="${DRY:-0}"

# port:arm:window. The arm number selects the config inside
# launch_specgap_when_free.sh; EXP_SUFFIX=_long matches the running experiments, which
# is what makes this a resume rather than a new run (a fresh suffix would mint a fresh
# checkpoint dir and start from zero).
BOXES=("2335:3:arm3" "2336:2:arm2")
EXP_SUFFIX=_long

# A pod that dies repeatedly is broken in a way relaunching will not fix, and each
# attempt costs a model load. Stop and leave it for a human.
MAX_REVIVALS="${MAX_REVIVALS:-3}"
# Long enough that a revival in progress (checkpoint load + 4 vLLM engines is ~7 min)
# is never mistaken for a second failure.
COOLDOWN_S="${COOLDOWN_S:-1800}"
# Free space below which saves start failing. Pruning drops only step dirs STRICTLY
# OLDER than the run's own latest_checkpointed_iteration, so every run stays resumable
# and keeps its final weights. Runs with no tracker file are hand-restored, never
# touched.
#
# "Strictly older" is not pedantry, it is the whole correctness argument. verl writes
# latest_checkpointed_iteration.txt only AFTER a save completes, so WHILE step N is
# being written the tracker still names N-20. A rule of "delete anything that is not the
# latest" therefore deletes the checkpoint currently being written -- which is exactly
# what happened on 2026-08-09 at 02:31: this watchdog removed global_step_180 out from
# under a live save and killed the run with "Parent directory .../global_step_180/actor
# does not exist", costing 20 steps. Re-reading the tracker just before the rm does NOT
# help; during a save the stale value is the correct value. Only the ordering test is
# safe: a dir NEWER than the tracker is in-progress or a failed save, never superseded.
DISK_FLOOR_G="${DISK_FLOOR_G:-400}"
# Second, independent guard: never touch a directory still being written to. An
# in-progress save touches its dir continuously, so anything modified recently is
# off-limits regardless of numbering.
PRUNE_MIN_AGE_MIN="${PRUNE_MIN_AGE_MIN:-30}"
# Log a status line every Nth poll even when everything is fine. Without it a healthy
# watchdog is byte-identical to a dead one: both produce nothing. 12 polls x 300s = 1h.
HEARTBEAT_EVERY="${HEARTBEAT_EVERY:-12}"
# The liveness pattern, overridable ONLY so the revival path can be exercised against
# healthy boxes (point it at a process that does not exist and the DEAD branch runs).
# Bracketed by default so it cannot match the shell evaluating it -- an unbracketed
# pattern self-matches and reports a trainer that is not there.
WD_PROC_PAT="${WD_PROC_PAT:-main[_]pp[o]}"
# Above this, a GPU is considered still in use and the box is NOT relaunched: vLLM
# workers outlive the trainer and relaunching into held memory dies on engine init with
# "Free memory on cuda:N < desired". Same threshold the launcher itself uses.
GPU_IDLE_MIB="${GPU_IDLE_MIB:-4000}"

declare -A REVIVALS STRIKES LAST_ACTION
for b in "${BOXES[@]}"; do
    p="${b%%:*}"; REVIVALS[$p]=0; STRIKES[$p]=0; LAST_ACTION[$p]=0
done

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

sshx() {  # sshx <port> <command>
    local port="$1"; shift
    timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -p "$port" "$HOST" "$@" 2>/dev/null
}

# Reclaim space using the keep-the-latest rule. Bounded, self-guarding, and it prints
# every path it removes so the log explains any missing checkpoint later.
prune_checkpoints() {
    local port="$1"
    sshx "$port" 'c=/scratch/sheng/self_evolving/checkpoints
age='"$PRUNE_MIN_AGE_MIN"'
for r in $c/*/*; do
  [ -d "$r" ] || continue
  lat=$(cat "$r/latest_checkpointed_iteration.txt" 2>/dev/null)
  # No tracker: hand-restored, or a run whose save never completed. Not ours to judge.
  [ -n "$lat" ] || continue
  case "$lat" in *[!0-9]*|"") continue;; esac
  for d in "$r"/global_step_*; do
    [ -d "$d" ] || continue
    n=$(basename "$d" | sed "s/global_step_//")
    case "$n" in *[!0-9]*|"") continue;; esac
    # STRICTLY older only. n >= lat means in-progress or failed, never superseded.
    [ "$n" -lt "$lat" ] || continue
    # And never a directory touched recently: an active save writes continuously.
    [ -n "$(find "$d" -maxdepth 0 -mmin +$age 2>/dev/null)" ] || continue
    g=$(du -sBG "$d" 2>/dev/null | cut -f1 | tr -d G)
    [ "${g:-0}" -lt 50 ] && continue
    lat2=$(cat "$r/latest_checkpointed_iteration.txt" 2>/dev/null)
    case "$lat2" in *[!0-9]*|"") continue;; esac
    [ "$n" -lt "$lat2" ] || continue
    rm -rf "$d" && echo "pruned ${g}G $d"
  done
done'
}

# True when no trainer process is left AND no GPU still holds memory. Both halves
# matter: vLLM workers routinely outlive the trainer and holding memory means a
# relaunch would OOM on engine init. The pgrep pattern is bracketed so it cannot match
# the shell running it -- an unbracketed pattern self-matches and reports a live
# trainer that is not there.
box_is_dead() {
    local port="$1" out
    out=$(sshx "$port" 'if pgrep -f "'"$WD_PROC_PAT"'" >/dev/null 2>&1; then echo ALIVE; else
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        if [ "${u:-99999}" -gt '"$GPU_IDLE_MIB"' ]; then echo "HOLDING ${u}"; else echo DEAD; fi; fi')
    [ -z "$out" ] && { echo "UNREACHABLE"; return 1; }
    echo "$out"
    [ "${out%% *}" = "DEAD" ]
}

revive() {
    local port="$1" arm="$2" win="$3"
    if [ "$DRY" = "1" ]; then log "[$port] DRY RUN: would relaunch ARM=$arm"; return; fi
    sshx "$port" "rm -f /dev/shm/vllm* 2>/dev/null
        tmux has-session -t hb 2>/dev/null || tmux new-session -d -s hb -n idle 'sleep infinity'
        tmux kill-window -t hb:$win 2>/dev/null
        tmux new-window -t hb -n $win \"cd $REPO && ARM=$arm EXP_SUFFIX=$EXP_SUFFIX \
            bash scripts/self_evolving/train/launch_specgap_when_free.sh \
            2>&1 | tee -a $LOGDIR/specgap_arm${arm}_launch.log\"
        echo relaunched"
    REVIVALS[$port]=$(( ${REVIVALS[$port]} + 1 ))
    LAST_ACTION[$port]=$SECONDS
    log "[$port] RELAUNCHED ARM=$arm (revival ${REVIVALS[$port]}/$MAX_REVIVALS); resume_mode=auto continues from the newest checkpoint"
    sshx "$port" "echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog relaunched arm$arm after pod death\" >> $LOGDIR/watchdog.log"
}

log "watchdog up: boxes=${BOXES[*]} poll=${POLL_S}s dry=$DRY max_revivals=$MAX_REVIVALS disk_floor=${DISK_FLOOR_G}G"

tick=0
while :; do
    tick=$(( tick + 1 ))
    hb=0
    { [ "$DRY" = "1" ] || [ $(( tick % HEARTBEAT_EVERY )) -eq 1 ]; } && hb=1
    hb_line=""

    for b in "${BOXES[@]}"; do
        port="${b%%:*}"; rest="${b#*:}"; arm="${rest%%:*}"; win="${rest##*:}"

        state=$(box_is_dead "$port") && dead=1 || dead=0
        hb_line="$hb_line ${port}=${state%% *}"
        if [ "$state" = "UNREACHABLE" ]; then
            # A pod mid-recreation refuses ssh for a while. Not evidence of anything
            # yet, and certainly not grounds to act.
            log "[$port] unreachable"
            STRIKES[$port]=0
            continue
        fi

        if [ "$dead" = "1" ]; then
            STRIKES[$port]=$(( ${STRIKES[$port]} + 1 ))
            log "[$port] $state (strike ${STRIKES[$port]}/2)"
            # Two consecutive strikes: one poll can land in the gap between a trainer
            # exiting and its relaunch, or during a human's own maintenance.
            if [ "${STRIKES[$port]}" -ge 2 ]; then
                since=$(( SECONDS - ${LAST_ACTION[$port]} ))
                if [ "${REVIVALS[$port]}" -ge "$MAX_REVIVALS" ]; then
                    log "[$port] NOT reviving: ${REVIVALS[$port]} revivals already; needs a human"
                elif [ "${LAST_ACTION[$port]}" -ne 0 ] && [ "$since" -lt "$COOLDOWN_S" ]; then
                    log "[$port] within cooldown (${since}s < ${COOLDOWN_S}s), waiting"
                else
                    free=$(sshx "$port" "df -BG /scratch | tail -1 | awk '{print \$4}' | tr -d G")
                    if [ -n "$free" ] && [ "${free:-9999}" -lt "$DISK_FLOOR_G" ]; then
                        log "[$port] only ${free}G free -- pruning non-latest checkpoints before relaunch"
                        prune_checkpoints "$port" | while read -r l; do log "[$port] $l"; done
                    fi
                    revive "$port" "$arm" "$win"
                fi
                STRIKES[$port]=0
            fi
        else
            [ "${STRIKES[$port]}" -ne 0 ] && log "[$port] recovered ($state)"
            STRIKES[$port]=0
        fi
    done

    # Disk is shared with at least eight other tenants and has moved 400G/h. A run that
    # dies on ENOSPC mid-save is the worst case, so prune before that rather than after.
    free=$(sshx 2336 "df -BG /scratch | tail -1 | awk '{print \$4}' | tr -d G")
    [ "$hb" = "1" ] && log "heartbeat:$hb_line scratch_free=${free:-?}G"
    if [ -n "$free" ] && [ "${free:-9999}" -lt "$DISK_FLOOR_G" ]; then
        log "DISK ${free}G free < ${DISK_FLOOR_G}G floor: pruning non-latest checkpoints"
        if [ "$DRY" = "1" ]; then log "DRY RUN: would prune"; else
            prune_checkpoints 2336 | while read -r l; do log "$l"; done
            log "DISK after prune: $(sshx 2336 "df -BG /scratch | tail -1 | awk '{print \$4}'") free"
        fi
    fi

    sleep "$POLL_S"
done
