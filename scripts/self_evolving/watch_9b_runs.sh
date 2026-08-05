#!/usr/bin/env bash
# Liveness watchdog for the two 9B runs. Emits ONLY on a state change, so a quiet
# night produces no notifications and any line that does appear is actionable.
#
# Why a poller and not just a log grep: the failure that has actually happened here
# is the trainer VANISHING (OOM-killer, preemption, driver reset) — no traceback is
# written, the log simply stops, and silence is indistinguishable from "still
# running". So this checks process liveness and log PROGRESS, not just error text.
#
#   ALIVE->DEAD          the trainer process is gone
#   STALLED              log has not grown in $STALL_S (default 45 min)
#   ORPHANS              GPU memory held with no trainer (the documented pattern:
#                        Ray-spawned vLLM workers outliving their driver)
#   BACK                 recovered after any of the above
#
# Usage: bash scripts/self_evolving/watch_9b_runs.sh    (run from the local machine)
set -uo pipefail

HOST=${HOST:-root@point.dd.works}
POLL_S=${POLL_S:-240}
STALL_S=${STALL_S:-2700}

# name:port:logfile
RUNS=(
  "retrieval:2335:/scratch/sheng/self_evolving/logs_hb9b/retrieval_launch.out"
  "control:2336:/scratch/sheng/self_evolving/logs_hb9b/ctrl_len8192.out"
)

declare -A STATE SIZE LASTGROW
for r in "${RUNS[@]}"; do
  n=${r%%:*}; STATE[$n]=INIT; SIZE[$n]=-1; LASTGROW[$n]=$(date +%s)
done

emit() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

while true; do
  for r in "${RUNS[@]}"; do
    name=${r%%:*}; rest=${r#*:}; port=${rest%%:*}; log=${rest#*:}

    # One short ssh per run. Compound remote commands get truncated on this host, so
    # this stays a single printf of three fields.
    # 'main[_]ppo' not 'main_ppo': the ssh command line ITSELF contains the pattern,
    # so a literal pgrep -f matches its own shell and the count is never 0 — DEAD
    # would then never fire, which is the one thing this watchdog exists to catch.
    # The bracket expression matches the running trainer but not this command's argv.
    out=$(ssh -o ConnectTimeout=15 -o BatchMode=yes -p "$port" "$HOST" \
          "printf '%s %s %s' \"\$(pgrep -cf 'main[_]ppo' 2>/dev/null || echo 0)\" \
                              \"\$(stat -c %s '$log' 2>/dev/null || echo 0)\" \
                              \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)\"" \
          2>/dev/null)
    if [[ -z "$out" ]]; then
      [[ "${STATE[$name]}" != UNREACHABLE ]] && emit "$name UNREACHABLE (pod down or network)"
      STATE[$name]=UNREACHABLE; continue
    fi
    read -r nproc size gpumem <<<"$out"
    now=$(date +%s)

    if [[ "$size" -gt "${SIZE[$name]}" ]]; then LASTGROW[$name]=$now; fi
    SIZE[$name]=$size
    stalled=$(( now - LASTGROW[$name] ))

    new=OK
    if [[ "${nproc:-0}" -lt 1 ]]; then
      new=DEAD
      [[ "${gpumem:-0}" -gt 20000 ]] && new=DEAD_WITH_ORPHANS
    elif [[ "$stalled" -gt "$STALL_S" ]]; then
      new=STALLED
    fi

    if [[ "$new" != "${STATE[$name]}" ]]; then
      case "$new" in
        DEAD)               emit "$name DEAD (no trainer; GPU ${gpumem}MiB)" ;;
        DEAD_WITH_ORPHANS)  emit "$name DEAD + ORPHANS holding ${gpumem}MiB — clear before relaunch" ;;
        STALLED)            emit "$name STALLED ${stalled}s with no log growth (trainer alive)" ;;
        OK) [[ "${STATE[$name]}" == INIT ]] && emit "$name watching (trainer up)" \
                                            || emit "$name BACK (recovered)" ;;
        UNREACHABLE)        emit "$name UNREACHABLE" ;;
      esac
      STATE[$name]=$new
    fi
  done
  sleep "$POLL_S"
done
