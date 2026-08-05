#!/usr/bin/env bash
# Emit one line describing a training run's state: "<trainers> <logbytes> <gpumib> <DONE|RUNNING>"
#
# Lives on the training box and is called by watch_9b_runs.sh. It exists because
# cramming four command substitutions with nested quotes into a single remote ssh
# string silently loses fields — an empty field shifts the rest and the watchdog then
# misreports a cleanly finished run as a crash.
#
#   bash probe_run_state.sh <logfile>
set -uo pipefail
LOG="${1:?usage: probe_run_state.sh <logfile>}"

# 'main[_]ppo' so the pattern never matches this script's own command line.
# NOTE: `pgrep -c` PRINTS "0" and EXITS 1 when there is no match, so `|| echo 0`
# appends a second line and the caller's `read` then sees a shifted, half-empty
# record. Use `|| true` and let the printed 0 stand.
n_train=$(pgrep -cf 'main[_]ppo' 2>/dev/null || true)
n_train=${n_train:-0}
log_bytes=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk '{s+=$1} END {print s+0}')
if tail -60 "$LOG" 2>/dev/null | grep -qE '^\+ exit 0'; then done=DONE; else done=RUNNING; fi

printf '%s %s %s %s\n' "${n_train:-0}" "${log_bytes:-0}" "${gpu_mib:-0}" "$done"
