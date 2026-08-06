#!/usr/bin/env bash
# Refresh mtimes under /scratch/dvdai_mit so AICR's 30-day mtime purge never
# collects live project data. Installed as a biweekly scrontab job (see
# `scrontab -l`; runs on the cpu partition on the 1st and 15th).
#
# Only paths whose mtime is already older than AGE_DAYS get touched, so fresh
# files keep their real timestamps (worst-case age between biweekly runs stays
# ~21 days, well under the 30-day purge line). Touching everything under the
# root on purpose: the dvdai_mit account is shared, and collaborators' data
# under /scratch/dvdai_mit should survive the purge too.
set -uo pipefail

ROOT=${1:-/scratch/dvdai_mit}
AGE_DAYS=${AGE_DAYS:-7}
LOG=/scratch/dvdai_mit/self_evolving/logs/touch_scratch.log

n=$(find "$ROOT" -mindepth 1 -mtime +"$AGE_DAYS" \( -type f -o -type d \) 2>/dev/null | wc -l)
find "$ROOT" -mindepth 1 -mtime +"$AGE_DAYS" \( -type f -o -type d \) -exec touch -c -a -m {} + 2>/dev/null

mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) touched $n paths with mtime>${AGE_DAYS}d under $ROOT (host $(hostname), job ${SLURM_JOB_ID:-none})" >> "$LOG"
