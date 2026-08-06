#!/usr/bin/env bash
# Pull data from the MSR NFS onto AICR. Run on the AICR login node.
#
# Relies on the ssh aliases in ~/.ssh/config.dd (msr = point.dd.works:2333,
# msr2 = :2334, mib = mib.media.mit.edu), authenticated with
# ~/.ssh/mib_transfer_ed25519 whose pubkey is installed on all three hosts.
#
# Usage:
#   sync_from_msr.sh <remote-path-under-/scratch/sheng/self_evolving> <local-dest>
#   sync_from_msr.sh --standard        # re-sync the small standard set (traces, val parquet)
#
# Standard layout on AICR:
#   /work/mit/ppliang_mit/dvdai/self_evolving   small persistent data (7-day snapshots)
#   /scratch/dvdai_mit/self_evolving            big artifacts (10 TiB quota, 30-DAY PURGE)
set -euo pipefail

MSR=/scratch/sheng/self_evolving
WORK_DATA=/work/mit/ppliang_mit/dvdai/self_evolving
SCRATCH_DATA=/scratch/dvdai_mit/self_evolving

if [[ "${1:-}" == "--standard" ]]; then
    mkdir -p "$WORK_DATA"
    rsync -az --partial --info=stats1 "msr:$MSR/*.parquet" "msr:$MSR/*.jsonl" "$WORK_DATA/"
    exit 0
fi

[[ $# -eq 2 ]] || { echo "usage: $0 <remote-path> <local-dest> | --standard" >&2; exit 1; }
mkdir -p "$(dirname "$2")"
exec rsync -a --partial --info=progress2 "msr:$1" "$2"
