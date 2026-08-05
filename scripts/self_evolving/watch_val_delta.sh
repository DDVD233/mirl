#!/usr/bin/env bash
# Report the held-out val curve of the two 9B arms, ON CHANGE ONLY.
#
# The naive version (tail the latest score every N minutes) re-emits the same
# number every poll, so real movement is buried in repeats. This tracks the eval
# COUNT per arm and speaks only when a new evaluation lands, printing the arm's
# whole curve plus the matched delta whenever both arms have reached the same step.
#
# The matched delta is the only number that means anything: the arms differ solely
# in retrieval, so (retrieval - control) at the SAME step is the effect. Comparing
# whatever each arm last reported is not that, since they drift out of lockstep.
#
# NOISE FLOOR: two step-0 evaluations of the same untrained model on the full 525
# came out 0.2758 and 0.2970 (2026-08-05), so ~+/-0.02 is sampling noise at
# temperature 1.0. A single-step gap under that is nothing.
set -uo pipefail

HOST=${HOST:-root@point.dd.works}
LOGDIR=/scratch/sheng/self_evolving/logs_hb9b
POLL_S=${POLL_S:-600}
declare -A N; N[RET]=-1; N[CTL]=-1
declare -A CURVE

curve_of() {  # $1=port $2=logfile
    ssh -o ConnectTimeout=15 -o BatchMode=yes -p "$1" "$HOST" \
        "grep -oE 'val-core/overall/acc/mean:[0-9.]+' $LOGDIR/$2 2>/dev/null | grep -oE '[0-9.]+' | tr '\n' ' '" 2>/dev/null
}

while true; do
    changed=0
    for spec in "RET:2335:gen_retrieval.out" "CTL:2336:gen_control.out"; do
        arm=${spec%%:*}; rest=${spec#*:}; port=${rest%%:*}; log=${rest#*:}
        c=$(curve_of "$port" "$log"); [[ -z "$c" ]] && continue
        n=$(wc -w <<<"$c")
        if [[ "$n" -ne "${N[$arm]}" ]]; then
            CURVE[$arm]="$c"; N[$arm]=$n; changed=1
            printf '[%s] %s eval#%d: %s\n' "$(date -u +%H:%MZ)" "$arm" "$((n-1))" \
                   "$(awk '{for(i=1;i<=NF;i++) printf "%.3f ", $i}' <<<"$c")"
        fi
    done

    # Matched-step delta over however many steps BOTH arms have reached.
    if [[ "$changed" -eq 1 && "${N[RET]}" -gt 0 && "${N[CTL]}" -gt 0 ]]; then
        read -r -a r <<<"${CURVE[RET]}"; read -r -a c <<<"${CURVE[CTL]}"
        k=$(( ${#r[@]} < ${#c[@]} ? ${#r[@]} : ${#c[@]} ))
        if [[ "$k" -ge 2 ]]; then
            awk -v k="$k" -v R="${CURVE[RET]}" -v C="${CURVE[CTL]}" 'BEGIN{
                nr=split(R,a," "); nc=split(C,b," "); s=0; pos=0;
                for(i=1;i<=k;i++){d=a[i]-b[i]; s+=d; if(d>0) pos++}
                printf "        matched steps=%d  mean delta=%+.4f  positive=%d/%d%s\n",
                       k, s/k, pos, k, (s/k > 0.02 ? "  (> noise floor)" : "  (within noise)")
            }'
        fi
    fi
    sleep "$POLL_S"
done
