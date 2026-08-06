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
# NOISE FLOOR: THREE step-0 evaluations of the same untrained model on the full 525
# came out 0.276 / 0.297 / 0.309 (2026-08-05) -> sd ~0.017 per eval, so a single
# matched delta carries ~0.024. A one-step gap near that size is nothing.
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
            # The effect is estimated over TRAINED steps only. Index 1 is step 0,
            # where both arms are the same untrained model — a pre-training baseline
            # whose true value was measured at +0.00005 on the full 525. It carries
            # noise like any other eval (+0.0211 on a rerun) but no signal, so
            # averaging it in only dilutes and inflates.
            #
            # Significance uses the noise of a DELTA, not of an eval: two independent
            # evals at ~0.02 each give a difference at ~0.028, and a mean of m
            # deltas at ~0.028/sqrt(m). Requiring 2 sigma keeps this from calling a
            # result off one lucky pair.
            awk -v k="$k" -v R="${CURVE[RET]}" -v C="${CURVE[CTL]}" 'BEGIN{
                split(R,a," "); split(C,b," "); s=0; pos=0; m=0;
                for(i=2;i<=k;i++){d=a[i]-b[i]; s+=d; m++; if(d>0) pos++; if(m==1) df=d; dl=d}
                b0=a[1]-b[1];
                if(m<1){ printf "        step0 baseline %+.4f (null by construction); no trained steps matched yet\n", b0; exit }
                mean=s/m; se=0.028/sqrt(m);
                # Two guards beyond the 2se test, both learned the hard way:
                #  - m>=3. At m=1 the test is one eval against one eval, so a single
                #    downward excursion in the OTHER arm reads as an effect (the control
                #    step-5 was the lowest point on its whole curve when this first fired).
                #  - |step0 baseline| < mean. Step 0 is the same untrained model in
                #    both arms, so it measures how far apart they landed by chance
                #    (0.276/0.297/0.309 across three launches of the same thing). If
                #    that null gap rivals the claimed effect, the effect is not
                #    separable from where the arms happened to start.
                verdict = "not yet significant";
                if (mean > 2*se && m >= 3 && (b0<0?-b0:b0) < mean) verdict = "SIGNIFICANT";
                else if (mean > 2*se && m < 3) verdict = sprintf("above 2se but only %d trained step(s) - not trusted", m);
                else if (mean > 2*se) verdict = "above 2se but step0 baseline rivals it - not trusted";
                # Trend across matched steps, reported either way. A real effect
                # should hold or grow as training proceeds; a decaying one is
                # consistent with regression toward no difference, which is how the
                # first (broken-retrieval) attempt looked before it hit zero.
                trend = (m >= 3) ? (dl - df) / (m - 1) : 0;
                printf "        trained steps=%d  mean delta=%+.4f  positive=%d/%d  (2se=%.3f) %s   [step0 baseline %+.4f]%s\n",
                       m, mean, pos, m, 2*se, verdict, b0,
                       (m >= 3 ? sprintf("  trend %+.4f/step%s", trend,
                                         (trend < -0.005 ? " DECAYING - treat verdict with suspicion" : "")) : "")
            }'
        fi
    fi
    sleep "$POLL_S"
done
