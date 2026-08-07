#!/usr/bin/env bash
# Wait for a box to free, then launch a specification-gap arm on it.
#
# WHY A WAITER AND NOT A MANUAL LAUNCH. The two arms that free up next
# (hb9b_gen_retrieval_evolve on 2335, hb9b_gen_retrieval_selfjudge on 2336) are
# mid-run and must finish: never cancel a running allocation. This polls for their
# natural completion and starts immediately after, so no GPU-hours are lost to a
# gap while nobody is watching.
#
# Run it ON the node it will launch on, inside a tmux window:
#   ARM=1 tmux new-window -t hb -n wait 'bash .../launch_specgap_when_free.sh'
#
# ARM=1  measurement only. SPEC_GAP=1 and nothing else: the referee ranks each GRPO
#        group and H is logged, but nothing the actor trains on changes. This is BOTH
#        the baseline val curve and the evidence for "the specification gap grows as
#        the policy trains" -- one run, two results, because measurement is inert.
# ARM=2  the full loop. Refinement before serving (PROBE+PATCH), on-policy exploit
#        repair (SPEC_GAP_SHIP), and the memo that stops future rubrics having the
#        same hole (HACK_MEMO). EVOLVE=1 is required: the exploit buffer drains on
#        the evolve round.
set -euo pipefail

S=/scratch/sheng/self_evolving
REPO=${REPO:-$S/verl_specgap}
ARM="${ARM:?set ARM=1 (measure-only) | 2 (full refine loop) | 3 (evolve-only control)}"
POLL_S="${POLL_S:-300}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"

case "$ARM" in
  1) ARM_ENV=(RETRIEVAL=0 EVOLVE=0 SPEC_GAP=1)
     EXP_NAME=hb9b_specgap_measure ;;
  2) # SPEC_GAP_SHIP is OFF, deliberately. The referee was measured against
     # physician-written rubrics and gets 41% of decisive pairs wrong, so routing
     # ITS verdicts into rubric patches would inject a near-coin-flip signal into
     # the reward. The treatment here is driven entirely by the FROZEN FARMER, whose
     # comparison is farmed-vs-honest on the SAME rubric under the SAME grader that
     # trains on it -- a measurement in the reward's own units, needing no second
     # opinion. SPEC_GAP=1 stays so H is still logged as a diagnostic.
     ARM_ENV=(RETRIEVAL=0 EVOLVE=1 SPEC_GAP=1 SPEC_GAP_SHIP=0 PROBE=1 PATCH=1
              HACK_MEMO=1 HB_PROBE_MODE=gate)
     EXP_NAME=hb9b_specgap_full ;;
  3) # The single-factor control for ARM=2, and the arm that was missing. ARM=2 runs
     # EVOLVE=1 as well as the refine loop, because the exploit memo is rewritten
     # inside the evolve round -- so comparing it against ARM=1 (EVOLVE=0) confounds
     # two factors. This is ARM=2 minus PROBE/PATCH/HACK_MEMO and nothing else, so
     # the difference between them IS the refine loop.
     ARM_ENV=(RETRIEVAL=0 EVOLVE=1 SPEC_GAP=1)
     EXP_NAME=hb9b_specgap_evolveonly ;;
  *) echo "FATAL: ARM must be 1, 2 or 3" >&2; exit 1 ;;
esac

echo "=== waiting to launch ARM=$ARM ($EXP_NAME) on $(hostname) ==="
echo "    repo=$REPO  poll=${POLL_S}s  give up after ${MAX_WAIT_H}h"

deadline=$(( SECONDS + MAX_WAIT_H * 3600 ))
while :; do
    # A trainer is done when no main_ppo process is left AND every GPU is idle.
    # Both checks matter: vLLM worker processes routinely outlive the trainer and
    # hold memory, and launching into that OOMs on engine init ("Free memory on
    # cuda:N < desired"). The pgrep pattern is bracketed so it cannot match this
    # script's own command line.
    if pgrep -f "main[_]pp[o]" >/dev/null 2>&1; then
        busy="trainer running"
    else
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
        if [ "${used:-99999}" -gt 4000 ]; then
            busy="trainer gone but ${used}MiB still held (orphan vLLM workers?)"
        else
            busy=""
        fi
    fi
    [ -z "$busy" ] && break

    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "FATAL: still busy after ${MAX_WAIT_H}h ($busy); not launching" >&2
        exit 1
    fi
    printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$busy"
    sleep "$POLL_S"
done

echo "=== box free at $(date -u) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

# Orphaned shared-memory segments from a previous vLLM survive process exit and make
# CUDA-graph capture fail with "custom_all_reduce.cuh: invalid argument".
rm -f /dev/shm/vllm* 2>/dev/null || true

# The run script hard-fails without WANDB_API_KEY, and this waiter lives in a tmux
# window whose server may have been started by a bare ssh command with no such
# variable -- which is exactly how one arm died on the first line after waiting for a
# box to free. Recover it from ~/.netrc (where wandb login puts it) rather than
# depending on an inherited environment.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.netrc" ]; then
    WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' \
                    "$HOME/.netrc")
    export WANDB_API_KEY
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "FATAL: no WANDB_API_KEY and none in ~/.netrc; the run script would exit on it" >&2
    exit 1
fi

cd "$REPO"
echo "=== launching $EXP_NAME from $(git log --oneline -1) ==="
exec env "${ARM_ENV[@]}" EXP="$EXP_NAME" REPO="$REPO" WANDB_API_KEY="$WANDB_API_KEY" \
     bash scripts/self_evolving/train/run_9b_hb_gen.sh
