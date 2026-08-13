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
ARM="${ARM:?set ARM=1 fixed-prompt | 2 refine-loop | 3 evolve-only | 4 v1+selfjudge | 5 v3+selfjudge | 6 v2+selfjudge | 7 full+retrieval | 8 full+retrieval+solver-websearch | 9 fixed-prompt+websearch | 10 adversary-v2 (ship+always-patch+multi-round)}"

# Where the self-judge arms get their 9B grader. server5 already serves Qwen3.5-9B on a
# dedicated GPU, exposed through frp, so pointing at it keeps all four GPUs on the
# TRAINING node available for training -- the run script only spawns a local frozen 9B
# when SUMM_BASE is not already answering (it curls $SUMM_BASE/models first), so simply
# naming a live remote endpoint suppresses the local copy and frees a GPU.
#
# Validation is unaffected: it stays pinned to gpt-chat-latest in every arm.
SJUDGE_REMOTE="${SJUDGE_REMOTE:-http://point.dd.works:18184/v1}"
# DEDICATED inference box for the retrieval arm: a whole 4-GPU node serving the frozen 9B
# at DP=4/TP=1 (see serve/serve_frozen9b_dp.sh). server5's single GPU is fine for a judge
# call per rollout, but the retrieval SUMMARIZER is ~256-800 calls per step, all arriving
# at once, all on the generation critical path -- so it, not the trainer, sets step time.
# The 18186 port is the pod's pre-existing frp "comfyui" tunnel to local 8188.
SUMM_DEDICATED="${SUMM_DEDICATED:-http://point.dd.works:18186/v1}"
POLL_S="${POLL_S:-300}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"
# Effectively unbounded by default. Safe because the LR schedule does not depend on
# it (lr_warmup_steps_ratio=0.0, constant LR), and because max_actor_ckpt_to_keep=1
# prunes the actor weights from every checkpoint but the newest -- an old step dir is
# 7.5K against 106G for the live one -- so a long run costs no more disk than a short
# one. Stop a run by killing it; resume_mode=auto brings it back from the last
# checkpoint if the box is preempted.
STEPS="${STEPS:-100000}"
# Appended to the arm's experiment name. A fresh suffix gives a fresh checkpoint dir,
# which is how you RESTART rather than resume: with the same name, resume_mode=auto
# would silently continue the previous run from its last checkpoint.
EXP_SUFFIX="${EXP_SUFFIX:-}"

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
  4) # Judge comparison. Identical to the completed hb9b_gen_control except that the
     # TRAINING reward is graded by the frozen local 9B instead of gpt-chat-latest;
     # validation stays on gpt-chat-latest in every arm, so the held-out numbers are
     # one comparable series. Single factor: the judge.
     #
     # Worth running because the existing partial pair points the counter-intuitive
     # way. The gpt judge assigns 0.87-0.94 on train rollouts -- nearly saturated,
     # which matches the farmer beating honest answers on 99% of rubrics -- while the
     # 9B assigns 0.69-0.79 and led on held-out accuracy at steps 20 and 25. A
     # stronger judge that gives points away may simply be a worse reward.
     #
     # No PROBE here on purpose: the probe grades with the generator's endpoint, not
     # the training judge, so its farmability number would describe a reward this arm
     # is not training on. That routing needs fixing before the two can be combined.
     # EVOLVE is off because the script forbids EVOLVE+SELF_JUDGE (two factors).
     ARM_ENV=(RETRIEVAL=0 EVOLVE=0 SELF_JUDGE=1 SPEC_GAP=1 SUMM_BASE="$SJUDGE_REMOTE")
     EXP_NAME=hb9b_specgap_measure_selfjudge ;;
  5) # JUDGE SWAP at fixed variant: this is ARM=2 (the full pipeline) with the TRAINING
     # reward graded by the frozen local 9B instead of gpt-chat-latest, and nothing else
     # changed. Its control is ARM=2 itself, so the single factor is the judge --
     # which is why ALLOW_EVOLVE_SELF_JUDGE is set: the run script's default guard exists
     # to stop EVOLVE+SELF_JUDGE being compared against the gpt-judge BASELINE, a
     # different and genuinely confounded comparison.
     #
     # Validation stays on gpt-chat-latest in every arm, so held-out numbers remain one
     # comparable series and only the training signal moves.
     ARM_ENV=(RETRIEVAL=0 EVOLVE=1 SPEC_GAP=1 SPEC_GAP_SHIP=0 PROBE=1 PATCH=1
              HACK_MEMO=1 HB_PROBE_MODE=gate SELF_JUDGE=1
              ALLOW_EVOLVE_SELF_JUDGE=1 SUMM_BASE="$SJUDGE_REMOTE")
     EXP_NAME=hb9b_specgap_full_selfjudge ;;
  6) # ARM=3 (evolution only) with the local 9B judge. Same swap, one variant down, so
     # the judge effect can be read at two pipeline depths rather than one.
     ARM_ENV=(RETRIEVAL=0 EVOLVE=1 SPEC_GAP=1 SELF_JUDGE=1
              ALLOW_EVOLVE_SELF_JUDGE=1 SUMM_BASE="$SJUDGE_REMOTE")
     EXP_NAME=hb9b_specgap_evolveonly_selfjudge ;;
  7) # THE FULL PIPELINE PLUS RETRIEVAL. Same as ARM=2 with RETRIEVAL=1.
     #
     # Why: the val analysis at step 165 put 34% of the remaining unmet positive mass on
     # criteria that need a specific fact the 9B does not have -- guideline EDITIONS (2024
     # AUA/SUFU, 2025 ESC/EACTS), trial identities and citations (ACORN, PAPILLON, SOAP
     # II), exact ICD-10 codes, exact thresholds. That shape is retrieval-shaped, not
     # scale-shaped, and no curriculum or reward change can invent it. Every other lever
     # measured (+0.115 traps, +0.03 clarification regressions) leaves that 34% untouched.
     #
     # SUMM_BASE names a REMOTE endpoint: RETRIEVAL=1 sets FROZEN_NEEDED=1, and the run
     # script only spawns a local frozen summarizer when SUMM_BASE is not already
     # answering -- so naming a live endpoint keeps all four GPUs on training instead of
     # surrendering one to a summarizer.
     #
     # Primary is the dedicated 4-GPU box (DP=4); server5 is the fallback, tried once per
     # brief before /retrieve degrades to raw passages. Ordering matters: the fallback is
     # correct but slow, so it must never be the thing serving 256 calls a step.
     # SUMMARY_CONCURRENCY rises with the primary's replica count -- the default 96 was
     # sized for one GPU and would leave three of four replicas idle.
     # WEB_EVIDENCE=1: Milvus alone cannot supply post-2019 guidance, and the criteria name
     # 2022-2025 guidelines and trials. Both the generator and the solver query it, through
     # one cache service, with a 60s ceiling that degrades to Milvus-only.
     ARM_ENV=(RETRIEVAL=1 EVOLVE=1 SPEC_GAP=1 SPEC_GAP_SHIP=0 PROBE=1 PATCH=1
              HACK_MEMO=1 HB_PROBE_MODE=gate HB_REFINE_MODE=rewrite
              SUMM_BASE="$SUMM_DEDICATED" SUMM_FALLBACK_BASE="$SJUDGE_REMOTE"
              SUMMARY_CONCURRENCY="${SUMMARY_CONCURRENCY:-320}"
              WEB_EVIDENCE="${WEB_EVIDENCE:-1}"
              # 16 was under-provisioned and the breaker paid for it. At ~8s a call, 16
              # slots serve ~2 calls/s while a step's ~800 retrieves arrive at ~2.7/s, so the
              # queue grew, the 60s budget (which covers queue wait) expired, and 96 timeouts
              # opened the breaker 19 times -- ~38 min with web evidence off. 32 slots serve
              # ~4/s, above arrival, and add ~240 req/min against the ~2000/60s TRAPI cap
              # shared with the judge and the generator.
              WEB_EVIDENCE_CONCURRENCY="${WEB_EVIDENCE_CONCURRENCY:-32}")
     EXP_NAME=hb9b_specgap_full_retrieval ;;
  8) # ARM=7 WITH THE WEB PATH MOVED INTO THE SOLVER. Identical pipeline, but the
     # GPT web lookup is OFF (WEB_EVIDENCE=0) and the solver instead carries its
     # own `web_search` tool whose results come VERBATIM from the Serper API
     # (kb/serper_cache_server.py; two-tool config medical_retrieval_web_tool.yaml).
     # No external model touches the evidence path anywhere: /retrieve is
     # Milvus + frozen-9B brief, web is raw SERP blocks for queries the policy
     # itself writes -- so the search behaviour AND the reading of raw results
     # both train, and no GPT assists answer production (audit 2026-08-12: the
     # GPT lookup complied in practice, but compliance was prompt-enforced only).
     #
     # NOT single-factor vs ARM=7: the web pathway swap necessarily also removes
     # the GENERATOR's web grounding of minted tasks (WEB_EVIDENCE gates both).
     # Read arm7-vs-arm8 as "GPT-mediated web vs solver-native web", not as a
     # controlled ablation of one line.
     ARM_ENV=(RETRIEVAL=1 EVOLVE=1 SPEC_GAP=1 SPEC_GAP_SHIP=0 PROBE=1 PATCH=1
              HACK_MEMO=1 HB_PROBE_MODE=gate HB_REFINE_MODE=rewrite
              SUMM_BASE="$SUMM_DEDICATED" SUMM_FALLBACK_BASE="$SJUDGE_REMOTE"
              SUMMARY_CONCURRENCY="${SUMMARY_CONCURRENCY:-320}"
              WEB_EVIDENCE="${WEB_EVIDENCE:-0}"
              WEB_SEARCH_TOOL=1)
     EXP_NAME=hb9b_specgap_full_retrieval ;;
  9) # THE FIXED-PROMPT CONTROL FOR ARM=8. Same solver stack -- retrieval, the
     # solver-native web_search tool, the same 9B, WEB_EVIDENCE off -- but the task
     # prompt is FIXED: no evolution, no probe/patch, no hack memo. SPEC_GAP=1 stays
     # because measurement is inert (ARM=1's rationale). Arm8 minus arm9 is the
     # contribution of the evolving pipeline under the web-search stack; arm9 vs the
     # completed lookup arms reads the web-pathway swap at fixed prompt.
     # This arm lives on SERVER1 (2 GPUs), so its box-specific env is baked HERE,
     # not passed at launch: the watchdog relaunches an arm with nothing but
     # ARM+EXP_SUFFIX, and defaults tuned for a 4-GPU box would hand verl
     # n_gpus_per_node=4 on a 2-GPU pod. Summarizer roles are SWAPPED vs arm8:
     # primary is server5's single 9B (freed when the 27B pipeline stopped,
     # already serving with the qwen3_coder parser), fallback is the 2335 DP=4
     # box -- so the two arms do not burst into the same primary. Serper snapshot
     # is per-arm: two boxes writing one snapshot file would alternate
     # full-db overwrites.
     ARM_ENV=(RETRIEVAL=1 EVOLVE=0 SPEC_GAP=1
              N_GPUS="${N_GPUS:-2}"
              SUMM_BASE="$SJUDGE_REMOTE" SUMM_FALLBACK_BASE="$SUMM_DEDICATED"
              SUMMARY_CONCURRENCY="${SUMMARY_CONCURRENCY:-96}"
              SEARCH_SNAPSHOT=/scratch/sheng/self_evolving/kb/search_cache_arm9.sqlite
              WEB_EVIDENCE="${WEB_EVIDENCE:-0}"
              WEB_SEARCH_TOOL=1)
     EXP_NAME=hb9b_specgap_measure_retrieval ;;
  10) # ADVERSARY V2: ARM=8's stack with the repair loop fully connected, after the
      # 0812 audit showed the v1 adversary never treated the training distribution
      # (SPEC_GAP_SHIP=0 meant zero on-policy patches; probe rate 0.34 + one weak
      # rewrite round touched 5.7% of served specs). Four changes, all measured-in:
      #
      #  SHIP ON. The referee's confirmed exploits (and, new, its ordering
      #  disagreements on H>0.5 groups) POST to /patch_spec. The ARM=2-era concern
      #  -- the referee is wrong on 41% of decisive pairs, so its verdicts must not
      #  write the reward -- is answered ARITHMETICALLY now: every minted criterion
      #  is graded against the contrast pair and kept only if it separates it, and
      #  the frozen farmer re-attacks the patched rubric between rounds. A wrong
      #  referee verdict costs a rejected mint, never a bad patch.
      #
      #  ALWAYS PATCH. Probe rate 1.0 (was 0.34), and when the farmer wins the spec
      #  is repaired best-effort even when no rewrite clears the admission bar --
      #  serving the least-farmable version measured beats serving the original.
      #
      #  MULTI-ROUND, DENSER. Two hack->patch rounds at admission AND on-policy;
      #  patches may add several positive and negative criteria (positives pay the
      #  substance the exploit withheld -- denser signal, and the repair direction a
      #  lone negative could never express). Patched rubrics may grow to 9 items.
      #
      #  BACKGROUND. Refinement chains and /patch_spec run as background tasks;
      #  admission and the trainer's evolve hook never block on the adversary.
      ARM_ENV=(RETRIEVAL=1 EVOLVE=1 SPEC_GAP=1 SPEC_GAP_SHIP=1 PROBE=1 PATCH=1
               HACK_MEMO=1 HB_PROBE_MODE=gate HB_REFINE_MODE=rewrite
               HB_PROBE_RATE=1.0 HB_REFINE_ROUNDS=2 HB_REFINE_BACKGROUND=1
               HB_PATCH_ROUNDS=2 HB_PATCH_ASYNC=1 HB_PATCH_MAX_PER_QID=6
               HB_PATCH_MIN_MARGIN=0.15 HB_PATCHED_MAX_ITEMS=9
               HB_MEMO_MAX_CHARS=2400
               SUMM_BASE="$SUMM_DEDICATED" SUMM_FALLBACK_BASE="$SJUDGE_REMOTE"
               SUMMARY_CONCURRENCY="${SUMMARY_CONCURRENCY:-320}"
               WEB_EVIDENCE="${WEB_EVIDENCE:-0}"
               WEB_SEARCH_TOOL=1)
      EXP_NAME=hb9b_specgap_ship_retrieval ;;
  *) echo "FATAL: ARM must be 1..10" >&2; exit 1 ;;
esac

EXP_NAME="${EXP_NAME}${EXP_SUFFIX}"
echo "=== waiting to launch ARM=$ARM ($EXP_NAME, STEPS=$STEPS) on $(hostname) ==="
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
exec env "${ARM_ENV[@]}" EXP="$EXP_NAME" REPO="$REPO" STEPS="$STEPS" \
     WANDB_API_KEY="$WANDB_API_KEY" \
     bash scripts/self_evolving/train/run_9b_hb_gen.sh
