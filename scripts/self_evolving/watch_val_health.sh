#!/usr/bin/env bash
# Per-eval health panel for the 9B retrieval arms: TRUNCATION and BUDGET first,
# then context accounting, lengths and scores. Speaks only when a NEW val step
# lands, so repeated polling does not bury the one line that changed.
#
# WHY THESE COUNTERS. Every failure this pipeline has had showed up first as a
# budget/truncation symptom, not as a score drop:
#   - unclosed think    : rollout never closes </think>; graded as empty, ~-0.55 each
#   - budget_exhausted  : a 3rd search attempt after the 2-call cap
#   - retrieval_truncated / answer_rescued : the loop had to intervene
#   - output at the cap : ran out of response budget mid-answer
# All of these are silent in the accuracy number until enough rollouts hit them.
#
# ANSWER LENGTH IS DERIVED, NOT READ. `extracted_answer` in the val jsonl is capped
# at 512 chars by _result() (it is a log field), so measuring it reports 512 for
# every row and looks like a hard truncation bug that does not exist. The graded
# answer is the full output minus the private think block minus the masked tool
# spans, which is what this computes.
#
#   bash scripts/self_evolving/watch_val_health.sh            # watch forever
#   ONCE=1 bash scripts/self_evolving/watch_val_health.sh     # single report
set -uo pipefail

HOST=${HOST:-root@point.dd.works}
PORT=${PORT:-2335}          # /scratch is shared NFS, so one host reads both arms
POLL_S=${POLL_S:-600}
VG=/scratch/sheng/self_evolving/logs_hb9b/val_generations
ARMS=${ARMS:-"hb9b_gen_retrieval_evolve hb9b_gen_retrieval_selfjudge"}

read -r -d '' PY <<'PYEOF'
import json, os, sys, glob
vg, arms = sys.argv[1], sys.argv[2].split()
state_path = "/tmp/.val_health_seen.json"
try:
    seen = json.load(open(state_path))
except Exception:
    seen = {}
changed = False
for arm in arms:
    files = sorted(glob.glob(os.path.join(vg, arm, "*.jsonl")),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not files:
        continue
    f = files[-1]
    step = int(os.path.basename(f).split(".")[0])
    key = f"{arm}:{step}"
    sz = os.path.getsize(f)
    # Size guard: a step's file is appended as the eval runs, so re-report only
    # when the step is new OR the file finished growing since we last looked.
    if seen.get(key) == sz:
        continue
    seen[key] = sz; changed = True
    rows = []
    for line in open(f):
        try: rows.append(json.loads(line))
        except Exception: pass
    n = len(rows)
    if not n:
        continue
    g = lambda r, k, d=0.0: float(r.get(k) if r.get(k) is not None else d)
    pc = lambda c: 100.0 * c / n
    def q(xs, f_):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(f_ * len(xs)))] if xs else 0

    searched = [r for r in rows if g(r, "n_search") > 0]
    ctx = [g(r, "retrieval_ctx_chars") for r in searched]
    per = [g(r, "retrieval_ctx_chars") / max(1.0, g(r, "n_search")) for r in searched]
    out_len = [len(r.get("output") or "") for r in rows]
    # graded answer = output - think - masked tool spans
    ans = [max(0, len(r.get("output") or "") - g(r, "think_chars") - g(r, "retrieval_ctx_chars"))
           for r in rows]
    unclosed = sum(1 for r in rows if r.get("think_closed") is not None and g(r, "think_closed") == 0.0)
    capped = sum(1 for x in out_len if x >= 28000)   # ~8192 tok at ~3.5 char/tok

    print(f"\n===== {arm}  step {step}  ({n} val rows) =====")
    flags = []
    for k, label in (("retrieval_truncated", "retrieval_truncated"),
                     ("budget_exhausted", "budget_exhausted"),
                     ("answer_rescued", "answer_rescued"),
                     ("retrieval_error", "retrieval_error")):
        c = sum(1 for r in rows if g(r, k) > 0)
        if c: flags.append(f"{label} {c} ({pc(c):.1f}%)")
    if unclosed: flags.append(f"unclosed-think {unclosed} ({pc(unclosed):.1f}%)")
    if capped:   flags.append(f"output-at-cap {capped} ({pc(capped):.1f}%)")
    jf = sum(g(r, "judge_fail") for r in rows) / n
    if jf > 0.001: flags.append(f"judge_fail mean {jf:.3f}")
    print("  TRUNCATION/BUDGET: " + ("; ".join(flags) if flags else "all clear (0 across every counter)"))

    print(f"  retrieval: {len(searched)}/{n} searched ({pc(len(searched)):.0f}%)"
          + (f" | brief chars med {q(per,.5):.0f} p90 {q(per,.9):.0f} max {max(per):.0f}"
             f" | total ctx med {q(ctx,.5):.0f} (~{q(ctx,.5)/3.6:.0f} tok) max {max(ctx):.0f}"
             if searched else ""))
    if per and max(per) >= 6000:
        print("    ^^ a brief hit the 6000-char tool_response cap and was right-truncated")
    print(f"  lengths: answer med {q(ans,.5):.0f} p90 {q(ans,.9):.0f}"
          f" | think med {q([g(r,'think_chars') for r in rows],.5):.0f}"
          f" | output med {q(out_len,.5):.0f} max {max(out_len)}")
    pen = 0.0147 * ((sum(ans) / n) - 2000) / 500.0
    print(f"  length adjustment being paid: {pen:+.4f} (mean answer {sum(ans)/n:.0f} chars; neutral at 2000)")
    for k in ("acc", "acc_raw", "acc_len_adj"):
        xs = [g(r, k) for r in rows if r.get(k) is not None]
        if xs: print(f"  {k}: {sum(xs)/len(xs):.4f}", end="")
    print()
if changed:
    json.dump(seen, open(state_path, "w"))
PYEOF

while true; do
    ssh -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" "$HOST" \
        "/usr/local/bin/python - '$VG' '$ARMS' <<'EOF'
$PY
EOF" 2>/dev/null
    [[ "${ONCE:-0}" == 1 ]] && break
    sleep "$POLL_S"
done
