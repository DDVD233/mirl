"""Replay the specification-gap statistic on COMPLETED runs. Zero GPU.

This is the "before" number, and the decision gate for the whole mechanism. It
imports the same `rank_group` / `group_stats` / `aggregate_stats` the trainer runs, so
what it reports is what training would have logged.

The pre-registered prediction it tests: H RISES over training on self-generated
rubrics. If H is flat, there is no accumulating hack to attenuate, and the claim
becomes about hack LEVEL rather than its slope — worth knowing before spending a
GPU-day per arm.

Rollout dumps carry no `uid` (that column is added by this branch), but rows sharing
an identical `input` are one group of 8, which recovers the grouping exactly.

Two conditioning facts that must be reported with any number from here:
  - The dumps are written AFTER the zero-variance filter, so H is conditional on the
    group having survived it. `--emulate-zero-var-filter` (default on) reproduces
    that selection so the measured population matches training.
  - `score` in these dumps is PRE retrieval bonus, which is the right quantity for
    the rubric's own ordering.

Usage:
  python scripts/self_evolving/analysis/replay_spec_gap.py \
      --dumps '/scratch/sheng/self_evolving/logs_hb9b/rollouts/hb9b_gen_control/*.jsonl' \
      --referee-base $TRAPI_BASE --referee-key $(cat $S/.trapi_key) \
      --referee-model gpt-chat-latest_2026-05-28 --provider trapi \
      --out /scratch/sheng/self_evolving/spec_gap_replay_control.json
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from verl.utils.reward_score.spec_gap import (  # noqa: E402
    aggregate_stats,
    measure_groups_sync,
)

SCORE_KEYS = ("acc_raw_signed", "acc_raw", "score")


def _strip_thinking(text: str) -> str:
    """Same contract as the reward's: tool spans out, then everything after the last
    </think>. Duplicated here rather than imported so this script needs no torch."""
    t = re.sub(r"<tool_call>.*?(?:</tool_call>|\Z)|<tool_response>.*?(?:</tool_response>|\Z)",
               "", text or "", flags=re.S)
    if "</think>" in t:
        return t.rsplit("</think>", 1)[-1].strip()
    if "<think>" in t:
        return ""
    return t.strip()


def _last_user_turn(prompt_text: str) -> str:
    """The clinician turns as the referee should see them. Dumps store the rendered
    chat template, so this is a best-effort recovery of the user content."""
    parts = re.split(r"<\|im_start\|>|<\|start_header_id\|>", prompt_text or "")
    turns = [p for p in parts if p.strip().startswith("user")]
    if turns:
        body = turns[-1]
        body = re.sub(r"^user\s*(<\|im_sep\|>|<\|end_header_id\|>)?", "", body.strip())
        return re.sub(r"<\|im_end\|>.*$", "", body, flags=re.S).strip()[:8000]
    return (prompt_text or "")[:8000]


def build_groups(rows: list, score_key: str, min_ranked: int,
                 emulate_filter: bool) -> tuple[list, dict]:
    """Group dump rows by identical prompt; return (payload, diagnostics)."""
    by_prompt: dict = {}
    for r in rows:
        by_prompt.setdefault(r.get("input", ""), []).append(r)

    used_key, key_found = score_key, False
    for k in (score_key,) + SCORE_KEYS:
        if any(k in r for r in rows):
            used_key, key_found = k, True
            break
    if not key_found:
        raise SystemExit(f"no score key found in the dump (tried {score_key}, {SCORE_KEYS})")

    payload, n_zero_var, n_excluded, n_small = [], 0, 0, 0
    for i, (prompt, grp) in enumerate(by_prompt.items()):
        rows_idx, answers, scores, lens = [], {}, {}, {}
        for j, r in enumerate(grp):
            # Floor rollouts sit at the reward minimum with no judge call, so pairs
            # against them are trivially concordant and would INFLATE concordance.
            if float(r.get("think_closed", 1.0)) <= 0.5:
                n_excluded += 1
                continue
            ans = _strip_thinking(r.get("output", ""))
            if len(ans.strip()) < 32:
                n_excluded += 1
                continue
            rows_idx.append(j)
            answers[j] = ans
            scores[j] = float(r.get(used_key, 0.0))
            lens[j] = len(ans)
        if len(rows_idx) < min_ranked:
            n_small += 1
            continue
        sc = [scores[j] for j in rows_idx]
        if emulate_filter and statistics.pstdev(sc) < 1e-6:
            n_zero_var += 1
            continue
        payload.append({"uid": f"g{i}", "task": _last_user_turn(prompt), "rows": rows_idx,
                        "answers": answers, "scores": scores, "lens": lens})
    diag = {"n_prompts": len(by_prompt), "n_groups": len(payload), "score_key": used_key,
            "n_zero_var_dropped": n_zero_var, "n_rollouts_excluded": n_excluded,
            "n_groups_too_small": n_small}
    return payload, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True,
                    help="rollout jsonl dumps: a glob, or a comma-separated list of "
                         "globs/paths (shell brace expansion does NOT reach here)")
    ap.add_argument("--referee-base", required=True)
    ap.add_argument("--referee-key", default="")
    ap.add_argument("--referee-model", required=True)
    ap.add_argument("--provider", default="trapi")
    ap.add_argument("--score-key", default="acc_raw_signed")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-pairs", type=int, default=3)
    ap.add_argument("--min-ranked", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-groups", type=int, default=0, help="0 = all; else cap per step")
    ap.add_argument("--no-swap", action="store_true",
                    help="skip the swap-order call (halves cost, loses the stability filter)")
    ap.add_argument("--keep-zero-var", action="store_true",
                    help="do NOT emulate the trainer's zero-variance filter")
    ap.add_argument("--out", default="")
    ap.add_argument("--contrasts", default="", help="jsonl of exploit contrasts to read by hand")
    a = ap.parse_args()

    paths = []
    for part in a.dumps.split(","):
        part = part.strip()
        paths.extend(glob.glob(part) if any(c in part for c in "*?[") else
                     ([part] if os.path.exists(part) else []))
    paths = sorted(set(paths), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    if not paths:
        raise SystemExit(f"no dumps matched {a.dumps}")
    print(f"{len(paths)} dumps: {', '.join(os.path.basename(p) for p in paths)}", flush=True)

    out_rows, contrasts = [], []
    for path in paths:
        step = int(re.sub(r"\D", "", os.path.basename(path)) or 0)
        rows = [json.loads(ln) for ln in open(path) if ln.strip()]
        payload, diag = build_groups(rows, a.score_key, a.min_ranked, not a.keep_zero_var)
        if a.max_groups and len(payload) > a.max_groups:
            payload = payload[:a.max_groups]
            diag["capped_to"] = a.max_groups
        if not payload:
            print(f"step {step}: no measurable groups ({diag})", flush=True)
            continue
        stats = measure_groups_sync(
            payload, a.referee_base, a.referee_key, a.referee_model, provider=a.provider,
            concurrency=a.concurrency, swap=not a.no_swap, margin=a.margin,
            min_pairs=a.min_pairs, step=step, timeout_s=120)
        m = aggregate_stats(stats)
        row = {"step": step, **diag, **{k: round(v, 4) for k, v in m.items()
                                        if isinstance(v, (int, float))}}
        out_rows.append(row)
        print(f"step {step:>4}  H={m['spec_gap/H/mean']:.3f}  "
              f"C={m['spec_gap/C/mean']:.3f}  "
              f"dec_pairs={m['spec_gap/decisive_pairs/mean']:.1f}  "
              f"measured={m['spec_gap/measured_groups_frac']:.2f}  "
              f"D_sd={m['spec_gap/D/std_mean']:.3f}  "
              f"longer_pref={m['spec_gap/referee/longer_pref']:.2f}  "
              f"pos_pref={m['spec_gap/referee/pos_pref']:.2f}  "
              f"unstable={m['spec_gap/referee/unstable_pair_frac']:.2f}  "
              f"fail={m['spec_gap/referee/fail_frac']:.2f}  "
              f"[zero_var_dropped={diag['n_zero_var_dropped']}]", flush=True)
        if a.contrasts:
            from verl.utils.reward_score.spec_gap import pick_exploit
            for g in payload:
                st = stats.get(g["uid"])
                if st is None:
                    continue
                ex = pick_exploit(g["uid"], st, g["scores"], g["answers"], {},
                                  task=g["task"], step=step)
                if ex:
                    contrasts.append(ex)

    if not out_rows:
        raise SystemExit("nothing measured")

    # The pre-registered test: does H rise with training? Spearman over steps.
    steps = [r["step"] for r in out_rows]
    hs = [r["spec_gap/H/mean"] for r in out_rows]
    rho = _spearman(steps, hs) if len(steps) >= 3 else float("nan")
    print("\n=== summary ===")
    print(f"  steps measured        {len(out_rows)}  ({min(steps)}..{max(steps)})")
    print(f"  H first / last        {hs[0]:.3f} -> {hs[-1]:.3f}   (delta {hs[-1] - hs[0]:+.3f})")
    print(f"  Spearman(H, step)     {rho:+.3f}   [prediction: > 0]")
    print(f"  mean decisive pairs   {statistics.mean(r['spec_gap/decisive_pairs/mean'] for r in out_rows):.1f}"
          "   [GATE: >= 4, else lower --margin]")
    print(f"  mean measured frac    {statistics.mean(r['spec_gap/measured_groups_frac'] for r in out_rows):.2f}"
          "   [GATE: >= 0.5]")
    print(f"  mean longer_pref      {statistics.mean(r['spec_gap/referee/longer_pref'] for r in out_rows):.2f}"
          "   [HARD GATE: 0.40-0.65]")
    print(f"  mean pos_pref         {statistics.mean(r['spec_gap/referee/pos_pref'] for r in out_rows):.2f}"
          "   [GATE: 0.40-0.60]")
    print(f"  mean unstable frac    {statistics.mean(r['spec_gap/referee/unstable_pair_frac'] for r in out_rows):.2f}"
          "   [GATE: <= 0.25]")
    print(f"  mean referee fail     {statistics.mean(r['spec_gap/referee/fail_frac'] for r in out_rows):.2f}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"dumps": a.dumps, "referee": a.referee_model, "margin": a.margin,
                       "spearman_H_step": rho,
                       "rows": out_rows}, f, indent=2)
        print(f"\nwrote {a.out}")
    if a.contrasts and contrasts:
        with open(a.contrasts, "w") as f:
            for c in contrasts:
                f.write(json.dumps(c) + "\n")
        print(f"wrote {len(contrasts)} contrasts -> {a.contrasts}\n"
              "READ TEN BY HAND before trusting any of this: if a human cannot see the "
              "exploit in the contrast, the patcher will not either.")
    return 0


def _spearman(xs: list, ys: list) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den else float("nan")


if __name__ == "__main__":
    sys.exit(main())
