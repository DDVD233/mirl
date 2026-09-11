#!/usr/bin/env python3
"""Quantitative anatomy of the hack-then-patch loop for the stage-2 paper.

Reads, per RRIMed run, the gen-server ledgers under logs_hb9b/<run>/ and the trainer
launch log, and writes one JSON with three views:

  funnel      exploit cases by source (referee-shipped vs adversary re-attack), patch
              outcomes, criteria accepted by sign, normalized margin removed per
              criterion, rounds per task, sealed rubrics, exploit-mode labels,
              exploit/honest score statistics, share of served tasks that were patched.
  series      per training step: referee discordance H / concordance C / decisive pairs,
              referee position and length preference, exploits found and buffered,
              exploits shipped, patches applied and criteria accepted, mean solver
              score; per memo version: admission-probe margin, honest mean, adversary
              win rate; per guidance version: validation score credited to it.
  patch_effect for every patched task, the mean served-rollout score reported to the
              server before the first patch and after it (server_reports rows), the
              number of rollouts on each side, and the pooled before/after means.

Everything is measured from logs; nothing is estimated. Run on a pod with the NFS:

  python3 stage2_adversary_ledger.py --root /scratch/sheng/self_evolving \
      --run hb9b_specgap_ship_retrieval_websearch --launch-log specgap_arm10_websearch_launch.log \
      --out ledger_hb9b.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict


def read_jsonl(paths):
    for p in paths:
        with open(p, errors="ignore") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    yield json.loads(l)
                except Exception:
                    continue


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def quantiles(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"p10": q(0.10), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "p90": q(0.90)}


def funnel(ev, patches, accepted, reports, sealed_count):
    src = Counter(e.get("source") for e in ev)
    modes = Counter(e.get("mode") for e in ev)
    items = [(e, it) for e in ev for it in ((e.get("minted_criterion") or {}).get("items") or [])]
    pos = [it for _, it in items if (it.get("points") or 0) > 0]
    neg = [it for _, it in items if (it.get("points") or 0) < 0]
    per_item = [pi for e in ev for pi in ((e.get("acceptance") or {}).get("per_item") or [])]
    gap_drop = [pi.get("gap_drop") for pi in per_item]
    gd_pos = [pi.get("gap_drop") for pi in per_item if pi.get("sign") == "positive"]
    gd_neg = [pi.get("gap_drop") for pi in per_item if pi.get("sign") == "negative"]
    exploit = [e["scores"].get("exploit_score_original") for e in ev if e.get("scores")]
    honest = [e["scores"].get("honest_score_original") for e in ev if e.get("scores")]
    margin = [x - y for x, y in zip(exploit, honest) if x is not None and y is not None]
    ref_margin = [e["scores"].get("referee_margin") for e in ev if e.get("scores")]
    rounds = Counter((e.get("minted_criterion") or {}).get("round") for e in ev)
    per_task = Counter(e["question_id"] for e in ev)
    items_per_task = Counter()
    for e, it in items:
        items_per_task[e["question_id"]] += 1
    crit_len = [len(it.get("criterion_text") or "") for _, it in items]
    orig_len = [len(c.get("criterion_text") or "") for e in ev for c in (e.get("original_rubric") or [])]
    orig_pts = [abs(c.get("points") or 0) for e in ev for c in (e.get("original_rubric") or [])]
    mint_pts = [abs(it.get("points") or 0) for _, it in items]
    num_re = re.compile(r"\d")
    specific = lambda s: bool(num_re.search(s or ""))
    orig_num = mean([1.0 if specific(c.get("criterion_text")) else 0.0
                     for e in ev for c in (e.get("original_rubric") or [])])
    mint_num = mean([1.0 if specific(it.get("criterion_text")) else 0.0 for _, it in items])
    mention_re = re.compile(r"\b(mentions?|in some way|addresses|discusses|acknowledges)\b", re.I)
    orig_mention = mean([1.0 if mention_re.search(c.get("criterion_text") or "") else 0.0
                         for e in ev for c in (e.get("original_rubric") or [])])
    mint_mention = mean([1.0 if mention_re.search(it.get("criterion_text") or "") else 0.0 for _, it in items])

    pt_out = Counter(p.get("outcome") for p in patches)
    applied = sum(1 for p in patches if p.get("applied"))
    served_tasks = {a.get("question_id") for a in accepted}
    patched_tasks = set(per_task)
    reported_tasks = {r.get("question_id") for r in reports}
    return {
        "cases": len(ev), "cases_by_source": dict(src), "tasks_with_case": len(per_task),
        "cases_per_task": dict(Counter(per_task.values())),
        "rounds": {str(k): v for k, v in rounds.items()},
        "criteria_accepted": len(items), "positive": len(pos), "negative": len(neg),
        "positive_share": (len(pos) / len(items)) if items else None,
        "criteria_per_case_mean": (len(items) / len(ev)) if ev else None,
        "criteria_per_patched_task": quantiles(list(items_per_task.values())),
        "gap_drop_mean": mean(gap_drop), "gap_drop_quantiles": quantiles(gap_drop),
        "gap_drop_mean_positive": mean(gd_pos), "gap_drop_mean_negative": mean(gd_neg),
        "exploit_score_mean": mean(exploit), "honest_score_mean": mean(honest),
        "exploit_minus_honest_mean": mean(margin), "exploit_minus_honest_quantiles": quantiles(margin),
        "referee_margin_mean": mean(ref_margin),
        "exploit_score_ge_1_share": mean([1.0 if (x or 0) >= 1.0 else 0.0 for x in exploit]),
        "modes": dict(modes),
        "patch_outcomes": dict(pt_out), "patch_attempts": len(patches), "patches_applied": applied,
        "sealed": sealed_count,
        "served_tasks": len(served_tasks), "patched_tasks": len(patched_tasks),
        "patched_share_of_served": (len(patched_tasks & served_tasks) / len(served_tasks)) if served_tasks else None,
        "patched_share_of_reported": (len(patched_tasks & reported_tasks) / len(reported_tasks)) if reported_tasks else None,
        "criterion_chars_minted": quantiles(crit_len), "criterion_chars_original": quantiles(orig_len),
        "points_abs_minted_mean": mean(mint_pts), "points_abs_original_mean": mean(orig_pts),
        "numeric_share_minted": mint_num, "numeric_share_original": orig_num,
        "mention_phrasing_share_minted": mint_mention, "mention_phrasing_share_original": orig_mention,
        "original_rubric_size": quantiles([len(e.get("original_rubric") or []) for e in ev]),
        "gap_drops_per_item": [{"sign": pi.get("sign"), "gap_drop": pi.get("gap_drop")} for pi in per_item],
    }


LAUNCH_KEYS = {
    "spec_gap/H/mean": "H", "spec_gap/C/mean": "C", "spec_gap/decisive_pairs/mean": "decisive_pairs",
    "spec_gap/n_groups": "n_groups", "spec_gap/measured_groups_frac": "measured_groups_frac",
    "spec_gap/frac_groups_H_gt_half": "frac_groups_H_gt_half",
    "spec_gap/referee/longer_pref": "referee_longer_pref", "spec_gap/referee/pos_pref": "referee_pos_pref",
    "spec_gap/referee/unstable_pair_frac": "referee_unstable_pair_frac",
    "spec_gap/referee/abstain_frac": "referee_abstain_frac", "spec_gap/referee/fail_frac": "referee_fail_frac",
    "spec_gap/exploits/found": "exploits_found", "spec_gap/exploits/buffered": "exploits_buffered",
    "spec_gap/exploits/disagreements": "exploits_disagreements",
    "critic/score/mean": "train_score_mean", "critic/rewards/mean": "train_reward_mean",
    "response_length/mean": "response_length_mean",
}


def parse_launch_log(path):
    """Per-step metric dict from the trainer's 'step:N - k:v - k:v' lines, plus shipped counts."""
    steps = {}
    shipped = defaultdict(int)
    if not path or not os.path.exists(path):
        return steps, shipped
    kv_re = re.compile(r"([A-Za-z0-9_/\.\-]+):(?:np\.float64\()?(-?[0-9.]+(?:e-?[0-9]+)?)\)?")
    ship_re = re.compile(r"\[spec_gap\] shipped (\d+) exploits -> /patch_spec: \{'step': (\d+)")
    with open(path, errors="ignore") as f:
        for line in f:
            m = ship_re.search(line)
            if m:
                shipped[int(m.group(2))] += int(m.group(1))
                continue
            if "spec_gap/H" not in line and "critic/score/mean" not in line:
                continue
            row = {}
            for k, v in kv_re.findall(line):
                if k in LAUNCH_KEYS:
                    try:
                        row[LAUNCH_KEYS[k]] = float(v)
                    except ValueError:
                        pass
            st = None
            m = re.search(r"training/global_step:(\d+)", line)
            if m:
                st = int(m.group(1))
            else:
                m = re.search(r"\bstep:(\d+)\b", line)
                if m:
                    st = int(m.group(1))
            if st is None or not row:
                continue
            steps.setdefault(st, {}).update(row)
    return steps, shipped


def series(ev, patches, launch_steps, shipped, memo_hist, evolve_hist, val_by_step):
    per_step = defaultdict(lambda: {"cases": 0, "cases_referee": 0, "cases_reattack": 0,
                                    "criteria_accepted": 0, "patches_applied": 0})
    for e in ev:
        s = per_step[int(e.get("step") or 0)]
        s["cases"] += 1
        s["cases_referee" if e.get("source") == "on_policy_exploit" else "cases_reattack"] += 1
        s["criteria_accepted"] += len((e.get("minted_criterion") or {}).get("items") or [])
    for p in patches:
        if p.get("applied"):
            per_step[int(p.get("step") or 0)]["patches_applied"] += 1
    steps = sorted(set(per_step) | set(launch_steps) | set(shipped) | set(val_by_step))
    rows = []
    for st in steps:
        r = {"step": st, **per_step.get(st, {}), **launch_steps.get(st, {}), "exploits_shipped": shipped.get(st, 0)}
        if st in val_by_step:
            r["val_acc_len_adj_signed"] = val_by_step[st]
        rows.append(r)
    memo = []
    for m in memo_hist or []:
        for o in m.get("outcomes") or []:
            memo.append({"version": m.get("version"), "committed_step": m.get("step"), **o})
    evolve = []
    for i, e in enumerate(evolve_hist or []):
        rec = {"version": i + 1, "step": e.get("step"), "changed": e.get("changed")}
        for o in e.get("outcomes") or []:
            rec.update({k: v for k, v in o.items()})
        evolve.append(rec)
    return {"per_step": rows, "memo_versions": memo, "guidance_versions": evolve}


def patch_effect(ev, reports):
    """Served-rollout score before vs after the first patch, per patched task."""
    first_patch = {}
    for e in ev:
        q, ts = e["question_id"], e.get("ts") or ""
        if q not in first_patch or ts < first_patch[q]:
            first_patch[q] = ts
    by_q = defaultdict(list)
    for r in reports:
        q = r.get("question_id")
        if q in first_patch:
            by_q[q].append((r.get("ts") or "", r.get("accuracy")))
    rows = []
    pooled_before, pooled_after = [], []
    for q, ts in first_patch.items():
        reps = by_q.get(q, [])
        before = [a for t, a in reps if t < ts and a is not None]
        after = [a for t, a in reps if t >= ts and a is not None]
        if before and after:
            rows.append({"question_id": q, "n_before": len(before), "n_after": len(after),
                         "mean_before": mean(before), "mean_after": mean(after)})
            pooled_before += before
            pooled_after += after
    deltas = [r["mean_after"] - r["mean_before"] for r in rows]
    return {"tasks_with_both_sides": len(rows), "patched_tasks": len(first_patch),
            "pooled_mean_before": mean(pooled_before), "pooled_mean_after": mean(pooled_after),
            "n_rollouts_before": len(pooled_before), "n_rollouts_after": len(pooled_after),
            "per_task_delta_mean": mean(deltas), "per_task_delta_quantiles": quantiles(deltas),
            "share_tasks_score_fell": mean([1.0 if d < 0 else 0.0 for d in deltas]),
            "per_task": rows}


def val_curve(root, run):
    out = {}
    d = os.path.join(root, "logs_hb9b", "val_generations", run)
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        try:
            st = int(os.path.basename(p).split(".")[0])
        except ValueError:
            continue
        vals = []
        with open(p, errors="ignore") as f:
            for l in f:
                if not l.strip():
                    continue
                try:
                    vals.append(float(json.loads(l).get("acc_len_adj_signed")))
                except Exception:
                    pass
        if vals:
            out[st] = sum(vals) / len(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/sheng/self_evolving")
    ap.add_argument("--run", required=True)
    ap.add_argument("--launch-log", default=None, help="file name under <root>/logs_hb9b, or absolute path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = os.path.join(a.root, "logs_hb9b", a.run)
    ev = list(read_jsonl(sorted(glob.glob(os.path.join(D, "server_patch_evidence_*.jsonl")))))
    patches = list(read_jsonl(sorted(glob.glob(os.path.join(D, "server_patches_*.jsonl")))))
    accepted = list(read_jsonl(sorted(glob.glob(os.path.join(D, "server_accepted_*.jsonl")))))
    reports = list(read_jsonl(sorted(glob.glob(os.path.join(D, "server_reports_*.jsonl")))))
    gen_log = os.path.join(a.root, "logs_hb9b", f"gen_server_{a.run}.log")
    sealed = 0
    if os.path.exists(gen_log):
        with open(gen_log, errors="ignore") as f:
            sealed = sum(1 for l in f if "SEALED spec" in l)
    memo_hist = evolve_hist = None
    for name, target in (("hack_memo_history.json", "memo"), ("evolve_history.json", "evolve")):
        p = os.path.join(D, "prompts", name)
        if os.path.exists(p):
            try:
                obj = json.load(open(p))
            except Exception:
                obj = None
            if target == "memo":
                memo_hist = obj
            else:
                evolve_hist = obj
    launch = a.launch_log
    if launch and not os.path.isabs(launch):
        launch = os.path.join(a.root, "logs_hb9b", launch)
    launch_steps, shipped = parse_launch_log(launch)
    vals = val_curve(a.root, a.run)
    out = {"run": a.run, "n_evidence": len(ev), "n_patch_attempts": len(patches),
           "n_accepted_specs": len(accepted), "n_reports": len(reports),
           "launch_log_steps": len(launch_steps), "val_evals": len(vals),
           "funnel": funnel(ev, patches, accepted, reports, sealed),
           "series": series(ev, patches, launch_steps, shipped, memo_hist, evolve_hist, vals),
           "patch_effect": patch_effect(ev, reports)}
    json.dump(out, open(a.out, "w"), indent=1)
    f = out["funnel"]; pe = out["patch_effect"]
    print(f"{a.run}: cases={f['cases']} {f['cases_by_source']} tasks={f['tasks_with_case']} "
          f"criteria={f['criteria_accepted']} pos_share={f['positive_share']:.3f} "
          f"gap_drop={f['gap_drop_mean']:.3f} exploit-honest={f['exploit_minus_honest_mean']:.3f} "
          f"sealed={f['sealed']} patched_share_served={f['patched_share_of_served']} "
          f"patch_effect before={pe['pooled_mean_before']} after={pe['pooled_mean_after']} "
          f"(tasks {pe['tasks_with_both_sides']}/{pe['patched_tasks']}) launch_steps={len(launch_steps)} "
          f"shipped_total={sum(shipped.values())}")


if __name__ == "__main__":
    main()
