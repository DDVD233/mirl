#!/usr/bin/env python3
"""Track the CITE SCORE: how often the model satisfies criteria that demand a source.

WHY THIS METRIC. The step-165 val analysis of hb9b_specgap_full_rewrite put 34% of the
remaining unmet positive mass on criteria needing a specific fact the 9B does not carry --
guideline editions, trial identities, citations, exact codes and thresholds. Rubric
criteria that explicitly demand a citation or name a study are the cleanest observable
slice of that: 14.3% of unmet positive mass, and their met rate was FLAT across 165 steps
(0.396 -> 0.406) while everything else moved. If retrieval works, this is the number that
must rise; overall accuracy is too diluted to show it early.

`has_guideline_cite` is decided from the criterion TEXT (keyword match), which is a
heuristic and is reported as such -- it selects the criteria that ask for a source, not the
criteria the model happened to fail.

Usage:
  python3 cite_score.py --run hb9b_specgap_full_retrieval
  python3 cite_score.py --run hb9b_specgap_full_rewrite     # pre-retrieval baseline
"""

import argparse
import glob
import json
import os
import re

# Criterion asks the answer to ground itself in a named source.
CITE_PAT = re.compile(
    r"\b(cite|citation|citing|reference[sd]?|source[sd]?|per the|according to|"
    r"guideline|guidance|consensus statement|trial|study|studies|RCT|cohort|"
    r"meta-analys|systematic review|society|ACC|AHA|ESC|NICE|WHO|USPSTF|IDSA|ASCO|NCCN)\b",
    re.I,
)
# Criterion demands a concrete quantity (the other half of the knowledge-shaped gap).
NUM_PAT = re.compile(r"\d")


def load_criteria(parquet):
    import pandas as pd
    d = pd.read_parquet(parquet)
    rows = []
    for ei in d["extra_info"]:
        # `ei or {}` raises on a parquet struct: numpy arrays have no scalar truthiness.
        # Same for `items or []` when rubric_items comes back as an ndarray.
        ei = ei if hasattr(ei, "get") else {}
        items = ei.get("rubric_items")
        items = [] if items is None else list(items)
        rows.append([
            {"text": str(it.get("criterion_text") or it.get("criterion") or ""),
             "points": float(it.get("points") or 0.0)}
            for it in items if hasattr(it, "get")
        ])
    return rows


def met_map(rubric_met):
    """`rubric_met` -> {criterion_text: met}. Tolerates str or list encodings."""
    out = {}
    v = rubric_met
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            return out
    if isinstance(v, dict):
        for k, m in v.items():
            out[str(k)] = bool(m)
    elif isinstance(v, list):
        for it in v:
            if isinstance(it, dict):
                t = it.get("criterion_text") or it.get("criterion") or it.get("text")
                if t is not None:
                    out[str(t)] = bool(it.get("met"))
    return out


def score_step(path, criteria):
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if len(rows) != len(criteria):
        return None
    agg = {k: [0, 0] for k in ("cite", "numeric", "other")}   # [met, total]
    for row, crits in zip(rows, criteria):
        mm = met_map(row.get("rubric_met"))
        for c in crits:
            if c["points"] <= 0:
                continue
            met = mm.get(c["text"])
            if met is None:
                continue
            bucket = "cite" if CITE_PAT.search(c["text"]) else (
                "numeric" if NUM_PAT.search(c["text"]) else "other")
            agg[bucket][0] += int(met)
            agg[bucket][1] += 1
    out = {"n_rows": len(rows)}
    for k, (m, t) in agg.items():
        out[f"{k}_met"] = (m / t) if t else None
        out[f"{k}_n"] = t
    # retrieval telemetry, when the arm has it
    for f in ("retrieval_used", "retrieval_coverage", "n_search"):
        vals = [r[f] for r in rows if isinstance(r.get(f), (int, float))]
        out[f] = (sum(vals) / len(vals)) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default="/scratch/sheng/self_evolving")
    a = ap.parse_args()

    criteria = load_criteria(os.path.join(a.root, "healthbench_pro_val.parquet"))
    d = os.path.join(a.root, "logs_hb9b", "val_generations", a.run)
    files = []
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        m = re.search(r"(\d+)\.jsonl$", os.path.basename(f))
        if m:
            files.append((int(m.group(1)), f))
    files.sort()
    if not files:
        print(f"no val dumps under {d}")
        return

    print(f"run: {a.run}   (cite = criteria whose text demands a named source)")
    print(f"{'step':>5} {'cite_met':>9} {'n':>5} {'numeric_met':>12} {'other_met':>10} "
          f"{'retr_used':>10} {'retr_cov':>9}")
    for step, f in files:
        s = score_step(f, criteria)
        if not s:
            continue
        def fmt(x, nd=4):
            return "n/a" if x is None else f"{x:.{nd}f}"
        print(f"{step:>5} {fmt(s['cite_met']):>9} {s['cite_n']:>5} "
              f"{fmt(s['numeric_met']):>12} {fmt(s['other_met']):>10} "
              f"{fmt(s['retrieval_used'],3):>10} {fmt(s['retrieval_coverage'],3):>9}")


if __name__ == "__main__":
    main()
