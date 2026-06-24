"""Aggregate the per-checkpoint MIMIC-rare eval JSONLs into one results.json:
per-ICD-category metrics for every checkpoint, with the best checkpoint per run.

Recomputes metrics directly from each <tag>.jsonl (not the per-run summary,
whose n_total reflects only the last resumed pass) so numbers are exact and
uniform. A checkpoint tag is "<run>__step<N>"; checkpoints are grouped by run
and the best one is the checkpoint with the highest overall judge_acc_lenient
(falling back to exact-match acc).

Usage (on server 1):
    /usr/local/bin/python scripts/self_evolving/aggregate_eval_results.py \
        --out_dir /scratch/sheng/self_evolving/eval_gpt53
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

METRICS = ("acc", "judge_acc_lenient", "judge_acc_strict",
           "answer_quality", "reasoning_quality", "embed_sim",
           "char_bleu", "format_ok", "score")
PRIMARY = "judge_acc_lenient"   # headline; falls back to acc if all-zero


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def summarize_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    n_errors = sum(1 for r in rows if r.get("error"))
    overall = {}
    for m in METRICS:
        vals = [r[m] for r in rows if isinstance(r.get(m), (int, float)) and not isinstance(r.get(m), bool)]
        overall[m] = round(_mean(vals), 4)
    by_cat = {}
    for ds in sorted({r.get("data_source", "") for r in rows if r.get("data_source")}):
        ds_rows = [r for r in rows if r.get("data_source") == ds]
        entry = {"n": len(ds_rows)}
        for m in METRICS:
            vals = [r[m] for r in ds_rows if isinstance(r.get(m), (int, float)) and not isinstance(r.get(m), bool)]
            entry[m] = round(_mean(vals), 4)
        by_cat[ds] = entry
    return overall, by_cat, len(rows), n_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/scratch/sheng/self_evolving/eval_gpt53")
    ap.add_argument("--results", default="")
    ap.add_argument("--judge_model", default="gpt-5.3-chat_2026-03-03")
    args = ap.parse_args()
    results_path = args.results or os.path.join(args.out_dir, "results.json")

    checkpoints = {}
    runs = defaultdict(list)
    cats_seen = set()
    for jp in sorted(glob.glob(os.path.join(args.out_dir, "out", "*.jsonl"))):
        tag = os.path.basename(jp)[:-len(".jsonl")]
        if not os.path.exists(jp.replace(".jsonl", ".summary.json")):
            continue  # incomplete
        m = re.match(r"^(?P<run>.+)__step(?P<step>\d+)$", tag)
        if not m:
            continue
        run, step = m.group("run"), int(m.group("step"))
        overall, by_cat, n_rows, n_err = summarize_jsonl(jp)
        cats_seen.update(by_cat.keys())
        checkpoints[tag] = {"run": run, "step": step, "n_rows": n_rows,
                            "n_errors": n_err, "overall": overall, "by_category": by_cat}
        runs[run].append(tag)

    run_out = {}
    for run, tags in runs.items():
        best = max(tags, key=lambda t: (checkpoints[t]["overall"].get(PRIMARY, 0.0),
                                        checkpoints[t]["overall"].get("acc", 0.0)))
        ov = checkpoints[best]["overall"]
        run_out[run] = {
            "checkpoints": sorted(tags, key=lambda t: checkpoints[t]["step"]),
            "best_checkpoint": best,
            "best_step": checkpoints[best]["step"],
            "best_overall": ov,
            "best_by_category": checkpoints[best]["by_category"],
        }

    results = {
        "judge_model": args.judge_model,
        "primary_metric": PRIMARY,
        "n_runs": len(run_out),
        "n_checkpoints": len(checkpoints),
        "categories": sorted(cats_seen),
        "runs": run_out,
        "checkpoints": checkpoints,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {results_path}: {len(checkpoints)} checkpoints, {len(run_out)} runs, "
          f"{len(cats_seen)} categories")
    print(f"\n{'RUN (best checkpoint)':46s} {PRIMARY:>10s} {'acc':>7s} {'fmt':>6s}")
    for run in sorted(run_out, key=lambda r: -run_out[r]["best_overall"].get(PRIMARY, 0)):
        ro = run_out[run]; ov = ro["best_overall"]
        print(f"  {run[:44]:44s} step{ro['best_step']:<5d} {ov.get(PRIMARY,0):8.4f} {ov.get('acc',0):7.4f} {ov.get('format_ok',0):6.2f}")


if __name__ == "__main__":
    main()
