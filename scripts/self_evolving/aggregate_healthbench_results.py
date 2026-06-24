"""Consolidate all HealthBench Professional results into one JSON + markdown table.

Scans the on-disk per-run artifacts (no W&B needed) and writes:
  <out>/HEALTHBENCH_RESULTS.json  — every run with headline + per-tag metrics
  <out>/HEALTHBENCH_RESULTS.md    — sorted table (base models + checkpoints)

Sources (robust to mixed formats):
  - result__*.json            (self-describing: base re-run + checkpoint sweep)
  - eval_healthbench/out/*.summary.json  (checkpoint summaries: {score, metrics})
  - run_all_v2.log            (backfill base models that predate result__ files)

Re-runnable anytime; safe to run while the sweeps are still going.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BASE_DIRS = ["/scratch/sheng/self_evolving/logs/healthbench_professional"]
CKPT_DIR = "/scratch/sheng/self_evolving/eval_healthbench"


def _canon(label: str) -> str:
    for pre in ("hbpro-ckpt-", "hbpro-"):
        if label.startswith(pre):
            label = label[len(pre):]
    return label


def _by_tag(metrics: dict) -> dict:
    out = {}
    if not isinstance(metrics, dict):
        return out
    for k, v in metrics.items():
        if ":" in k and not k.endswith((":bootstrap_std", ":n_samples")):
            out[k] = v
    return out


def _record(label, kind, metrics, *, grader=None, n=None, extra=None, src=None):
    canon = _canon(label)
    m = re.search(r"__step(\d+)$", canon)
    step = int(m.group(1)) if m else None
    model = canon[: m.start()] if m else canon
    rec = {
        "label": canon,
        "kind": kind,
        "model": model,
        "step": step,
        "grader": grader or "gpt-5.3-chat_2026-03-03",
        "n_examples": n if n is not None else (metrics or {}).get("overall_score:n_samples"),
        "overall_score": (metrics or {}).get("overall_score"),
        "overall_score_length_adjusted": (metrics or {}).get("overall_score_length_adjusted"),
        "by_tag": _by_tag(metrics),
        "source_file": src,
    }
    if extra:
        rec.update(extra)
    return rec


def collect() -> dict:
    runs: dict[str, dict] = {}

    def add(rec):
        # prefer richer record (one with by_tag / grader / n)
        key = rec["label"]
        cur = runs.get(key)
        score = lambda r: (len(r.get("by_tag") or {}) > 0) + (r.get("n_examples") is not None)
        if cur is None or score(rec) > score(cur):
            runs[key] = rec

    # 1) self-describing result__*.json (base + checkpoints)
    pats = [f"{d}/result__*.json" for d in BASE_DIRS]
    pats += [f"{CKPT_DIR}/files/*/result__*.json"]
    for f in sorted(p for pat in pats for p in glob.glob(pat)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        kind = "checkpoint" if "__step" in _canon(d.get("label", "")) else "base"
        add(_record(d.get("label", Path(f).stem), kind, d.get("metrics", {}),
                    grader=d.get("grader_model"), n=d.get("n_examples"),
                    extra={"model_under_test": d.get("model_under_test")}, src=f))

    # 2) checkpoint summary.json ({score, metrics}) — fallback for any without result__
    for f in sorted(glob.glob(f"{CKPT_DIR}/out/*.summary.json")):
        tag = Path(f).name[: -len(".summary.json")]
        try:
            d = json.load(open(f))
        except Exception:
            continue
        add(_record(tag, "checkpoint", d.get("metrics", d), src=f))

    # 3) backfill base models from run_all_v2.log (predate result__ files)
    for logf in ("/tmp/run_all_v2.log",):
        try:
            txt = Path(logf).read_text(errors="ignore")
        except Exception:
            continue
        # blocks: "EVAL <model> (...) [grader ...]" then later overall/len-adj prints
        for mb in re.finditer(
            r"EVAL\s+(\S+).*?overall_score\s*:\s*([0-9.]+).*?overall_score_length_adjusted\s*:\s*([0-9.]+)",
            txt, re.S):
            model, ov, la = mb.group(1), float(mb.group(2)), float(mb.group(3))
            label = f"hbpro-{model.replace('/', '_')}"
            add(_record(label, "base",
                        {"overall_score": ov, "overall_score_length_adjusted": la},
                        extra={"model_under_test": model}, src=logf))

    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=CKPT_DIR)
    args = ap.parse_args()

    runs = collect()
    rows = sorted(runs.values(), key=lambda r: (r["kind"], r["model"], r["step"] or 0))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    payload = {
        "benchmark": "healthbench_professional",
        "grader": "gpt-5.3-chat_2026-03-03 (reasoning_effort=none)",
        "generated": stamp,
        "n_runs": len(rows),
        "runs": rows,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "HEALTHBENCH_RESULTS.json").write_text(json.dumps(payload, indent=2))

    # markdown table
    lines = [
        f"# HealthBench Professional results ({len(rows)} runs)",
        f"_grader gpt-5.3-chat_2026-03-03 · generated {stamp}_", "",
        "| kind | model | step | n | overall | length-adj (primary) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    def fnum(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"
    for r in rows:
        lines.append(
            f"| {r['kind']} | {r['model']} | {r['step'] if r['step'] is not None else '—'} "
            f"| {r['n_examples'] or '—'} | {fnum(r['overall_score'])} "
            f"| {fnum(r['overall_score_length_adjusted'])} |")
    (out / "HEALTHBENCH_RESULTS.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {out}/HEALTHBENCH_RESULTS.json and .md  ({len(rows)} runs)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
