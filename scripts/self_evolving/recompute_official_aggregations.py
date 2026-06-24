"""Recompute HealthBench Professional's OFFICIAL paper aggregations from the
saved per-example grades, in length-adjusted form (the paper's primary metric).

Paper groupings (Figs 4-6):
  - by use case:      consult / writing / research
  - by dataset slice: good_faith x typical, good_faith x difficult,
                      red_teaming x difficult   (JOINT type x difficulty)
  - by specialty:     per HF specialty tag

Per example we have the RAW rubric score + the model's completion; the
length-adjusted score is  la = raw - 0.0147*(chars - 2000)/500 . Each group
score = clip(mean(la over the group's examples), 0, 1) (and raw likewise),
matching the harness's clipped-mean aggregation.

Run on server4 (per-example files live there):
    python3 recompute_official_aggregations.py
Writes <out>/HEALTHBENCH_OFFICIAL_AGG.json.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

CKPT_DIR = "/scratch/sheng/self_evolving/eval_healthbench"
BASE_DIR = "/scratch/sheng/self_evolving/logs/healthbench_professional"
HF_HOME = "/scratch/sheng/self_evolving/hf_cache"
LEN_CENTER, LEN_PEN_PER_500 = 2000.0, 0.0147

BASE_DISP = {
    "Qwen_Qwen3.5-9B": "Qwen3.5-9B", "Qwen_Qwen3.6-27B": "Qwen3.6-27B",
    "google_gemma-4-31B-it": "gemma-4-31B-it", "gpt-5.3-chat_2026-03-03": "gpt-5.3-chat",
}


def load_base_overalls():
    """Exact v2 (gpt-5.3-graded) base overall scores from the consolidated file —
    used to identify which per-example file is which model (the true v2 file
    reproduces this score exactly, so a tight match is unambiguous)."""
    d = json.load(open(f"{CKPT_DIR}/HEALTHBENCH_RESULTS.json"))
    return {BASE_DISP[r["model"]]: r["overall_score"]
            for r in d["runs"] if r["kind"] == "base" and r["model"] in BASE_DISP}


def clip01(x):
    return max(0.0, min(1.0, x))


def load_hf_tags():
    os.environ["HF_HOME"] = HF_HOME
    from datasets import load_dataset
    ds = load_dataset("openai/healthbench-professional", split="test")
    return {r["id"]: {"use_case": r["use_case"], "type": r["type"],
                      "difficulty": r["difficulty"], "specialty": r["specialty"]}
            for r in ds}


def per_example(allresults_path):
    """Yield (prompt_id, raw_score, length_adjusted_score)."""
    d = json.load(open(allresults_path))
    for e in d.get("metadata", {}).get("example_level_metadata", []) or []:
        raw = e.get("score")
        if raw is None:
            continue
        comp = e.get("completion") or []
        chars = len(comp[0].get("content", "")) if comp else 0
        la = raw - LEN_PEN_PER_500 * ((chars - LEN_CENTER) / 500.0)
        yield e.get("prompt_id"), raw, la


def aggregate(rows, tags):
    """rows: list of (pid, raw, la). Returns nested official aggregations."""
    def group(keyfn):
        acc = {}
        for pid, raw, la in rows:
            t = tags.get(pid)
            if not t:
                continue
            k = keyfn(t)
            if k is None:
                continue
            acc.setdefault(k, []).append((raw, la))
        return {k: {"n": len(v),
                    "raw": clip01(sum(r for r, _ in v) / len(v)),
                    "length_adjusted": clip01(sum(l for _, l in v) / len(v))}
                for k, v in acc.items()}

    allraw = [r for _, r, _ in rows]
    allla = [l for _, _, l in rows]
    overall = {"n": len(rows),
               "raw": clip01(sum(allraw) / len(allraw)) if allraw else None,
               "length_adjusted": clip01(sum(allla) / len(allla)) if allla else None}
    return {
        "overall": overall,
        "by_use_case": group(lambda t: t["use_case"]),
        "by_slice": group(lambda t: f'{t["type"]}__{t["difficulty"]}'),
        "by_specialty": group(lambda t: t["specialty"]),
    }


def main():
    tags = load_hf_tags()
    print(f"loaded {len(tags)} HF tag rows")
    runs = {}

    # checkpoints: files/<tag>/allresults_*.json
    for f in sorted(glob.glob(f"{CKPT_DIR}/files/*/allresults_*.json")):
        tag = Path(f).parent.name
        rows = list(per_example(f))
        if rows:
            runs[tag] = {"label": tag, "kind": "checkpoint", "n": len(rows),
                         **aggregate(rows, tags)}

    # base models: the v2 per-example file reproduces the known v2 overall
    # EXACTLY. Global-greedy assign (smallest distance first, unique) with a
    # tight tolerance so v1 (Qwen-judged) leftovers whose score coincidentally
    # sits near another model's v2 score never win.
    base_overall = load_base_overalls()
    cands = []
    for f in sorted(glob.glob(f"{BASE_DIR}/allresults_*.json")):
        rows = list(per_example(f))
        if rows:
            cands.append((f, clip01(sum(r for _, r, _ in rows) / len(rows)), rows))
    pairs = sorted(
        ((abs(base_overall[m] - ov), m, f, ov, rows)
         for (f, ov, rows) in cands for m in base_overall),
        key=lambda x: x[0])
    seen_m, seen_f = set(), set()
    for dist, m, f, ov, rows in pairs:
        if dist > 0.0015 or m in seen_m or f in seen_f:
            continue
        seen_m.add(m); seen_f.add(f)
        runs[m] = {"label": m, "kind": "base", "n": len(rows), **aggregate(rows, tags)}
        print(f"  base {f} -> {m} (overall {ov:.4f}, dist {dist:.5f})")

    out = {
        "benchmark": "healthbench_professional",
        "grader": "gpt-5.3-chat_2026-03-03",
        "note": "official paper aggregations recomputed from per-example grades; "
                "length_adjusted = paper primary metric; raw = unadjusted rubric fraction",
        "slices": ["good_faith__typical", "good_faith__difficult", "red_teaming__difficult"],
        "runs": sorted(runs.values(), key=lambda r: (r["kind"], r["label"])),
    }
    Path(f"{CKPT_DIR}/HEALTHBENCH_OFFICIAL_AGG.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {CKPT_DIR}/HEALTHBENCH_OFFICIAL_AGG.json ({len(runs)} runs)")
    # quick sanity print
    for r in out["runs"]:
        if r["kind"] == "base":
            uc = r["by_use_case"]
            print(f"  {r['label']:16s} overall(la)={r['overall']['length_adjusted']:.3f} "
                  f"consult={uc.get('consult',{}).get('length_adjusted')} "
                  f"writing={uc.get('writing',{}).get('length_adjusted')} "
                  f"research={uc.get('research',{}).get('length_adjusted')}")


if __name__ == "__main__":
    main()
