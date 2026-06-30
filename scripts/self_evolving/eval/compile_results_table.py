"""Compile the gpt-chat-latest rejudge JSONs (trained checkpoints + baselines)
into combined markdown tables: overall + per-ICD-category accuracy.

    python compile_results_table.py
"""
import json, glob, os, sys

TRAINED = "/scratch/sheng/self_evolving/eval_full_gpt55/rejudge_chatlatest"
BASELINE = "/scratch/sheng/self_evolving/eval_baselines/rejudge_chatlatest"

CATS = ["neoplasms", "nervous_system", "blood_immune", "circulatory",
        "digestive", "endocrine_metabolic", "infectious", "other"]
ABBR = {"neoplasms": "neo", "nervous_system": "nerv", "blood_immune": "blood",
        "circulatory": "circ", "digestive": "dig", "endocrine_metabolic": "endo",
        "infectious": "infx", "other": "other"}


def load(d):
    rows = []
    for f in sorted(glob.glob(f"{d}/*.json")):
        r = json.load(open(f))
        rows.append(r)
    return rows


def fmt_table(rows, title):
    rows = sorted(rows, key=lambda r: r["overall"], reverse=True)
    hdr = ["model", "overall", "n"] + [ABBR[c] for c in CATS]
    print(f"\n### {title}\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        pc = r.get("per_cat", {})
        cells = [r["model"], f"{r['overall']:.3f}", str(r["n"])]
        cells += [f"{pc[c]:.3f}" if c in pc else "-" for c in CATS]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    bl = load(BASELINE)
    tr = load(TRAINED)
    print(f"# MIMIC-IV rare-disease test (n=2452) — judged by gpt-chat-latest_2026-05-28")
    print(f"# standard lenient rubric, boxed->cue->tail extraction, multimodal, thinking-off")
    fmt_table(bl, f"BASELINES ({len(bl)} models)")
    fmt_table(tr, f"TRAINED CHECKPOINTS ({len(tr)} models)")
