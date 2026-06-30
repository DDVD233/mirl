"""Combined ranked table: baselines + best-per-run trained checkpoints, judged by
gpt-chat-latest. Each training run collapses to its single best-overall step.

    python compile_combined_table.py
"""
import json, glob, re

TRAINED = "/scratch/sheng/self_evolving/eval_full_gpt55/rejudge_chatlatest"
BASELINE = "/scratch/sheng/self_evolving/eval_baselines/rejudge_chatlatest"
CATS = ["neoplasms", "nervous_system", "blood_immune", "circulatory",
        "digestive", "endocrine_metabolic", "infectious", "other"]
ABBR = {"neoplasms": "neo", "nervous_system": "nerv", "blood_immune": "blood",
        "circulatory": "circ", "digestive": "dig", "endocrine_metabolic": "endo",
        "infectious": "infx", "other": "other"}
MEDICAL = ("lingshu", "medgemma", "llava")


def load(d):
    return [json.load(open(f)) for f in glob.glob(f"{d}/*.json")]


bl = load(BASELINE)
tr = load(TRAINED)

# collapse each training run (strip __stepNN) to its best-overall checkpoint
runs = {}
for r in tr:
    run = re.split(r"__step", r["model"])[0]
    if run not in runs or r["overall"] > runs[run]["overall"]:
        runs[run] = r

rows = []
for r in bl:
    kind = "teacher" if "397" in r["model"] else (
        "medical" if any(m in r["model"] for m in MEDICAL) else "base")
    rows.append((r["model"], kind, r))
for run, r in runs.items():
    step = re.search(r"__step(\d+)", r["model"])
    label = f"{run} (step{step.group(1)})" if step else run
    rows.append((label, "trained", r))

rows.sort(key=lambda x: x[2]["overall"], reverse=True)

hdr = ["model", "kind", "overall"] + [ABBR[c] for c in CATS]
print("# MIMIC-IV rare-disease (n=2452) — gpt-chat-latest judge, multimodal, thinking-off")
print("# baselines + best checkpoint per training run\n")
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for label, kind, r in rows:
    pc = r.get("per_cat", {})
    cells = [label, kind, f"{r['overall']:.3f}"]
    cells += [f"{pc[c]:.3f}" if c in pc else "-" for c in CATS]
    print("| " + " | ".join(cells) + " |")
