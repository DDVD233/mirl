"""Emit a LaTeX file with the full MIMIC-rare eval results: per-ICD-category
exact-match accuracy and embedding similarity for every run's best checkpoint,
plus the frontier baseline (gpt-5.1-chat). Reads results.json (the aggregated
sweep) + the baseline summary.json.

Usage:
    python scripts/self_evolving/make_results_tex.py \
        --results /scratch/sheng/self_evolving/eval_gpt53/results.json \
        --baseline /scratch/sheng/self_evolving/eval_gpt53/baseline/gpt51_chat.summary.json \
        --out tex/results.tex
"""

import argparse
import json
import os

CATS = ["neoplasms", "endocrine_metabolic", "nervous_system", "blood_immune",
        "circulatory", "infectious", "digestive", "other"]
SHORT = {"neoplasms": "Neopl", "endocrine_metabolic": "Endo", "nervous_system": "Nerv",
         "blood_immune": "Blood", "circ": "Circ", "circulatory": "Circ",
         "infectious": "Infec", "digestive": "Diges", "other": "Other"}


def esc(s):
    return s.replace("_", r"\_")


def load_rows(results_path, baseline_path):
    d = json.load(open(results_path))
    rows = []  # (label, overall_metric_dict, by_cat_dict, is_baseline)
    for run, ro in d["runs"].items():
        rows.append((run, ro["best_overall"], ro["best_by_category"], False))
    if baseline_path and os.path.exists(baseline_path):
        b = json.load(open(baseline_path))
        # baseline by_category values are dicts of {metric: {mean,n}} OR flat;
        # eval_sota summary stores _agg dicts -> take ['mean'].
        bycat = {}
        for ds, m in b.get("by_category", {}).items():
            key = ds.replace("mimic_rare/", "")
            bycat[ds] = {mk: (mv["mean"] if isinstance(mv, dict) else mv) for mk, mv in m.items()}
        overall = {mk: (mv["mean"] if isinstance(mv, dict) else mv) for mk, mv in b["overall"].items()}
        model = b.get("model", "gpt-5.1-chat")
        rows.append((f"{model} (baseline)", overall, bycat, True))
    return rows


def cat_val(by_cat, cat, metric):
    return by_cat.get("mimic_rare/" + cat, {}).get(metric, 0.0) or 0.0


def make_table(rows, metric, caption, label):
    # order by overall metric desc
    rows = sorted(rows, key=lambda r: -(r[1].get(metric, 0.0) or 0.0))
    # best per column (overall + each cat) for bolding
    best_ovr = max((r[1].get(metric, 0.0) or 0.0) for r in rows)
    best_cat = {c: max(cat_val(r[2], c, metric) for r in rows) for c in CATS}
    col = "l" + "c" * (1 + len(CATS))
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{col}}}", r"\toprule"]
    hdr = ["Run / model", "Overall"] + [SHORT[c] for c in CATS]
    lines.append(" & ".join(hdr) + r" \\")
    lines.append(r"\midrule")
    for label_, overall, by_cat, is_base in rows:
        ov = overall.get(metric, 0.0) or 0.0
        cells = [rf"\texttt{{{esc(label_)}}}" + (r" $^\dagger$" if is_base else "")]
        ovs = f"{ov:.3f}"
        cells.append(rf"\textbf{{{ovs}}}" if abs(ov - best_ovr) < 1e-9 else ovs)
        for c in CATS:
            v = cat_val(by_cat, c, metric)
            vs = f"{v:.3f}"
            cells.append(rf"\textbf{{{vs}}}" if abs(v - best_cat[c]) < 1e-9 and v > 0 else vs)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/sheng/self_evolving/eval_gpt53/results.json")
    ap.add_argument("--baseline", default="/scratch/sheng/self_evolving/eval_gpt53/baseline/gpt51_chat.summary.json")
    ap.add_argument("--out", default="tex/results.tex")
    ap.add_argument("--judge", default="gpt-5.3-chat (sweep) / gpt-5.1-chat (baseline)")
    args = ap.parse_args()

    rows = load_rows(args.results, args.baseline)
    ncats = "Neopl=neoplasms (n=695), Endo=endocrine/metabolic (494), Nerv=nervous system (388), " \
            "Blood=blood/immune (223), Circ=circulatory (141), Infec=infectious (102), " \
            "Diges=digestive (88), Other (321)."

    doc = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[landscape,margin=1.5cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\begin{document}",
        r"\section*{MIMIC-IV rare-disease diagnosis: per-ICD-category evaluation}",
        r"All 15 training runs (best checkpoint per run) plus a frontier baseline, evaluated on the "
        r"2452-case MIMIC-IV rare-disease test set (greedy, $\le$8192 tokens), scored with the verl "
        r"reward pipeline. \textbf{Accuracy is the lenient LLM-judge disease match} (judge: "
        r"gpt-5.3-chat\_2026-03-03): a prediction is correct if the judge rules the diagnosed disease "
        r"matches the ground truth, even if the exact ICD-10 code or phrasing differs (strict exact-match "
        r"is lower). Embedding similarity is cosine between the predicted and ground-truth diagnosis "
        r"(Qwen3-VL-Embedding-2B). $^\dagger$ frontier baseline. Best per column in \textbf{bold}.",
        r"\\[4pt]",
        rf"\footnotesize Categories: {ncats}",
        r"\normalsize",
        make_table(rows, "judge_acc_lenient", "Per-category accuracy (lenient LLM-judge disease match, gpt-5.3-chat).", "tab:acc"),
        make_table(rows, "embed_sim", "Per-category embedding similarity to ground-truth diagnosis.", "tab:embed"),
        r"\end{document}",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"Wrote {args.out} ({len(rows)} rows incl. baseline)")


if __name__ == "__main__":
    main()
