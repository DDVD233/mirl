#!/usr/bin/env python3
"""Render the stage-2 paper tables from the raw HB-Pro result JSONs.

Inputs (all under paper_data/stage2/):
  arms.json                       paper row -> run mapping (blocks, ablations)
  hb_pro_results_{msr,aicr}_0821.json   outputs of hb_pro_results_table.py

Outputs:
  paper_data/stage2/stage2_table.json   best-validation summary per row (keeps the
                                        selected eval index internally; never rendered)
  paper_stage2/tables/hbpro_main.tex, hbpro_ablation.tex, hbpro_specialty.tex

Rules: one endpoint per row = the eval with the highest official accuracy
(acc_len_adj_signed, length-adjusted signed rubric score); raw rubric accuracy is
reported from the same eval. "Untrained" = the first eval of the run (before any
update), same grader as the trained rows of its block. No checkpoint indices appear
in any rendered output.
"""
import argparse, json, os

PRIM, SEC = "acc_len_adj_signed", "acc_raw"
CATS = [("consult", "Consult"), ("research", "Research"), ("writing", "Writing")]
DIFF = [("typical", "Typical"), ("difficult", "Difficult")]


def load(root):
    arms = json.load(open(os.path.join(root, "arms.json")))
    files = {"msr": json.load(open(os.path.join(root, "hb_pro_results_msr_0821.json"))),
             "aicr": json.load(open(os.path.join(root, "hb_pro_results_aicr_0821.json")))}
    return arms, files


def pick(files, row):
    run = files[row["file"]]["runs"][row["run"]]
    ps = run["per_step"]
    if row.get("use") == "step0":
        key = str(min(int(k) for k in ps))
    else:
        key = str(run["summary"]["best"]["step"])
    st = ps[key]
    out = {"label": row["label"], "run": row["run"], "ours": row.get("ours", False),
           "untrained": row.get("use") == "step0", "_eval_key": key,
           "n_evals": run["summary"]["n_evals"],
           "overall": {PRIM: st[PRIM], SEC: st[SEC]},
           "category": {k: {PRIM: v[PRIM], SEC: v[SEC], "n": v["n"]} for k, v in st.get("category", {}).items()},
           "difficulty": {k: {PRIM: v[PRIM], SEC: v[SEC], "n": v["n"]} for k, v in st.get("difficulty", {}).items()},
           "specialty": {k: {PRIM: v[PRIM], SEC: v[SEC], "n": v["n"]} for k, v in st.get("specialty", {}).items()},
           "aux": {k: st.get(k) for k in ("think_chars", "format_ok", "judge_fail", "think_closed")}}
    return out


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def cell(x, bold):
    s = f3(x)
    return f"\\textbf{{{s}}}" if bold else s


def render_block_rows(rows, metric, bold_best=True):
    """rows -> list of tex lines; bold = best among trained rows per column."""
    cols = [("overall", None)] + [("category", c) for c, _ in CATS] + [("difficulty", d) for d, _ in DIFF]
    vals = []
    for r in rows:
        v = []
        for dim, k in cols:
            v.append(r[dim][metric] if k is None else r[dim].get(k, {}).get(metric))
        vals.append(v)
    best = []
    for j in range(len(cols)):
        cand = [vals[i][j] for i, r in enumerate(rows) if not r["untrained"] and vals[i][j] is not None]
        best.append(max(cand) if cand else None)
    lines = []
    for i, r in enumerate(rows):
        name = r["label"]
        if r["ours"]:
            name = f"\\textbf{{{name}}}"
        cells = [cell(vals[i][j], bold_best and not r["untrained"] and vals[i][j] is not None and vals[i][j] == best[j])
                 for j in range(len(cols))]
        # order: categories, difficulty, overall last (matches the stage-1 table convention)
        ordered = cells[1:4] + cells[4:6] + [cells[0]]
        lines.append(f"\\quad {name} & " + " & ".join(ordered) + " \\\\")
    return lines


HEADER = (r"Model / training & Consult & Research & Writing & Typical & Difficult & Overall \\")


def render_main(table, metric, caption, label, size=r"\small"):
    L = [r"\begin{table}[t]", r"\begin{center}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         size, r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{3}{c}{By category} & \multicolumn{2}{c}{By difficulty} & \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}", HEADER, r"\midrule"]
    for b in table["blocks"]:
        L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{b['title']}}}}} \\\\")
        L += render_block_rows(b["rows"], metric)
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def render_ablation(table, metric, caption, label):
    L = [r"\begin{table}[t]", r"\begin{center}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\small", r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{3}{c}{By category} & \multicolumn{2}{c}{By difficulty} & \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}", HEADER, r"\midrule"]
    for key in ("ablation", "ablation_retrieval"):
        b = table[key]
        L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{b['title']}}}}} \\\\")
        L += render_block_rows(b["rows"], metric)
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def render_specialty(table, caption, label):
    """Per-specialty accuracy (primary metric) for every block: fixed prompt vs ours."""
    specs = sorted({s for b in table["blocks"] for r in b["rows"] for s in r["specialty"]})
    cols = []
    for b in table["blocks"]:
        fp = next(r for r in b["rows"] if not r["untrained"] and not r["ours"])
        ours = next(r for r in b["rows"] if r["ours"])
        cols.append((b["key"], fp, ours))
    L = [r"\begin{table}[h]", r"\begin{center}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{lr" + "cc" * len(cols) + "}", r"\toprule",
         "Specialty & $n$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{Block {k}}}" for k, _, _ in cols) + r" \\",
         " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(cols))),
         "& & " + " & ".join("Fixed & Ours" for _ in cols) + r" \\", r"\midrule"]
    for s in specs:
        n = next((r["specialty"][s]["n"] for _, fp, _ in cols for r in [fp] if s in r["specialty"]), None)
        cells = []
        for _, fp, ours in cols:
            a = fp["specialty"].get(s, {}).get(PRIM)
            b = ours["specialty"].get(s, {}).get(PRIM)
            ba = a is not None and b is not None and a > b
            bb = a is not None and b is not None and b >= a
            cells += [cell(a, ba), cell(b, bb)]
        L.append(f"{s} & {n if n is not None else '--'} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paper_data/stage2")
    ap.add_argument("--paper", default="paper_stage2")
    a = ap.parse_args()
    arms, files = load(a.data)
    table = {"primary_metric": PRIM, "secondary_metric": SEC, "val_tasks": 525,
             "blocks": [{"key": b["key"], "title": b["title"], "base": b["base"], "judge": b["judge"],
                         "retrieval": b["retrieval"], "rows": [pick(files, r) for r in b["rows"]]}
                        for b in arms["blocks"]]}
    for key in ("ablation", "ablation_retrieval"):
        table[key] = {"title": arms[key]["title"], "rows": [pick(files, r) for r in arms[key]["rows"]]}
    json.dump(table, open(os.path.join(a.data, "stage2_table.json"), "w"), indent=1)

    T = os.path.join(a.paper, "tables")
    os.makedirs(T, exist_ok=True)
    provenance = "% GENERATED by scripts/self_evolving/analysis/stage2_paper_tables.py from paper_data/stage2/stage2_table.json. Do not edit by hand.\n"
    cap_main = (r"HealthBench Professional accuracy (official length-adjusted rubric score, all 525 tasks) "
                r"by category, by difficulty, and overall. Each group of rows is one training setting, sharing one "
                r"base model and grading protocol; within a setting, \emph{Fixed prompt} and \emph{SER (ours)} share "
                r"the identical RL recipe, solver tools, grader, and data budget, and differ only in whether the "
                r"reward is held fixed or allowed to evolve. "
                r"Each trained row reports its best validation evaluation. Bold marks the better trained row per column.")
    open(os.path.join(T, "hbpro_main.tex"), "w").write(provenance + render_main(table, PRIM, cap_main, "tab:main"))
    cap_raw = (r"Unadjusted rubric accuracy (fraction of rubric points earned, with the length term omitted) "
               r"at the evaluations of Table~\ref{tab:main}.")
    open(os.path.join(T, "hbpro_main_raw.tex"), "w").write(provenance + render_main(table, SEC, cap_raw, "tab:main-raw"))
    cap_abl = (r"Component ablation of SER on Qwen3.5-9B under the GPT grader. The upper group runs with retrieval "
               r"off, with meta-prompt evolution alone and then evolution with the admission-time adversary probe. "
               r"The lower group runs with retrieval and web search, from the fixed prompt through evolution with "
               r"the admission-time probe to the full on-policy hack-then-patch loop. Official length-adjusted "
               r"accuracy, best validation evaluation per row.")
    open(os.path.join(T, "hbpro_ablation.tex"), "w").write(provenance + render_ablation(table, PRIM, cap_abl, "tab:ablation"))
    cap_sp = (r"Per-specialty accuracy (official length-adjusted score) for the fixed-prompt and SER rows "
              r"of each setting in Table~\ref{tab:main}. $n$ is the number of validation tasks in the specialty.")
    open(os.path.join(T, "hbpro_specialty.tex"), "w").write(provenance + render_specialty(table, cap_sp, "tab:specialty"))

    for b in table["blocks"] + [table["ablation"], table["ablation_retrieval"]]:
        print("==", b["title"])
        for r in b["rows"]:
            print(f"   {r['label']:45s} {f3(r['overall'][PRIM])} / {f3(r['overall'][SEC])}  "
                  f"diff={ {k: f3(v[PRIM]) for k, v in r['difficulty'].items()} }")


if __name__ == "__main__":
    main()
