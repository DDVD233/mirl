#!/usr/bin/env python3
"""Render the 'no training data' comparison: every method that touches no benchmark
training data, on the HealthBench Professional validation set, with and without
retrieval at test time, under one grader per column.

Rows come from regrade outputs (scripts/self_evolving/analysis/regrade_hbpro_dumps.py)
in paper_data/stage2/regrade/<tag>/<run>_<step>.json, so a frozen model with an
inference-time RAG method, a frozen model with the solver's tool loop, a model trained
on generated tasks with the fixed proposer prompt, and RRI are all scored by the
same instrument on the same answers format. "Tool loop" rows of checkpoints trained
with tools are their in-loop validation dumps at the same step; the other cells are
val-only runs of the same weights (scripts/self_evolving/eval/onoff_queue.sh).

Usage: python3 scripts/self_evolving/analysis/stage2_onoff_table.py
       [--data paper_data/stage2] [--paper paper_stage2]
"""
import argparse
import json
import os

GRADERS = [("chatlatest_std", "gpt-chat-latest"), ("gpt54low", "GPT-5.4 low")]
METRIC = "acc_len_adj_signed"

# (group title, [(label, retrieval column text, regrade key)])
GROUPS = [
    ("Qwen3.6-27B", [
        ("Frozen, direct answer", "none", "hbpro_methods_qwen36_27b_direct_0"),
        ("Frozen, medical RAG", "one retrieval", "hbpro_methods_qwen36_27b_medrag_0"),
        ("Frozen, RAG-Fusion", "fused queries", "hbpro_methods_qwen36_27b_rag_fusion_0"),
        ("Frozen, i-MedRAG", "iterative", "hbpro_methods_qwen36_27b_imedrag_0"),
        ("Frozen, GEPA evolved prompt", "none", "hbpro_methods_qwen36_27b_gepa_0"),
        ("Frozen, ACE evolved playbook", "none", "hbpro_methods_qwen36_27b_ace_0"),
        ("Frozen, tool loop", "solver tool calls", "hb27b_specgap_ship_retrieval_websearch_0"),
        ("Frozen, no tools", "none", "hbpro_27b_base_notools_0"),
        ("Generated-data training, no tools", "none", "hbpro_27b_fixed60_notools_0"),
        ("Generated-data training, tool loop", "solver tool calls", "hb27b_specgap_simple_retrieval_aicr_60"),
        ("RRI, no tools", "none", "hbpro_27b_ser200_notools_0"),
        ("RRI, tool loop", "solver tool calls", "hb27b_specgap_ship_retrieval_websearch_200"),
    ]),
    ("Qwen3.5-9B", [
        ("Frozen, direct answer", "none", "hbpro_methods_qwen35_9b_direct_0"),
        ("Frozen, medical RAG", "one retrieval", "hbpro_methods_qwen35_9b_medrag_0"),
        ("Frozen, RAG-Fusion", "fused queries", "hbpro_methods_qwen35_9b_rag_fusion_0"),
        ("Frozen, i-MedRAG", "iterative", "hbpro_methods_qwen35_9b_imedrag_0"),
        ("Frozen, GEPA evolved prompt", "none", "hbpro_methods_qwen35_9b_gepa_0"),
        ("Frozen, ACE evolved playbook", "none", "hbpro_methods_qwen35_9b_ace_0"),
        ("Frozen, no tools", "none", "hb9b_specgap_ship_noretrieval_sj_aicr_0"),
        ("Frozen, tool loop", "solver tool calls", "hb9b_specgap_ship_retrieval_websearch_0"),
        ("Generated-data training without tools, no tools", "none", "hb9b_specgap_simple_noretrieval_sj_aicr_860"),
        ("Generated-data training without tools, tool loop", "solver tool calls", "hbpro_9bD_fixed860_tools_0"),
        ("RRI trained without tools, no tools", "none", "hb9b_specgap_ship_noretrieval_sj_aicr_480"),
        ("RRI trained without tools, tool loop", "solver tool calls", "hbpro_9bD_ser480_tools_0"),
        ("Generated-data training with tools, no tools", "none", "hbpro_9bC_fixed80_notools_0"),
        ("Generated-data training with tools, tool loop", "solver tool calls", "hb9b_specgap_simple_retrieval_self9b_websearch_80"),
        ("RRI trained with tools, no tools", "none", "hbpro_9bC_ser200_notools_0"),
        ("RRI trained with tools, tool loop", "solver tool calls", "hb9b_specgap_ship_retrieval_self9b_websearch_200"),
    ]),
]

CAPTION = (r"\caption{Methods that use no training data, on the HealthBench Professional "
           r"validation set, length-adjusted score under two graders applied to the same "
           r"answers. Frozen rows either follow the stage-1 inference protocol, where medical RAG "
           r"retrieves once, RAG-Fusion fuses three generated queries, and i-MedRAG asks three "
           r"rounds of follow-up questions, or answer through the solver's tool loop with clinical "
           r"corpus retrieval and web search. GEPA and ACE evolve the prompt or a playbook of the frozen model on "
           r"generated tasks only and answer the validation set once with the frozen result. Trained rows are the surviving checkpoint of each run, "
           r"answering once with the tool loop and once with no tools. Generated-data training "
           r"is the fixed proposer prompt without the adversary.}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paper_data/stage2")
    ap.add_argument("--paper", default="paper_stage2")
    a = ap.parse_args()
    vals = {}
    for tag, _ in GRADERS:
        d = os.path.join(a.data, "regrade", tag)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if f.endswith(".json"):
                vals[(tag, f[:-5])] = json.load(open(os.path.join(d, f)))["overall"][METRIC]
    L = [r"\begin{table}[h]", r"\begin{center}", CAPTION, r"\label{tab:onoff}", r"\small",
         r"\begin{tabular}{llrr}", r"\toprule",
         "Model & Retrieval at test time & " + " & ".join(g for _, g in GRADERS) + r" \\", r"\midrule"]
    missing = []
    for title, rows in GROUPS:
        L.append(r"\multicolumn{4}{l}{\emph{%s}} \\" % title)
        best = {tag: max((vals.get((tag, k), float("-inf")) for _, _, k in rows), default=None) for tag, _ in GRADERS}
        for label, ret, key in rows:
            cells = []
            for tag, _ in GRADERS:
                v = vals.get((tag, key))
                if v is None:
                    missing.append((tag, key)); cells.append("")
                else:
                    s = f"{v:.3f}"
                    cells.append(r"\textbf{%s}" % s if abs(v - best[tag]) < 1e-12 else s)
            L.append(r"\quad %s & %s & %s \\" % (label, ret, " & ".join(cells)))
        if title != GROUPS[-1][0]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table}"]
    out = os.path.join(a.paper, "tables", "onoff.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("% GENERATED by scripts/self_evolving/analysis/stage2_onoff_table.py. Do not edit by hand.\n"
                         + "\n".join(L) + "\n")
    print("wrote", out)
    if missing:
        print("missing cells:", len(missing))
        for m in missing:
            print("  ", m)
    render_heldout(a)


HELDOUT_CAPTION = (r"\caption{The held-out arms' surviving checkpoints with and without their tool at "
                   r"test time, Qwen3.5-9B, gpt-chat-latest grader. The in-loop validation metric is "
                   r"the rubric score for PRBench and ProfBench and exact-match accuracy for "
                   r"MedXpertQA. Untrained is the base model answering with the same tool.}")


def render_heldout(a):
    """Held-out benchmarks: untrained + tool, trained + tool, trained without tool."""
    p = os.path.join(a.data, "onoff_heldout.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    rows = [("PRBench Hard, finance and legal", "web search", d["prbench_hard"]),
            ("ProfBench, PhD and MBA reports", "web search", d["profbench"]),
            ("MedXpertQA, text and image", "retrieval and web search", d["medxpertqa"])]
    L = [r"\begin{table}[h]", r"\begin{center}", HELDOUT_CAPTION, r"\label{tab:onoff_heldout}", r"\small",
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Benchmark & Tool & Untrained, tool & RRI, tool & RRI, no tool \\", r"\midrule"]
    for name, tool, r in rows:
        vals = [r["untrained_with_tool"], r["trained_with_tool"], r["trained_without_tool"]]
        best = max(vals)
        cells = [(r"\textbf{%.3f}" % v if v == best else "%.3f" % v) for v in vals]
        L.append(r"%s & %s & %s \\" % (name, tool, " & ".join(cells)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table}"]
    out = os.path.join(a.paper, "tables", "onoff_heldout.tex")
    open(out, "w").write("% GENERATED by scripts/self_evolving/analysis/stage2_onoff_table.py. Do not edit by hand.\n"
                         + "\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
