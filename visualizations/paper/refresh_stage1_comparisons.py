"""Refresh only the stage-1 comparisons and official HealthBench breakdowns."""

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "paper_data/stage1"
PAPER = ROOT / "paper"
OUT = Path(__file__).resolve().parent
FONT = "Computer Modern, CMU Serif, serif"
TEAL, SLATE, PINK, GOLD = "#7dbfa7", "#6c8ebf", "#da81c1", "#e0b45c"
CATS = [
    "neoplasms",
    "nervous_system",
    "blood_immune",
    "circulatory",
    "digestive",
    "endocrine_metabolic",
    "infectious",
    "other",
]
CAT_NAMES = ["Neoplasms", "Nervous", "Blood/Imm.", "Circulatory", "Digestive", "Endocrine", "Infectious", "Other"]
SLICES = ["good_faith__typical", "good_faith__difficult", "red_teaming__difficult"]


def read(path):
    return json.loads(path.read_text())


def clip(value):
    return min(1.0, max(0.0, value))


def aggregate_healthbench(directory):
    summaries = list(directory.glob("result__*.json"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one pinned summary in {directory}")
    summary = read(summaries[0])
    if summary["grader_model"] != "gpt-5.4_2026-03-05" or summary["n_examples"] != 525:
        raise ValueError("Wrong grader or partial benchmark")
    stamp = summary["stamp"]
    grade_path = directory / f"allresults_{stamp}.json"
    input_path = directory / f"healthbench_professional_{stamp}.jsonl"
    inputs = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    tags = {r["prompt_id"]: dict(t.split(":", 1) for t in r["example_tags"]) for r in inputs}
    examples = read(grade_path)["metadata"]["example_level_metadata"]
    ids = [e["prompt_id"] for e in examples]
    if len(inputs) != 525 or len(tags) != 525 or len(examples) != 525 or len(set(ids)) != 525 or set(ids) != set(tags):
        raise ValueError("Missing, duplicate, or mismatched HealthBench examples")
    groups = defaultdict(list)
    for example in examples:
        raw = example["score"]
        if raw is None or not math.isfinite(raw):
            raise ValueError("Missing per-example grade")
        content = example["completion"][0]["content"]
        if not isinstance(content, str):
            raise ValueError("Unexpected completion format")
        adjusted = raw - 0.0147 * (len(content) - 2000) / 500
        tag = tags[example["prompt_id"]]
        for key in ("overall", "use/" + tag["use_case"], "slice/" + tag["type"] + "__" + tag["difficulty"]):
            groups[key].append((raw, adjusted))
    result = {
        k: {
            "n": len(v),
            "raw": clip(sum(x[0] for x in v) / len(v)),
            "length_adjusted": clip(sum(x[1] for x in v) / len(v)),
        }
        for k, v in groups.items()
    }
    for metric, key in (("raw", "overall_score"), ("length_adjusted", "overall_score_length_adjusted")):
        if not math.isfinite(summary[key]) or abs(result["overall"][metric] - summary[key]) > 1e-10:
            raise ValueError(f"Recomputed {metric} disagrees with official summary")
    result["provenance"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (summaries[0], grade_path, input_path)
    }
    return result


def save(fig, name, height=430):
    fig.update_layout(
        width=1000,
        height=height,
        font=dict(family=FONT, size=17, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=35, b=100, l=72, r=22),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.23),
    )
    fig.update_xaxes(showgrid=False, linecolor="black")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    for extension in ("pdf", "png"):
        target = OUT / f"{name}.{extension}"
        fig.write_image(str(target), scale=2 if extension == "png" else 1)
        (PAPER / "figures" / target.name).write_bytes(target.read_bytes())


def best_curve(name):
    rows = [(p, read(p)) for p in (ARCHIVE / "fig1_scaling/regraded" / name).glob("*.json")]
    if not rows or any(r["n"] != 2452 for _, r in rows):
        raise ValueError("Missing or partial MIMIC curve")
    return max(rows, key=lambda item: item[1]["overall"])


def refresh_healthbench_ledger():
    hb = {}
    for directory in sorted((ARCHIVE / "healthbench_pro_gpt54").iterdir()):
        if directory.is_dir() and list(directory.glob("result__*.json")):
            hb[directory.name] = aggregate_healthbench(directory)
    (ARCHIVE / "healthbench_official_refresh.json").write_text(
        json.dumps({"grader": "gpt-5.4_2026-03-05", "runs": hb}, indent=2) + "\n"
    )
    return hb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--healthbench-ledger-only", action="store_true", help="Validate saved runs without redrawing figures"
    )
    args = parser.parse_args()
    hb = refresh_healthbench_ledger()
    if args.healthbench_ledger_only:
        print(json.dumps({"official_hb_runs": len(hb), "overall": {k: v["overall"] for k, v in hb.items()}}))
        return
    series = [
        ("Qwen3.6-27B base", "qwen36_27b_base", PINK),
        ("RSIMed-27B", "ours_rlonly_selfimprove", TEAL),
        ("Gemma-4-31B base", "gemma4_31b_base", "#e8a9d2"),
        ("RSIMed-Gemma4", "gemma4_31b_selfimprove", "#4f9e86"),
        ("GPT-5.6", "gpt56_sol", GOLD),
    ]
    lines = [
        r"% Generated by visualizations/paper/refresh_stage1_comparisons.py.",
        r"\begin{table}[h]",
        r"\caption{HealthBench Professional by use case and dataset slice, length-adjusted,",
        r"using the official grader and the variants reported in Table~\ref{tab:main-hb}.",
        r"Group sizes: consult $236$, writing $142$, research $147$;",
        r"good-faith typical $256$, good-faith difficult $78$, red-teaming difficult $191$.}",
        r"\label{tab:hb-slices}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Model & Overall & Consult & Writing & Research & GF typ. & GF diff. & RT diff. \\",
        r"\midrule",
    ]
    columns = ["overall", "use/consult", "use/writing", "use/research"] + ["slice/" + s for s in SLICES]
    for label, key, _ in series:
        vals = [hb[key][col]["length_adjusted"] for col in columns]
        lines.append(label + " & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (PAPER / "tables/healthbench_slices.tex").write_text("\n".join(lines) + "\n")
    fig = go.Figure()
    for label, key, color in series:
        vals = [hb[key]["slice/" + s]["length_adjusted"] for s in SLICES]
        fig.add_bar(
            name=label,
            x=["Good faith, typical", "Good faith, difficult", "Red teaming, difficult"],
            y=vals,
            marker_color=color,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Length-adjusted score", range=[0, 1.05])
    save(fig, "fig_hb_by_slice", 470)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Length-adjusted", "Raw"])
    for col, metric in enumerate(("length_adjusted", "raw"), 1):
        for label, keys, color in [
            ("Base", ["qwen36_27b_base", "gemma4_31b_base", None], PINK),
            ("RSIMed", ["ours_rlonly_selfimprove", "gemma4_31b_selfimprove", None], TEAL),
            ("Frontier reference", [None, None, "gpt56_sol"], GOLD),
        ]:
            vals = [hb[k]["overall"][metric] if k else None for k in keys]
            fig.add_bar(
                x=["Qwen3.6-27B", "Gemma-4-31B", "GPT-5.6"],
                y=vals,
                name=label,
                marker_color=color,
                legendgroup=label,
                showlegend=col == 1,
                text=[f"{v:.3f}" if v is not None else "" for v in vals],
                textposition="outside",
                textfont=dict(size=12),
                row=1,
                col=col,
            )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 0.72], title_text="Score")
    save(fig, "fig_hb_by_family")
    ours_path, ours = best_curve("selfevolving_27b")
    static_path, static = best_curve("staticpool")
    fig = go.Figure()
    for label, row, color in [("Static pool (SFT + RL)", static, SLATE), ("Full pipeline (SFT + RL)", ours, TEAL)]:
        vals = [row["per_cat"][c] for c in CATS] + [row["overall"]]
        fig.add_bar(
            name=label,
            x=CAT_NAMES + ["Overall"],
            y=vals,
            marker_color=color,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Diagnostic accuracy", range=[0, 0.85])
    save(fig, "fig_staticpool_comparison")
    mimic = read(ROOT / "visualizations/mimic_rare/MIMIC_RARE_RESULTS.json")
    candidates = mimic["baselines"] + mimic["trained_all"]
    teachers = [
        ("Base", "qwen36_27b_base"),
        ("Kimi-K2.6", "qwen36_27b_full_kimi"),
        ("DeepSeek<br>V4-Pro", "qwen36_27b_deepseekv4pro"),
        ("GPT-5.5", "qwen36_27b_full_gpt55"),
        ("Qwen3.5<br>397B", "qwen36_27b_full_opd_qwen397b_v2"),
        ("Self-generation<br>RL only", "qwen36_27b_selfimprove__"),
    ]
    labels = [name for name, _ in teachers] + ["Static pool<br>SFT + RL", "Full pipeline<br>SFT + RL"]
    vals = [max(r["overall"] for r in candidates if r["model"].startswith(prefix)) for _, prefix in teachers]
    vals += [static["overall"], ours["overall"]]
    colors = [PINK] + [SLATE] * 4 + ["#4f9e86", GOLD, TEAL]
    (PAPER / "figures/teacher_comparison_data.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"label": label, "accuracy": value, "color": color}
                    for label, value, color in zip(labels, vals, colors, strict=True)
                ],
                "full_pipeline_source": str(ours_path.relative_to(ROOT)),
                "static_pool_source": str(static_path.relative_to(ROOT)),
                "teacher_source": "visualizations/mimic_rare/MIMIC_RARE_RESULTS.json",
            },
            indent=2,
        )
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("paper_teacher_plot", PAPER / "figures/plot_teacher_comparison.py")
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    fig = plotter.make_figure()
    save(fig, "fig_teacher_comparison")
    (ARCHIVE / "comparison_refresh.json").write_text(
        json.dumps(
            {
                "ours": {"source": str(ours_path.relative_to(ROOT)), "overall": ours["overall"]},
                "static_pool": {"source": str(static_path.relative_to(ROOT)), "overall": static["overall"]},
                "teacher_figure": dict(zip(labels, vals, strict=True)),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"official_hb_runs": len(hb), "ours": ours["overall"], "static_pool": static["overall"]}))


if __name__ == "__main__":
    main()
