"""Paper-ready versions of the two main-text figures.

The figures under visualizations/mimic_rare and visualizations/health_bench_pro are
built for screen reading: very wide, small type, and a title baked into the image.
For the paper the caption carries the title and the figure has to stay legible at
\\textwidth, so these are re-emitted with paper aspect ratios, larger type, and no
in-image titles. Data, palette, and typography are unchanged.

  fig_delta_selfevolving_vs_trainset  self-evolving vs train-set RL, per ICD category
  fig_teacher_comparison              self-evolving vs four external teachers, both benchmarks

Outputs {pdf,png} next to this file.
"""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from refresh_stage1_comparisons import main as refresh_stage1_comparisons

HERE = Path(__file__).resolve().parent
VIS = HERE.parent
DELTA = json.load(open(VIS / "mimic_rare" / "DELTA_DATA.json"))
MIMIC = json.load(open(VIS / "mimic_rare" / "MIMIC_RARE_RESULTS.json"))
HB = json.load(open(VIS / "health_bench_pro" / "HEALTHBENCH_RESULTS.json"))

FONT = "Computer Modern, CMU Serif, serif"
GRID = "rgba(128,128,128,0.2)"
TEAL, PINK, GOLD, SLATE = "#7dbfa7", "#da81c1", "#e0b45c", "#6c8ebf"

CATS = ["neoplasms", "nervous_system", "blood_immune", "circulatory",
        "digestive", "endocrine_metabolic", "infectious", "other"]
CATNAME = {"neoplasms": "Neoplasms", "nervous_system": "Nervous", "blood_immune": "Blood/Imm.",
           "circulatory": "Circulatory", "digestive": "Digestive",
           "endocrine_metabolic": "Endocrine", "infectious": "Infectious", "other": "Other"}


def save(fig, stem):
    if stem in {"fig_teacher_comparison", "fig_hb_by_family", "fig_hb_by_slice"}:
        return  # Written from pinned archives by refresh_stage1_comparisons below.
    fig.write_image(str(HERE / f"{stem}.pdf"))
    fig.write_image(str(HERE / f"{stem}.png"), scale=2)
    print(f"wrote {stem}")


# ---------- Figure 1: self-evolving vs training on the real train set ----------
def deltas(run):
    base, best = DELTA[run]["base"], DELTA[run]["best"]
    d = [best["per_cat"][c] - base["per_cat"][c] for c in CATS]
    d.append(best["overall"] - base["overall"])
    return [100.0 * v for v in d]


SI, TD = deltas("self_improve"), deltas("train_only")
XCATS = [CATNAME[c] for c in CATS] + ["Overall"]

fig = go.Figure()
fig.add_bar(name="RL on the real training set", x=XCATS, y=TD, marker_color=SLATE,
            text=[f"{v:+.1f}" for v in TD], textposition="outside",
            textfont=dict(size=13, family=FONT, color="black"), cliponaxis=False)
fig.add_bar(name="Self-evolving (ours)", x=XCATS, y=SI, marker_color=TEAL,
            text=[f"{v:+.1f}" for v in SI], textposition="outside",
            textfont=dict(size=13, family=FONT, color="black"), cliponaxis=False)
fig.add_vline(x=7.5, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.35)")
fig.update_layout(
    height=430, width=1000, barmode="group", bargap=0.26, bargroupgap=0.06,
    font=dict(family=FONT, color="black", size=17),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=18, b=58, l=88, r=18),
    legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0.01,
                font=dict(size=16, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
fig.update_xaxes(showgrid=False, tickfont=dict(family=FONT, color="black", size=15),
                 linecolor="black", tickcolor="black")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID,
                 range=[min(min(SI + TD) - 1.3, -0.8), max(SI + TD) + 1.5],
                 title_text="Δ accuracy vs. start (pp)",
                 title_font=dict(family=FONT, size=17, color="black"),
                 tickfont=dict(family=FONT, color="black", size=15), linecolor="black",
                 zeroline=True, zerolinewidth=2, zerolinecolor="rgba(0,0,0,0.55)",
                 tickformat="+.0f")
save(fig, "fig_delta_selfevolving_vs_trainset")


# ---------- Figure 2: self-evolving vs external teachers, both benchmarks ----------
KIND_COLOR = {"self": TEAL, "teacher": SLATE, "base": PINK, "frontier": GOLD}
KIND_NAME = {"self": "Self-evolving (same model proposes and solves)",
             "teacher": "Distilled from a stronger external teacher",
             "base": "Untrained base model",
             "frontier": "Frontier reference"}
ROWS = [
    ("Base", "base", "qwen36_27b_base", "Qwen_Qwen3.6-27B"),
    ("Kimi-K2.6", "teacher", "qwen36_27b_full_kimi", "qwen36_27b_full_kimi"),
    ("DeepSeek-V4-Pro", "teacher", "qwen36_27b_deepseekv4pro", "qwen36_27b_deepseekv4pro"),
    ("GPT-5.5", "teacher", "qwen36_27b_full_gpt55", "qwen36_27b_full_gpt55"),
    ("Qwen3.5-397B", "teacher", "qwen36_27b_full_opd_qwen397b_v2",
     "qwen36_27b_full_opd_qwen397b_v2"),
    ("Self-evolving", "self", "qwen36_27b_selfimprove__", "qwen36_27b_selfimprove__"),
    ("GPT-5.3", "frontier", "gpt-5.3-chat", "gpt-5.3-chat"),
]

best_mimic = lambda p: max(r["overall"] for r in MIMIC["baselines"] + MIMIC["trained_all"]
                           if r["model"].startswith(p))
best_hb = lambda p: max(r["overall_score_length_adjusted"] for r in HB["runs"]
                        if r["label"].startswith(p))

labels = [r[0] for r in ROWS]
panels = [(1, [best_mimic(r[2]) for r in ROWS]), (2, [best_hb(r[3]) for r in ROWS])]

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                    subplot_titles=("MIMIC-IV rare disease (accuracy)",
                                    "HealthBench Professional (length-adjusted)"))
for col, vals in panels:
    for kind in ["base", "teacher", "self", "frontier"]:
        ys = [v if r[1] == kind else None for v, r in zip(vals, ROWS)]
        fig.add_bar(x=labels, y=ys, name=KIND_NAME[kind], marker_color=KIND_COLOR[kind],
                    legendgroup=kind, showlegend=(col == 1),
                    text=[f"{v:.3f}" if v is not None else "" for v in ys],
                    textposition="outside", cliponaxis=False,
                    textfont=dict(size=14, family=FONT, color="black"), row=1, col=col)
fig.update_layout(
    height=500, width=1000, barmode="stack", bargap=0.3,
    font=dict(family=FONT, color="black", size=17),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=42, b=132, l=66, r=18),
    legend=dict(orientation="h", yanchor="top", y=-0.42, xanchor="center", x=0.5,
                font=dict(size=15), traceorder="normal"))
fig.update_yaxes(range=[0, 0.45], gridcolor=GRID, zeroline=False, title_text="Accuracy",
                 tickfont=dict(size=15), row=1, col=1)
fig.update_yaxes(range=[0, 0.47], gridcolor=GRID, zeroline=False, title_text="Score",
                 tickfont=dict(size=15), row=1, col=2)
fig.update_xaxes(tickangle=-28, showgrid=False, linecolor="black", ticks="outside",
                 tickfont=dict(size=15))
for a in fig.layout.annotations:
    a.font = dict(family=FONT, size=17, color="black")
save(fig, "fig_teacher_comparison")

# ---------- Appendix figure: overall ranking with paper-facing model names ----------
NAME = [  # (internal prefix, display name, kind); first match wins, order = display order
    ("gpt-5.3-chat", "GPT-5.3", "frontier"),
    ("qwen397_a17b_fp8", "Qwen3.5-397B", "frontier"),
    ("qwen36_27b_base", "Qwen3.6-27B", "base"),
    ("qwen35_9b_base", "Qwen3.5-9B", "base"),
    ("gemma4_31b_base", "Gemma-4-31B", "base"),
    ("qwen36_27b_selfimprove_sft", "Qwen3.6-27B + self-evolving (supervised only)", "self"),
    ("qwen36_27b_selfimprove", "Qwen3.6-27B + self-evolving", "self"),
    ("qwen35_9b_selfimprove", "Qwen3.5-9B + self-evolving", "self"),
    ("gemma4_31b_selfimprove", "Gemma-4-31B + self-evolving", "self"),
    ("qwen36_27b_full_opd_qwen397b", "Qwen3.6-27B + 397B distill", "teacher"),
    ("qwen36_27b_full_gpt55", "Qwen3.6-27B + GPT-5.5 distill", "teacher"),
    ("qwen36_27b_deepseekv4pro", "Qwen3.6-27B + DeepSeek-V4-Pro distill", "teacher"),
    ("qwen36_27b_full_kimi", "Qwen3.6-27B + Kimi-K2.6 distill", "teacher"),
    ("qwen35_9b_full_gpt55", "Qwen3.5-9B + GPT-5.5 distill", "teacher"),
    ("qwen36_27b_sft_long", "Qwen3.6-27B + warm start only", "warm"),
    ("qwen35_9b_sft_long", "Qwen3.5-9B + warm start only", "warm"),
    ("lingshu_32b", "Lingshu-32B (medical)", "medical"),
    ("lingshu_7b", "Lingshu-7B (medical)", "medical"),
    ("medgemma_1.5_4b", "MedGemma-1.5-4B (medical)", "medical"),
]
RANK_COLOR = {"self": TEAL, "teacher": SLATE, "base": PINK, "frontier": GOLD,
              "warm": "#b9c6db", "medical": "#4f9e86"}
RANK_LABEL = {"self": "Self-evolving (ours)", "teacher": "Distilled from an external teacher",
              "base": "Untrained base model", "frontier": "Frontier / large reference",
              "warm": "Supervised warm start only", "medical": "Open medical model"}

rows = []
for prefix, disp, kind in NAME:
    cand = [r for r in MIMIC["baselines"] + MIMIC["trained_all"]
            if r["model"].startswith(prefix)]
    if cand:
        rows.append((disp, kind, max(r["overall"] for r in cand)))
rows.sort(key=lambda t: t[2])

fig = go.Figure()
for kind in ["base", "warm", "medical", "teacher", "frontier", "self"]:
    xs = [v if k == kind else None for _, k, v in rows]
    fig.add_bar(orientation="h", y=[d for d, _, _ in rows], x=xs, name=RANK_LABEL[kind],
                marker_color=RANK_COLOR[kind],
                text=[f"{v:.3f}" if k == kind else "" for _, k, v in rows],
                textposition="outside", cliponaxis=False,
                textfont=dict(size=13, family=FONT, color="black"))
fig.update_layout(
    height=620, width=1000, barmode="stack",
    font=dict(family=FONT, color="black", size=16),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=54, b=54, l=300, r=52),
    legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1,
                font=dict(size=13, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, 0.40],
                 title_text="Overall accuracy", tickfont=dict(size=14), linecolor="black")
fig.update_yaxes(tickfont=dict(family=FONT, color="black", size=14), linecolor="black",
                 ticksuffix="  ")
save(fig, "fig_mimic_ranking")


# ---------- Appendix figure: base vs self-evolving per category, three families ----------
FAMS = [("Qwen3.6-27B", "qwen36_27b_base", "qwen36_27b_selfimprove__"),
        ("Qwen3.5-9B", "qwen35_9b_base", "qwen35_9b_selfimprove"),
        ("Gemma-4-31B", "gemma4_31b_base", "gemma4_31b_selfimprove")]
byname = {r["model"]: r for r in MIMIC["baselines"] + MIMIC["trained_all"]}


def best_rec(prefix):
    return max((r for r in byname.values() if r["model"].startswith(prefix)),
               key=lambda r: r["overall"])


fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.05,
                    subplot_titles=[f for f, _, _ in FAMS])
for i, (fam, bp, sp) in enumerate(FAMS):
    for nm, prefix, color in [("Base", bp, PINK), ("Self-evolving", sp, TEAL)]:
        r = best_rec(prefix)
        fig.add_trace(go.Bar(x=[CATNAME[c] for c in CATS],
                             y=[r["per_cat"].get(c) for c in CATS],
                             name=nm, marker_color=color, showlegend=(i == 0)),
                      row=1, col=i + 1)
fig.update_layout(
    height=430, width=1000, barmode="group", bargap=0.26, bargroupgap=0.05,
    font=dict(family=FONT, color="black", size=16),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=76, b=104, l=68, r=18),
    legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5,
                font=dict(size=15, family=FONT, color="black")))
fig.update_xaxes(showgrid=False, tickangle=42, tickfont=dict(family=FONT, color="black", size=12),
                 linecolor="black", tickcolor="black")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, 0.62],
                 tickfont=dict(size=13), linecolor="black")
fig.update_yaxes(title_text="Accuracy", title_font=dict(family=FONT, size=16, color="black"),
                 col=1)
for a in fig.layout.annotations:
    a.font = dict(family=FONT, size=17, color="black")
save(fig, "fig_base_vs_selfevolving")


# ---------- Appendix figures: HealthBench Professional ----------
AGG = json.load(open(VIS / "health_bench_pro" / "HEALTHBENCH_OFFICIAL_AGG.json"))
agg = {r["label"]: r for r in AGG["runs"]}

HB_FAMS = [("Qwen3.6-27B", "Qwen3.6-27B", "qwen36_27b_selfimprove__step20"),
           ("Gemma-4-31B", "gemma-4-31B-it", "gemma4_31b_selfimprove_split__step360"),
           ("Qwen3.5-9B", "Qwen3.5-9B", "qwen35_9b_selfimprove__step150")]

# (a) base vs self-evolving per family, both metrics, with GPT-5.3 as a reference pair
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                    subplot_titles=("Length-adjusted", "Raw"))
xs = [f for f, _, _ in HB_FAMS] + ["GPT-5.3"]
for col, metric in ((1, "length_adjusted"), (2, "raw")):
    base_y = [agg[b]["overall"][metric] for _, b, _ in HB_FAMS] + [None]
    self_y = [agg[s]["overall"][metric] for _, _, s in HB_FAMS] + [None]
    ref_y = [None] * len(HB_FAMS) + [agg["gpt-5.3-chat"]["overall"][metric]]
    for nm, ys, color in [("Base", base_y, PINK), ("Self-evolving", self_y, TEAL),
                          ("Frontier reference", ref_y, GOLD)]:
        fig.add_bar(x=xs, y=ys, name=nm, marker_color=color, legendgroup=nm,
                    showlegend=(col == 1),
                    text=[f"{v:.3f}" if v is not None else "" for v in ys],
                    textposition="outside", cliponaxis=False,
                    textfont=dict(size=13, family=FONT, color="black"), row=1, col=col)
fig.update_layout(
    height=420, width=1000, barmode="group", bargap=0.3, bargroupgap=0.05,
    font=dict(family=FONT, color="black", size=16),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=44, b=76, l=66, r=18),
    legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5,
                font=dict(size=15)))
fig.update_yaxes(range=[0, 0.47], gridcolor=GRID, title_text="Score", tickfont=dict(size=14))
fig.update_xaxes(showgrid=False, linecolor="black", tickfont=dict(size=15))
for a in fig.layout.annotations:
    a.font = dict(family=FONT, size=17, color="black")
save(fig, "fig_hb_by_family")

# (b) by dataset slice
SLICES = [("good_faith__typical", "Good faith, typical"),
          ("good_faith__difficult", "Good faith, difficult"),
          ("red_teaming__difficult", "Red teaming, difficult")]
SLICE_SERIES = [  # (display, results key, colour) --- the two families of Table 4, plus GPT-5.3
    ("Qwen3.6-27B base", "Qwen3.6-27B", PINK),
    ("Qwen3.6-27B self-evolving", "qwen36_27b_selfimprove__step20", TEAL),
    ("Gemma-4-31B base", "gemma-4-31B-it", "#e8a9d2"),
    ("Gemma-4-31B self-evolving", "gemma4_31b_selfimprove_split__step360", "#4f9e86"),
    ("GPT-5.3", "gpt-5.3-chat", GOLD),
]
fig = go.Figure()
for disp, key, color in SLICE_SERIES:
    ys = [agg[key]["by_slice"][s]["length_adjusted"] for s, _ in SLICES]
    fig.add_bar(x=[lbl for _, lbl in SLICES], y=ys, name=disp, marker_color=color,
                text=[f"{v:.2f}" for v in ys], textposition="outside", cliponaxis=False,
                textfont=dict(size=12, family=FONT, color="black"))
fig.update_layout(
    height=440, width=1000, barmode="group", bargap=0.28,
    font=dict(family=FONT, color="black", size=16),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=20, b=96, l=68, r=18),
    legend=dict(orientation="h", yanchor="bottom", y=-0.34, xanchor="center", x=0.5,
                font=dict(size=15)))
fig.update_yaxes(range=[0, 0.72], gridcolor=GRID, title_text="Length-adjusted score",
                 tickfont=dict(size=14))
fig.update_xaxes(showgrid=False, linecolor="black", tickfont=dict(size=13))
save(fig, "fig_hb_by_slice")

# These comparisons now use pinned official grades and best-checkpoint archives.
refresh_stage1_comparisons()


print("\nself-evolving :", [f"{v:+.2f}" for v in SI])
print("train-set RL  :", [f"{v:+.2f}" for v in TD])
for lab, m, h in zip(labels, panels[0][1], panels[1][1]):
    print(f"  {lab:18s} mimic={m:.4f}  hb_la={h:.4f}")
