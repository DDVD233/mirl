"""MIMIC-IV rare-disease figures (gpt-chat-latest judge), lab paper style.

Reads MIMIC_RARE_RESULTS.json (baselines + best-per-run trained checkpoints,
overall + per-ICD-category accuracy). Writes wide figures:
  fig1_overall_ranking          all models ranked by overall acc (colored by kind)
  fig2_by_category              per-ICD-category acc, curated cross-family models
  fig3_base_vs_selfimprove      base vs self-improve, per category, 3 family panels
  fig4_category_heatmap         category x model accuracy heatmap (wide)

Same style: palette #7dbfa7/#da81c1/#e0b45c/#6c8ebf, Computer Modern, white bg,
black text, light grids. Outputs {pdf,png,html}.
"""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "MIMIC_RARE_RESULTS.json"))

FONT = "Computer Modern, CMU Serif, serif"
GRID = "rgba(128,128,128,0.2)"
TEAL, PINK, GOLD, SLATE = "#7dbfa7", "#da81c1", "#e0b45c", "#6c8ebf"
KIND_COLOR = {"trained": TEAL, "base": PINK, "teacher": GOLD, "medical": SLATE}
KIND_NAME = {"trained": "Self-improve / RL-trained", "base": "Base model",
             "teacher": "Frontier reference (teacher)", "medical": "Open medical MLLM"}

CATS = ["neoplasms", "nervous_system", "blood_immune", "circulatory",
        "digestive", "endocrine_metabolic", "infectious", "other"]
CATNAME = {"neoplasms": "Neoplasms", "nervous_system": "Nervous", "blood_immune": "Blood/Imm",
           "circulatory": "Circulatory", "digestive": "Digestive",
           "endocrine_metabolic": "Endocrine", "infectious": "Infectious", "other": "Other"}

BL = D["baselines"]
BEST = D["trained_best"]
ALL = BL + BEST
BY = {}  # model -> record
for r in ALL:
    BY[r["model"]] = r


def step_of(r):
    import re
    m = re.search(r"__step(\d+)", r["model"])
    return m.group(1) if m else None


def label_of(r):
    if r["kind"] == "trained":
        run = r.get("run") or r["model"].split("__step")[0]
        s = step_of(r)
        return f"{run}·s{s}" if s else run
    return r["model"]


def save(fig, stem):
    fig.write_image(str(HERE / f"{stem}.pdf"))
    fig.write_image(str(HERE / f"{stem}.png"), scale=2)
    fig.write_html(str(HERE / f"{stem}.html"), include_plotlyjs="cdn")
    print(f"wrote {stem}")


# ---------- Fig 1: overall ranking, all models, colored by kind ----------
rows = sorted(ALL, key=lambda r: r["overall"])  # ascending -> top of horizontal bar = best
# fig1 labels drop the ·step suffix (each run appears once here)
labels = [(r.get("run") or r["model"].split("__step")[0]) if r["kind"] == "trained" else r["model"]
          for r in rows]
fig = go.Figure()
for kind in ["base", "teacher", "medical", "trained"]:
    xs = [r["overall"] if r["kind"] == kind else None for r in rows]
    fig.add_bar(orientation="h", y=labels, x=xs, name=KIND_NAME[kind],
                marker_color=KIND_COLOR[kind],
                text=[f"{r['overall']:.3f}" if r["kind"] == kind else "" for r in rows],
                textposition="outside", textfont=dict(size=12, family=FONT, color="black"),
                cliponaxis=False)
fig.update_layout(
    height=800, width=1180, barmode="stack",
    font=dict(family=FONT, color="black", size=15),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=130, b=60, l=300, r=70),
    title=dict(text="MIMIC-IV rare-disease — overall accuracy (gpt-chat-latest judge, n=2452)",
               font=dict(family=FONT, size=19, color="black"), x=0.5, xanchor="center",
               y=0.985, yanchor="top"),
    legend=dict(orientation="h", yanchor="bottom", y=1.012, xanchor="right", x=1,
                font=dict(size=13, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, 0.40],
                 title_text="overall accuracy", title_font=dict(family=FONT, size=16, color="black"),
                 tickfont=dict(family=FONT, color="black", size=13), linecolor="black", zeroline=True,
                 zerolinecolor="rgba(0,0,0,0.35)")
fig.update_yaxes(tickfont=dict(family=FONT, color="black", size=12.5), linecolor="black",
                 ticksuffix="  ")
save(fig, "fig1_overall_ranking")


# ---------- Fig 2: per-category, curated cross-family models ----------
CURATED = [
    ("qwen36_27b_selfimprove__step40", "Qwen3.6-27B Self-Improve", TEAL),
    ("qwen36_27b_base", "Qwen3.6-27B base", PINK),
    ("qwen397_a17b_fp8", "Qwen3.5-397B teacher", GOLD),
    ("gemma4_31b_base", "gemma-4-31B base", SLATE),
    ("qwen35_9b_base", "Qwen3.5-9B base", "#b06a98"),
    ("lingshu_32b", "Lingshu-32B (medical)", "#4f9e86"),
]
fig = go.Figure()
xcats = [CATNAME[c] for c in CATS]
for model, disp, color in CURATED:
    r = BY.get(model)
    if not r:
        continue
    y = [r["per_cat"].get(c) for c in CATS]
    fig.add_bar(name=disp, x=xcats, y=y, marker_color=color)
fig.update_layout(
    height=660, width=1500, barmode="group", bargap=0.22, bargroupgap=0.05,
    font=dict(family=FONT, color="black", size=16),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=140, b=70, l=80, r=40),
    title=dict(text="MIMIC-IV rare-disease — accuracy by ICD category",
               font=dict(family=FONT, size=20, color="black"), x=0.5, xanchor="center",
               y=0.98, yanchor="top"),
    legend=dict(orientation="h", yanchor="bottom", y=1.012, xanchor="center", x=0.5,
                font=dict(size=14, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
fig.update_xaxes(showgrid=False, tickfont=dict(family=FONT, color="black", size=15),
                 linecolor="black", tickcolor="black")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, 0.65],
                 title_text="accuracy", title_font=dict(family=FONT, size=18, color="black"),
                 tickfont=dict(family=FONT, color="black", size=15), linecolor="black",
                 zeroline=True, zerolinecolor="rgba(0,0,0,0.35)")
save(fig, "fig2_by_category")


# ---------- Fig 3: base vs self-improve, per category, 3 family panels ----------
FAMS = [
    ("Qwen3.6-27B", "qwen36_27b_base", "qwen36_27b_selfimprove__step40"),
    ("Qwen3.5-9B", "qwen35_9b_base", "qwen35_9b_selfimprove__step200"),
    ("gemma-4-31B", "gemma4_31b_base", "gemma4_31b_selfimprove_split__step340"),
]
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{f} (base → Self-Improve)" for f, _, _ in FAMS],
                    horizontal_spacing=0.055)
for i, (fam, base_m, si_m) in enumerate(FAMS):
    for name, model, color in [("Base", base_m, PINK), ("Self-Improve", si_m, TEAL)]:
        r = BY.get(model)
        y = [r["per_cat"].get(c) for c in CATS] if r else [None] * len(CATS)
        fig.add_trace(go.Bar(x=[CATNAME[c] for c in CATS], y=y, name=name, marker_color=color,
                             showlegend=(i == 0)), row=1, col=i + 1)
fig.update_layout(
    height=620, width=1600, barmode="group", bargap=0.28, bargroupgap=0.06,
    font=dict(family=FONT, color="black", size=15),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=150, b=110, l=70, r=30),
    title=dict(text="Base vs Self-Improve by ICD category", x=0.5, xanchor="center",
               font=dict(family=FONT, size=20, color="black"), y=0.985, yanchor="top"),
    legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5,
                font=dict(size=16, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
fig.update_xaxes(showgrid=False, tickangle=30, tickfont=dict(family=FONT, color="black", size=12.5),
                 linecolor="black", tickcolor="black")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, 0.62],
                 tickfont=dict(family=FONT, color="black", size=13), linecolor="black",
                 zeroline=True, zerolinecolor="rgba(0,0,0,0.35)")
fig.update_yaxes(title_text="accuracy", title_font=dict(family=FONT, size=16, color="black"), col=1)
for a in fig["layout"]["annotations"]:
    a["font"] = dict(size=15, family=FONT, color="black")
save(fig, "fig3_base_vs_selfimprove")


# ---------- Fig 4: category x model heatmap (wide) ----------
hm = sorted(ALL, key=lambda r: -r["overall"])
xlab = [label_of(r) for r in hm]
ycats = ["Overall"] + [CATNAME[c] for c in CATS]
Z = []
Z.append([r["overall"] for r in hm])
for c in CATS:
    Z.append([r["per_cat"].get(c) for r in hm])
text = [[f"{v:.2f}" if v is not None else "" for v in row] for row in Z]
fig = go.Figure(go.Heatmap(
    z=Z, x=xlab, y=ycats, text=text, texttemplate="%{text}",
    textfont=dict(size=11, family=FONT), colorscale="Tealgrn", zmin=0, zmax=0.6,
    colorbar=dict(title=dict(text="accuracy", font=dict(family=FONT, size=14)),
                  tickfont=dict(family=FONT, size=12), outlinecolor="black", outlinewidth=1)))
fig.update_layout(
    height=520, width=1650, font=dict(family=FONT, color="black", size=14),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=70, b=210, l=110, r=20),
    title=dict(text="MIMIC-IV rare-disease — accuracy heatmap (overall + by ICD category)",
               x=0.5, xanchor="center", font=dict(family=FONT, size=19, color="black")))
fig.update_xaxes(tickangle=40, tickfont=dict(family=FONT, color="black", size=12), side="bottom")
fig.update_yaxes(tickfont=dict(family=FONT, color="black", size=14), autorange="reversed")
save(fig, "fig4_category_heatmap")

print("done")
