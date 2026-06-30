"""MIMIC-IV rare-disease — per-category accuracy DELTA of training over the base
model, for self-improve vs. train-data-only (the mimiciv_rare_qwen36_27b_baseline
control), in the lab-paper style.

Both runs start from the SAME Qwen3.6-27B base and were judged identically:
gpt-chat-latest_2026-05-28, STANDARD lenient rubric, on each run's own
training-time validation generations (verl validation_data_dir dumps, 4096-tok
rollout, thinking-off). Delta = (best-epoch-overall checkpoint) - (that run's
step-0 base eval), per ICD category and overall. Using each run's own step-0 as
the reference isolates the training effect and controls for run-to-run rollout
variance. Self-improve lifts most categories; train-data-only barely moves
overall and trades some categories down for others.

Reads DELTA_DATA.json (written by assemble step from the val-trace rejudge).
Same palette/typography as plot_mimic_rare.py. Outputs {pdf,png,html}.
"""
import json
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "DELTA_DATA.json"))

FONT = "Computer Modern, CMU Serif, serif"
GRID = "rgba(128,128,128,0.2)"
TEAL, PINK, GOLD, SLATE = "#7dbfa7", "#da81c1", "#e0b45c", "#6c8ebf"

CATS = ["neoplasms", "nervous_system", "blood_immune", "circulatory",
        "digestive", "endocrine_metabolic", "infectious", "other"]
CATNAME = {"neoplasms": "Neoplasms", "nervous_system": "Nervous", "blood_immune": "Blood/Imm",
           "circulatory": "Circulatory", "digestive": "Digestive",
           "endocrine_metabolic": "Endocrine", "infectious": "Infectious", "other": "Other"}

XCATS = [CATNAME[c] for c in CATS] + ["Overall"]


def deltas(run):
    base, best = D[run]["base"], D[run]["best"]
    d = [best["per_cat"][c] - base["per_cat"][c] for c in CATS]
    d.append(best["overall"] - base["overall"])
    return [100.0 * v for v in d]  # percentage points


SI = deltas("self_improve")
TD = deltas("train_only")

# y-range: tight & asymmetric to the data (nearly all deltas are small positives,
# so a symmetric range would waste the lower half). Leave headroom for the
# outside value labels above/below each bar.
dmin, dmax = min(SI + TD), max(SI + TD)
lo = min(dmin - 1.3, -0.8)   # always show a little below zero so the axis reads as signed
hi = dmax + 1.4

fig = go.Figure()
# "train data only" first (drawn behind in the group), then "self improve"
fig.add_bar(name="train data only", x=XCATS, y=TD, marker_color=SLATE,
            text=[f"{v:+.1f}" for v in TD], textposition="outside",
            textfont=dict(size=11, family=FONT, color="black"), cliponaxis=False)
fig.add_bar(name="self improve", x=XCATS, y=SI, marker_color=TEAL,
            text=[f"{v:+.1f}" for v in SI], textposition="outside",
            textfont=dict(size=11, family=FONT, color="black"), cliponaxis=False)

# visually separate the "Overall" summary column from the per-category ones
fig.add_vline(x=7.5, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.35)")

fig.update_layout(
    height=620, width=1500, barmode="group", bargap=0.28, bargroupgap=0.06,
    font=dict(family=FONT, color="black", size=16),
    plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=110, b=70, l=90, r=40),
    title=dict(text="MIMIC-IV rare-disease — accuracy gain over base model, by ICD category"
                    "<br><span style='font-size:14px'>Δ lenient accuracy in percentage points "
                    "(best epoch − base), gpt-chat-latest judge, n=2452</span>",
               font=dict(family=FONT, size=20, color="black"), x=0.5, xanchor="center"),
    legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.012,
                font=dict(size=15, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
fig.update_xaxes(showgrid=False, tickfont=dict(family=FONT, color="black", size=14),
                 linecolor="black", tickcolor="black")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[lo, hi],
                 title_text="Δ accuracy vs. base (pp)", title_font=dict(family=FONT, size=17, color="black"),
                 tickfont=dict(family=FONT, color="black", size=14), linecolor="black",
                 zeroline=True, zerolinewidth=2, zerolinecolor="rgba(0,0,0,0.55)",
                 tickformat="+.0f")

fig.write_image(str(HERE / "fig5_delta_selfimprove_vs_trainonly.pdf"))
fig.write_image(str(HERE / "fig5_delta_selfimprove_vs_trainonly.png"), scale=2)
fig.write_html(str(HERE / "fig5_delta_selfimprove_vs_trainonly.html"), include_plotlyjs="cdn")
print("wrote fig5_delta_selfimprove_vs_trainonly")
print("self improve  :", [f"{v:+.3f}" for v in SI])
print("train data only:", [f"{v:+.3f}" for v in TD])
