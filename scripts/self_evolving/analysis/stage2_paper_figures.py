#!/usr/bin/env python3
"""Render the stage-2 paper figures from paper_data/stage2/stage2_table.json.

fig_main.pdf  (1x2)
  (a) overall HealthBench Professional accuracy per training setting, three bars
      per setting (untrained, fixed prompt, SER)
  (b) accuracy by category and difficulty for the 9B GPT-graded setting,
      fixed prompt vs SER

Style follows visualizations/paper/plot_paper_figures.py (stage-1 figures):
Computer Modern serif, white surface, slate/teal palette, value labels outside.
Run from the repo root; writes into paper_stage2/figures/.
"""
import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = "paper_data/stage2/stage2_table.json"
OUT = "paper_stage2/figures"
PRIM = "acc_len_adj_signed"

FONT = "Computer Modern, CMU Serif, serif"
GRID = "rgba(128,128,128,0.2)"
TEAL, SLATE, GREY = "#7dbfa7", "#6c8ebf", "#b9b9b9"

table = json.load(open(DATA))
blocks = table["blocks"]

SETTING = ["9B<br>GPT grader", "27B<br>GPT grader", "9B<br>all roles 9B",
           "9B, 9B grader<br>no retrieval"]


def row(b, label_part):
    return next(r for r in b["rows"] if label_part in r["label"])


def overall(label_part):
    return [row(b, label_part)["overall"][PRIM] for b in blocks]


fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                    column_widths=[0.5, 0.5])

series_a = [("Untrained", "Untrained", GREY),
            ("Fixed prompt", "Fixed prompt", SLATE),
            ("SER (ours)", "SER", TEAL)]
for name, part, color in series_a:
    ys = overall(part)
    fig.add_bar(name=name, x=SETTING, y=ys, marker_color=color, legendgroup=name,
                text=[f"{v:.2f}" for v in ys], textposition="outside", cliponaxis=False,
                textfont=dict(size=12, family=FONT, color="black"), row=1, col=1)

nine_b = blocks[0]
SLICES = [("category", "consult", "Consult"), ("category", "research", "Research"),
          ("category", "writing", "Writing"), ("difficulty", "typical", "Typical"),
          ("difficulty", "difficult", "Difficult")]
XB = [s[2] for s in SLICES]
for name, part, color in series_a[1:]:
    r = row(nine_b, part)
    ys = [r[kind][key][PRIM] for kind, key, _ in SLICES]
    fig.add_bar(name=name, x=XB, y=ys, marker_color=color, legendgroup=name,
                showlegend=False,
                text=[f"{v:.2f}" for v in ys], textposition="outside", cliponaxis=False,
                textfont=dict(size=12, family=FONT, color="black"), row=1, col=2)
fig.add_vline(x=2.5, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.35)",
              row=1, col=2)

fig.update_layout(
    height=390, width=1180, barmode="group", bargap=0.28, bargroupgap=0.06,
    font=dict(family=FONT, color="black", size=16),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=16, b=76, l=70, r=14),
    legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0.005,
                font=dict(size=15, family=FONT, color="black"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
fig.update_xaxes(showgrid=False, tickfont=dict(family=FONT, color="black", size=13.5),
                 linecolor="black", tickcolor="black")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, 0.74],
                 tickfont=dict(family=FONT, color="black", size=13.5),
                 linecolor="black", zeroline=True, zerolinewidth=1.5,
                 zerolinecolor="rgba(0,0,0,0.55)")
fig.update_yaxes(title_text="HealthBench Professional accuracy",
                 title_font=dict(family=FONT, size=15.5, color="black"), row=1, col=1)
for i, tag in enumerate(["(a)", "(b)"]):
    fig.add_annotation(text=tag, xref="paper", yref="paper",
                       x=0.235 + 0.55 * i, y=-0.24, showarrow=False,
                       font=dict(size=16, family=FONT, color="black"))

os.makedirs(OUT, exist_ok=True)
fig.write_image(os.path.join(OUT, "fig_main.pdf"))
fig.write_image(os.path.join(OUT, "fig_main.png"), scale=2)
print("wrote", os.path.join(OUT, "fig_main.pdf"))
