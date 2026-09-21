#!/usr/bin/env python3
"""Figure: anatomy of the hack-then-patch loop over training, from the ledger JSONs
written by stage2_adversary_ledger.py (paper_data/stage2/ledger/<run>.json).

fig_anatomy.pdf (1x3)
  (a) cumulative accepted criteria over training steps for the 9B and 27B settings,
      with validation accuracy on a second axis (9B setting)
  (b) served-rollout score on patched tasks, before vs after the patch (per task,
      9B setting): the patched rubric stops paying the exploit
  (c) distribution of the normalized margin each accepted criterion removes (9B + 27B)

Style follows stage2_paper_figures.py (Computer Modern, white, teal/slate/grey).
Run from the repo root; writes into paper_stage2/figures/.
"""
import json
import os

import plotly.io as _pio
_pio.kaleido.scope.mathjax = None  # no "Loading [MathJax]" box in the exported PDF
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LEDGER = "paper_data/stage2/ledger"
OUT = "paper_stage2/figures"
FONT = "Computer Modern, CMU Serif, serif"
GRID = "rgba(128,128,128,0.2)"
TEAL, SLATE, GREY, DARK = "#7dbfa7", "#6c8ebf", "#b9b9b9", "#444444"

R9 = json.load(open(os.path.join(LEDGER, "hb9b_specgap_ship_retrieval_websearch.json")))
R27 = json.load(open(os.path.join(LEDGER, "hb27b_specgap_ship_retrieval_websearch.json")))


def cumulative(run):
    rows = sorted(run["series"]["per_step"], key=lambda r: r["step"])
    xs, ys, tot = [], [], 0
    for r in rows:
        tot += r.get("criteria_accepted", 0)
        xs.append(r["step"]); ys.append(tot)
    return xs, ys


def val(run):
    rows = [r for r in run["series"]["per_step"] if r.get("val_acc_len_adj_signed") is not None]
    rows.sort(key=lambda r: r["step"])
    return [r["step"] for r in rows], [r["val_acc_len_adj_signed"] for r in rows]


fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.09,
                    specs=[[{"secondary_y": True}, {}, {}]])

# (a)
for run, name, color in ((R9, "9B setting", TEAL), (R27, "27B setting", SLATE)):
    xs, ys = cumulative(run)
    fig.add_scatter(x=xs, y=ys, mode="lines", name=f"accepted criteria, {name}",
                    line=dict(color=color, width=2.2), row=1, col=1, secondary_y=False)
vx, vy = val(R9)
fig.add_scatter(x=vx, y=vy, mode="lines", name="validation accuracy, 9B setting",
                line=dict(color=DARK, width=1.4, dash="dot"), row=1, col=1, secondary_y=True)

# (b)
pt = R9["patch_effect"]["per_task"]
before = [r["mean_before"] for r in pt]
after = [r["mean_after"] for r in pt]
fig.add_scatter(x=before, y=after, mode="markers", name="patched task (9B setting)",
                marker=dict(color=TEAL, size=6, opacity=0.65, line=dict(color=DARK, width=0.4)),
                row=1, col=2)
fig.add_scatter(x=[0, 1.05], y=[0, 1.05], mode="lines", showlegend=False,
                line=dict(color=GREY, width=1, dash="dash"), row=1, col=2)

# (c)
def per_item_gap_drops(path):
    d = json.load(open(path))
    return [x["gap_drop"] for x in d["funnel"].get("gap_drops_per_item", []) if x.get("gap_drop") is not None]


gd9 = per_item_gap_drops(os.path.join(LEDGER, "hb9b_specgap_ship_retrieval_websearch.json"))
gd27 = per_item_gap_drops(os.path.join(LEDGER, "hb27b_specgap_ship_retrieval_websearch.json"))
for gd, name, color in ((gd9, "9B setting", TEAL), (gd27, "27B setting", SLATE)):
    if gd:
        fig.add_histogram(x=gd, name=f"margin removed, {name}", histnorm="probability",
                          xbins=dict(start=0, end=1.0, size=0.05), marker_color=color, opacity=0.6,
                          row=1, col=3)

fig.update_layout(barmode="overlay", template="plotly_white", width=1500, height=430,
                  font=dict(family=FONT, size=15, color="black"),
                  legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                              font=dict(size=13)),
                  margin=dict(l=60, r=30, t=70, b=60), paper_bgcolor="white", plot_bgcolor="white")
fig.update_xaxes(title_text="(a) training step", row=1, col=1, gridcolor=GRID)
fig.update_yaxes(title_text="accepted criteria (cumulative)", row=1, col=1, gridcolor=GRID, secondary_y=False)
fig.update_yaxes(title_text="validation accuracy", row=1, col=1, secondary_y=True, showgrid=False, range=[0.3, 0.6])
fig.update_xaxes(title_text="(b) served score before patch", row=1, col=2, gridcolor=GRID, range=[0, 1.05])
fig.update_yaxes(title_text="served score after patch", row=1, col=2, gridcolor=GRID, range=[0, 1.05])
fig.update_xaxes(title_text="(c) normalized margin removed per criterion", row=1, col=3, gridcolor=GRID)
fig.update_yaxes(title_text="share of accepted criteria", row=1, col=3, gridcolor=GRID)

os.makedirs(OUT, exist_ok=True)
fig.write_image(os.path.join(OUT, "fig_anatomy.pdf"))
fig.write_image(os.path.join(OUT, "fig_anatomy.png"), scale=2)
print("wrote", os.path.join(OUT, "fig_anatomy.pdf"))
