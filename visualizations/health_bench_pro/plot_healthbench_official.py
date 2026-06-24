"""Paper-style HealthBench Professional figures using the OFFICIAL aggregations
(use case / dataset slice / specialty), length-adjusted, for the base models.

Mirrors the paper's Fig 4 (use case), Fig 5 (dataset slice), Fig 6 (specialty).
Reads HEALTHBENCH_OFFICIAL_AGG.json; writes:
  fig4_by_use_case.{pdf,png}
  fig5_by_dataset_slice.{pdf,png}
  fig6_by_specialty.{pdf,png}
Same lab style: palette #7dbfa7 / #da81c1 / #e0b45c (+ slate), Computer Modern,
white background, black text, light grids. Length-adjusted score (paper primary).
"""

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "HEALTHBENCH_OFFICIAL_AGG.json"))
BASE = {r["label"]: r for r in D["runs"] if r["kind"] == "base"}
CKPT = {r["label"]: r for r in D["runs"] if r["kind"] == "checkpoint"}

FONT = "Computer Modern, CMU Serif, serif"
GRID = "rgba(128,128,128,0.2)"
TEAL, PINK, GOLD, SLATE = "#7dbfa7", "#da81c1", "#e0b45c", "#6c8ebf"
MET = "length_adjusted"  # paper primary metric

def best_variant(variant):
    """Best step (by length-adj overall) of a specific trained variant."""
    cands = [r for lbl, r in CKPT.items() if lbl.startswith(variant + "__step")]
    return max(cands, key=lambda r: r["overall"]["length_adjusted"]) if cands else None


# Our TRAINED (self-improve) models, best step per family, + gpt-5.3-chat reference.
# These REPLACE the raw base models in the by-category figures.
MODEL_RUNS = [
    ("Qwen3.5-9B-Self-Improve", best_variant("qwen35_9b_selfimprove")),
    ("Qwen3.6-27B-Self-Improve", best_variant("qwen36_27b_selfimprove")),
    ("gemma-4-31B-Self-Improve", best_variant("gemma4_31b_selfimprove_split")),
    ("gpt-5.3-chat", BASE["gpt-5.3-chat"]),
]
MODEL_NAMES = [d for d, _ in MODEL_RUNS]


def style(fig, *, height, width, ytitle, ymax=0.65, legend_title=None, tickangle=0):
    fig.update_layout(
        height=height, width=width, barmode="group", bargap=0.25, bargroupgap=0.08,
        font=dict(family=FONT, color="black", size=18),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=140, l=80, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    title=dict(text=legend_title or "", font=dict(family=FONT, size=16)),
                    font=dict(size=18, family=FONT, color="black"),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
    fig.update_xaxes(showgrid=False, tickangle=tickangle,
                     tickfont=dict(family=FONT, color="black", size=16),
                     tickcolor="black", linecolor="black")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, ymax],
                     title_text=ytitle, title_font=dict(family=FONT, color="black", size=20),
                     tickfont=dict(family=FONT, color="black", size=18),
                     tickcolor="black", linecolor="black", zeroline=True,
                     zerolinecolor="rgba(0,0,0,0.35)")
    return fig


def save(fig, stem):
    fig.write_image(str(HERE / f"{stem}.pdf"))
    fig.write_image(str(HERE / f"{stem}.png"), scale=2)
    fig.write_html(str(HERE / f"{stem}.html"), include_plotlyjs="cdn")
    print(f"wrote {stem}")


def grouped(field, cats, colors, stem, ytitle, ymax=0.65, legend_title=None):
    """cats: list of (key, display). One bar series per category; x = our trained
    models (self-improve) + gpt-5.3-chat."""
    fig = go.Figure()
    for (key, disp), color in zip(cats, colors):
        y = [run[field].get(key, {}).get(MET) for _, run in MODEL_RUNS]
        fig.add_bar(name=disp, x=MODEL_NAMES, y=y, marker_color=color,
                    text=[f"{v:.2f}" if v is not None else "" for v in y],
                    textposition="outside", textfont=dict(size=13, family=FONT, color="black"),
                    cliponaxis=False)
    style(fig, height=600, width=1080, ytitle=ytitle, ymax=ymax,
          legend_title=legend_title, tickangle=18)
    save(fig, stem)


# Fig 4 — by use case
grouped("by_use_case",
        [("consult", "Care consult"), ("writing", "Writing & doc"), ("research", "Medical research")],
        [TEAL, PINK, GOLD], "fig4_by_use_case",
        "length-adjusted score", ymax=0.7, legend_title="Use case")

# Fig 5 — by dataset slice (joint type x difficulty)
grouped("by_slice",
        [("good_faith__typical", "Good faith typical"),
         ("good_faith__difficult", "Good faith difficult"),
         ("red_teaming__difficult", "Red teaming difficult")],
        [TEAL, GOLD, PINK], "fig5_by_dataset_slice",
        "length-adjusted score", ymax=0.65, legend_title="Dataset slice")

# Fig 6 — by specialty (top-10 by count), grouped by model
g = BASE["gpt-5.3-chat"]["by_specialty"]
top = [k for k, _ in sorted(g.items(), key=lambda kv: -kv[1]["n"])[:10]]
SPNAME = {"cards": "Cardiology", "neuro": "Neurology", "obgyn": "OB/GYN",
          "psychiatry": "Psychiatry", "peds": "Pediatrics", "heme/onc": "Heme/Onc",
          "derm": "Dermatology", "ortho": "Orthopedics", "anesthesia": "Anesthesia",
          "nephro": "Nephrology", "id": "Infectious", "endo": "Endocrine"}
xlabels = [SPNAME.get(s, s) for s in top]
fig = go.Figure()
for (mdisp, run), color in zip(MODEL_RUNS, [TEAL, PINK, GOLD, SLATE]):
    y = [run["by_specialty"].get(s, {}).get(MET) for s in top]
    fig.add_bar(name=mdisp, x=xlabels, y=y, marker_color=color)
style(fig, height=600, width=1300, ytitle="length-adjusted score", ymax=0.8, legend_title="Model")
fig.update_xaxes(tickangle=35, tickfont=dict(size=15, family=FONT, color="black"))
save(fig, "fig6_by_specialty")


# ---- base vs Self-Improve (trained), per category, one panel per family ----
COMPARE_FAMS = [
    ("Qwen3.5-9B", BASE["Qwen3.5-9B"], best_variant("qwen35_9b_selfimprove")),
    ("Qwen3.6-27B", BASE["Qwen3.6-27B"], best_variant("qwen36_27b_selfimprove")),
    ("gemma-4-31B-it", BASE["gemma-4-31B-it"], best_variant("gemma4_31b_selfimprove_split")),
]


def cmp_style(fig, *, height, width, ymax, tickangle, sp_size):
    fig.update_layout(
        height=height, width=width, barmode="group", bargap=0.3, bargroupgap=0.08,
        font=dict(family=FONT, color="black", size=16),
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=100, b=100, l=75, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5,
                    font=dict(size=18, family=FONT, color="black"),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
    fig.update_xaxes(showgrid=False, tickangle=tickangle,
                     tickfont=dict(family=FONT, color="black", size=15),
                     tickcolor="black", linecolor="black")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, ymax],
                     tickfont=dict(family=FONT, color="black", size=14),
                     tickcolor="black", linecolor="black", zeroline=True,
                     zerolinecolor="rgba(0,0,0,0.35)")
    for a in fig["layout"]["annotations"]:
        a["font"] = dict(size=sp_size, family=FONT, color="black")


def compare_bt(field, cats, stem, *, ymax, rows=1, cols=3, height=540, width=1280,
               vspace=0.18, hspace=0.07, tickangle=0):
    titles = [f"{fam}  (Self-Improve)" for fam, _, tr in COMPARE_FAMS]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        vertical_spacing=vspace, horizontal_spacing=hspace)
    x = [disp for _, disp in cats]
    for i, (fam, bs, tr) in enumerate(COMPARE_FAMS):
        r, c = (i // cols + 1, i % cols + 1)
        for name, run, color in [("Base", bs, PINK), ("Self-Improve", tr, TEAL)]:
            y = [run[field].get(k, {}).get(MET) for k, _ in cats]
            fig.add_trace(go.Bar(
                x=x, y=y, name=name, marker_color=color, showlegend=(i == 0),
                text=[f"{v:.2f}" if v is not None else "" for v in y],
                textposition="outside", textfont=dict(size=11, family=FONT, color="black"),
                cliponaxis=False), row=r, col=c)
    fig.update_yaxes(title_text="length-adjusted score",
                     title_font=dict(size=15, family=FONT), col=1)
    cmp_style(fig, height=height, width=width, ymax=ymax, tickangle=tickangle, sp_size=15)
    save(fig, stem)


compare_bt("by_use_case",
           [("consult", "Care consult"), ("writing", "Writing & doc"), ("research", "Medical research")],
           "fig7_use_case_base_vs_trained", ymax=0.7)
compare_bt("by_slice",
           [("good_faith__typical", "GF typical"), ("good_faith__difficult", "GF difficult"),
            ("red_teaming__difficult", "RT difficult")],
           "fig8_slice_base_vs_trained", ymax=0.7)
compare_bt("by_specialty", [(s, SPNAME.get(s, s)) for s in top],
           "fig9_specialty_base_vs_trained", ymax=0.85, rows=3, cols=1,
           height=950, width=1050, vspace=0.13, tickangle=25)


# ---- INTERACTIVE figures: model multi-select (checkboxes) + metric toggle ----
# Shared 7-model list (gpt + base & Self-Improve per family).
ISI = {"qwen35_9b": best_variant("qwen35_9b_selfimprove"),
       "qwen36_27b": best_variant("qwen36_27b_selfimprove"),
       "gemma4_31b": best_variant("gemma4_31b_selfimprove_split")}
IMODELS = [
    ("gpt-5.3-chat", BASE["gpt-5.3-chat"]),
    ("Qwen3.5-9B (base)", BASE["Qwen3.5-9B"]),
    ("Qwen3.5-9B · Self-Improve", ISI["qwen35_9b"]),
    ("Qwen3.6-27B (base)", BASE["Qwen3.6-27B"]),
    ("Qwen3.6-27B · Self-Improve", ISI["qwen36_27b"]),
    ("gemma-4-31B (base)", BASE["gemma-4-31B-it"]),
    ("gemma-4-31B · Self-Improve", ISI["gemma4_31b"]),
]
ICOLORS = ["#e0b45c", "#e7a6cf", "#a6d8c4", "#da81c1", "#7dbfa7", "#b06a98", "#4f9e86"]

ITEMPL = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family:"Computer Modern","CMU Serif",serif; color:#000; margin:18px; background:#fff; }}
 h2 {{ font-weight:normal; font-size:20px; }}
 #modelpanel {{ margin:6px 0 12px 0; padding:8px 10px; border:1px solid #ddd; border-radius:6px; }}
 .mlbl {{ display:inline-block; margin:3px 14px 3px 0; white-space:nowrap; cursor:pointer; font-size:15px; }}
 .sw {{ display:inline-block; width:13px; height:13px; margin:0 5px; border-radius:2px; vertical-align:middle; }}
 button {{ font-family:inherit; margin-right:6px; padding:2px 9px; cursor:pointer; }}
</style></head><body>
<h2>{title} &mdash; add/remove models &amp; toggle metric</h2>
<div id="modelpanel"><b>Models:</b>&nbsp; {rows} &nbsp;
  <button onclick="setAll(true)">Select all</button>
  <button onclick="setAll(false)">Clear all</button>
</div>
{plot_div}
<script>
 function toggleModel(cb, i) {{ Plotly.restyle('hbplot', {{visible: cb.checked}}, [i]); }}
 function setAll(state) {{
   var cbs = document.querySelectorAll('.modelcb');
   cbs.forEach(function(cb){{ cb.checked = state; }});
   Plotly.restyle('hbplot', {{visible: state}}, Array.from(cbs).map(function(_, i){{ return i; }}));
 }}
</script></body></html>"""


def interactive_fig(field, cats, stem, title, default_pred, ymax=0.78, tickangle=0):
    keys = [k for k, _ in cats]
    xl = [d for _, d in cats]

    def series(run, metric):
        return [run[field].get(k, {}).get(metric) for k in keys]

    fig = go.Figure()
    for (name, run), color in zip(IMODELS, ICOLORS):
        fig.add_bar(name=name, x=xl, y=series(run, "length_adjusted"),
                    marker_color=color, visible=default_pred(name))
    la_all = [series(run, "length_adjusted") for _, run in IMODELS]
    raw_all = [series(run, "raw") for _, run in IMODELS]
    metric_menu = dict(
        type="buttons", direction="right", showactive=True, x=0.0, xanchor="left",
        y=1.17, yanchor="top", pad=dict(l=2, r=2, t=2, b=2),
        buttons=[
            dict(label="Length-adjusted", method="update",
                 args=[{"y": la_all}, {"yaxis.title.text": "length-adjusted score"}]),
            dict(label="Accuracy (raw)", method="update",
                 args=[{"y": raw_all}, {"yaxis.title.text": "accuracy (raw rubric) score"}]),
        ])
    fig.update_layout(
        height=560, width=1280, barmode="group", bargap=0.2, bargroupgap=0.05,
        font=dict(family=FONT, color="black", size=15), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(t=70, b=120, l=80, r=40),
        updatemenus=[metric_menu],
        annotations=[dict(text="Metric:", x=0.0, xref="paper", y=1.13, yref="paper",
                          showarrow=False, xanchor="left",
                          font=dict(size=14, family=FONT, color="black"))])
    fig.update_xaxes(tickangle=tickangle, showgrid=False, linecolor="black", tickcolor="black",
                     tickfont=dict(family=FONT, color="black", size=14))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID, range=[0, ymax],
                     title_text="length-adjusted score", title_font=dict(family=FONT, size=18),
                     tickfont=dict(family=FONT, color="black", size=14),
                     zeroline=True, zerolinecolor="rgba(0,0,0,0.35)", linecolor="black")
    plot_div = fig.to_html(include_plotlyjs=True, full_html=False, div_id="hbplot")
    rows = "\n".join(
        f'<label class="mlbl"><input type="checkbox" class="modelcb" {"checked" if default_pred(nm) else ""} '
        f'onchange="toggleModel(this,{i})"><span class="sw" style="background:{c}"></span>{nm}</label>'
        for i, ((nm, _), c) in enumerate(zip(IMODELS, ICOLORS)))
    (HERE / f"{stem}.html").write_text(ITEMPL.format(title=title, rows=rows, plot_div=plot_div))
    print(f"wrote {stem}.html (checkbox multi-select)")


# default-shown = what the static figure currently shows (gpt + the Self-Improve models)
_si_default = lambda nm: nm == "gpt-5.3-chat" or "Self-Improve" in nm
interactive_fig(
    "by_use_case",
    [("consult", "Care consult"), ("writing", "Writing & doc"), ("research", "Medical research")],
    "fig4_interactive", "HealthBench Professional by use case", _si_default, ymax=0.72)
interactive_fig(
    "by_slice",
    [("good_faith__typical", "Good faith typical"), ("good_faith__difficult", "Good faith difficult"),
     ("red_teaming__difficult", "Red teaming difficult")],
    "fig5_interactive", "HealthBench Professional by dataset slice", _si_default, ymax=0.78)
interactive_fig(
    "by_specialty", [(s, SPNAME.get(s, s)) for s in top],
    "fig6_interactive", "HealthBench Professional by specialty",
    lambda nm: nm == "gpt-5.3-chat" or nm.startswith("Qwen3.6-27B"), ymax=0.78, tickangle=35)
print("done")
