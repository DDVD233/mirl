"""HealthBench Professional result figures (plotly, white bg, Computer Modern).

Reads HEALTHBENCH_RESULTS.json (same dir) and writes PDF+PNG bar plots:
  fig1_base_models.{pdf,png}        — 4 base models, overall vs length-adjusted
  fig2_checkpoints_by_family.{pdf,png} — 2x2 subplots per base family, length-adj,
                                          bars colored by improvement over base + base line
  fig3_all_runs_ranked.{pdf,png}    — all 32 runs ranked by length-adjusted

Style follows the lab reference: palette #7dbfa7 (teal) / #da81c1 (pink),
Computer Modern serif font, white plot/paper background, black text, light grids.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "HEALTHBENCH_RESULTS.json"))
RUNS = D["runs"]

# --- style ---
TEAL = "#7dbfa7"   # improvement / checkpoints
PINK = "#da81c1"   # regression / base models
GRAY = "#9aa0a6"
FONT = "Computer Modern, CMU Serif, serif"  # serif fallback if CM not installed
GRID = "rgba(128,128,128,0.2)"

BASE_DISP = {
    "Qwen_Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen_Qwen3.6-27B": "Qwen3.6-27B",
    "google_gemma-4-31B-it": "gemma-4-31B-it",
    "gpt-5.3-chat_2026-03-03": "gpt-5.3-chat",
}
# (family title, checkpoint model prefix, base model key or None)
FAMILIES = [
    ("Qwen3.6-27B", "qwen36_27b_", "Qwen_Qwen3.6-27B"),
    ("gemma-4-31B-it", "gemma4_31b_", "google_gemma-4-31B-it"),
    ("Qwen3.5-9B", "qwen35_9b_", "Qwen_Qwen3.5-9B"),
    ("gemma-4-E4B", "gemma4_e4b_", None),
]

base = {r["model"]: r for r in RUNS if r["kind"] == "base"}
ckpts = [r for r in RUNS if r["kind"] == "checkpoint"]


def style(fig, *, height, width):
    fig.update_layout(
        height=height, width=width,
        font=dict(family=FONT, color="black", size=18),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=90, b=120, l=80, r=40),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID,
                     tickfont=dict(family=FONT, color="black"),
                     tickcolor="black", linecolor="black", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID,
                     tickfont=dict(family=FONT, color="black"),
                     title_font=dict(family=FONT, color="black"),
                     tickcolor="black", linecolor="black", zeroline=True,
                     zerolinecolor="rgba(0,0,0,0.35)")
    for a in fig["layout"]["annotations"]:
        a["font"] = dict(size=22, family=FONT, color="black")
    return fig


def save(fig, stem):
    fig.write_image(str(HERE / f"{stem}.pdf"), engine="kaleido")
    fig.write_image(str(HERE / f"{stem}.png"), engine="kaleido", scale=2)
    fig.write_html(str(HERE / f"{stem}.html"), include_plotlyjs="cdn")
    print(f"wrote {stem}.pdf / .png / .html")


# ---------------------------------------------------------------- figure 1
def fig_base_models():
    def best_ckpt(prefix):
        return max((r for r in ckpts if r["model"].startswith(prefix)),
                   key=lambda r: r["overall_score_length_adjusted"])
    rows = []
    for fam, pre in [("Qwen3.5-9B", "qwen35_9b_"), ("Qwen3.6-27B", "qwen36_27b_"),
                     ("gemma-4-31B-it", "gemma4_31b_")]:
        r = best_ckpt(pre)
        rows.append((f"{fam}<br>({r['model'][len(pre):]})", r))  # best checkpoint per family
    rows.append(("gpt-5.3-chat", base["gpt-5.3-chat_2026-03-03"]))
    x = [name for name, _ in rows]
    overall = [r["overall_score"] for _, r in rows]
    lenadj = [r["overall_score_length_adjusted"] for _, r in rows]
    fig = go.Figure()
    fig.add_bar(x=x, y=overall, name="Overall", marker_color=TEAL,
                text=[f"{v:.3f}" for v in overall], textposition="outside",
                textfont=dict(size=16, family=FONT, color="black"))
    fig.add_bar(x=x, y=lenadj, name="Length-adjusted", marker_color=PINK,
                text=[f"{v:.3f}" for v in lenadj], textposition="outside",
                textfont=dict(size=16, family=FONT, color="black"))
    fig.update_layout(
        barmode="group", title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=20, family=FONT, color="black"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1),
    )
    fig.update_yaxes(title_text="HealthBench Professional score", range=[0, 0.48],
                     title_font=dict(size=22, family=FONT))
    fig.update_xaxes(tickfont=dict(size=16, family=FONT, color="black"))
    style(fig, height=580, width=940)
    save(fig, "fig1_best_checkpoints")


# ---------------------------------------------------------------- figure 2
def fig_checkpoints_by_family():
    def fam_title(t, bm):
        if bm and bm in base:
            return f"{t} fine-tunes  (base {base[bm]['overall_score_length_adjusted']:.3f})"
        return f"{t} fine-tunes"
    fams = FAMILIES[:2]  # top 2 panels only: Qwen3.6-27B, gemma-4-31B-it
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[fam_title(t, bm) for t, _, bm in fams],
                        horizontal_spacing=0.11)
    for idx, (title, prefix, basem) in enumerate(fams):
        row, col = 1, idx + 1
        # one bar per variant: for duplicate steps, keep the best
        best = {}
        for r in (c for c in ckpts if c["model"].startswith(prefix)):
            cur = best.get(r["model"])
            if cur is None or r["overall_score_length_adjusted"] > cur["overall_score_length_adjusted"]:
                best[r["model"]] = r
        fam = sorted(best.values(), key=lambda r: r["overall_score_length_adjusted"], reverse=True)
        xlab = [r["model"][len(prefix):] for r in fam]   # variant only (no step number)
        y = [r["overall_score_length_adjusted"] for r in fam]
        bscore = base[basem]["overall_score_length_adjusted"] if basem else None
        colors = [TEAL if (bscore is not None and v >= bscore) else PINK for v in y]
        fig.add_trace(go.Bar(x=xlab, y=y, marker_color=colors, showlegend=False,
                             text=[f"{v:.3f}" for v in y], textposition="outside",
                             textfont=dict(size=13, family=FONT, color="black"),
                             cliponaxis=False),
                      row=row, col=col)
        fig.update_yaxes(title_text="length-adj score", range=[0.1, 0.42], row=row, col=col,
                         title_font=dict(size=17, family=FONT))
        fig.update_xaxes(tickangle=40, tickfont=dict(size=13, family=FONT, color="black"),
                         row=row, col=col)
        if bscore is not None:
            fig.add_hline(y=bscore, line=dict(color="black", dash="dash", width=1.5),
                          row=row, col=col)
    # legend proxies
    fig.add_trace(go.Bar(x=[None], y=[None], name="≥ base (improves)", marker_color=TEAL))
    fig.add_trace(go.Bar(x=[None], y=[None], name="< base (regresses)", marker_color=PINK))
    fig.add_trace(go.Scatter(x=[None], y=[None], name="untrained base", mode="lines",
                             line=dict(color="black", dash="dash", width=1.5)))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.16, xanchor="center", x=0.5,
                    font=dict(size=16, family=FONT, color="black"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
    style(fig, height=540, width=1150)
    fig.update_layout(margin=dict(t=120, b=120, l=80, r=40))  # room for legend above titles
    save(fig, "fig2_checkpoints_by_family")


# ---------------------------------------------------------------- figure 3
def fig_all_ranked(metric="overall_score_length_adjusted",
                   stem="fig3_all_runs_ranked", metric_label="length-adjusted", xmax=0.42):
    # dedupe checkpoints to the best step per variant (by this metric); keep base models
    best_ck, runs2 = {}, []
    for r in RUNS:
        if r["kind"] == "base":
            runs2.append(r)
        else:
            cur = best_ck.get(r["model"])
            if cur is None or (r[metric] or 0) > (cur[metric] or 0):
                best_ck[r["model"]] = r
    runs2 += list(best_ck.values())
    # truncate at gemma-4-31B-it: keep only runs scoring >= it
    thr = base["google_gemma-4-31B-it"][metric]
    allr = [r for r in runs2 if (r[metric] or 0) >= thr - 1e-9]
    allr = sorted(allr, key=lambda r: r[metric] or 0)
    labels, y, colors = [], [], []
    for r in allr:
        if r["kind"] == "base":
            labels.append(f'★ {BASE_DISP.get(r["model"], r["model"])}')
            colors.append(PINK)
        else:
            labels.append(r["model"])  # variant only (no step number)
            colors.append(TEAL)
        y.append(r[metric])
    fig = go.Figure(go.Bar(
        x=y, y=labels, orientation="h", marker_color=colors, showlegend=False,
        text=[f"{v:.3f}" for v in y], textposition="outside",
        textfont=dict(size=12, family=FONT, color="black"), cliponaxis=False))
    fig.add_trace(go.Bar(x=[None], y=[None], name="base model", marker_color=PINK))
    fig.add_trace(go.Bar(x=[None], y=[None], name="checkpoint", marker_color=TEAL))
    fig.update_layout(
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=16, family=FONT, color="black"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1))
    fig.update_xaxes(title_text=f"HealthBench Professional {metric_label} score",
                     range=[0, xmax], title_font=dict(size=20, family=FONT))
    fig.update_yaxes(tickfont=dict(size=13, family=FONT, color="black"))
    style(fig, height=max(360, 46 * len(allr) + 120), width=950)
    save(fig, stem)


if __name__ == "__main__":
    fig_base_models()
    fig_checkpoints_by_family()
    fig_all_ranked()  # length-adjusted (fig3)
    fig_all_ranked("overall_score", "fig3b_all_runs_ranked_overall", "overall (non-length-adjusted)", 0.47)
    print("done")
