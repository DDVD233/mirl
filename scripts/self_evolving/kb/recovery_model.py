"""Translate a measured retrieval-recovery rate into expected HealthBench points.

Two multipliers separate "the fact is now in the context window" from "the score
went up", and both are measured, not assumed:

  r  = P(the merged multi-query passage set STATES the criterion's fact)
       measured by multiquery_probe.py, NET of the control arm (the control is
       the probe's own variant-A query, whose non-zero score is the LATENT
       bucket's own false-positive rate).
  d  = the met-rate lift from having the fact in context. Estimated from the
       probe itself: criteria whose fact IS in the natural top-k (RETRIEVABLE)
       are met at 0.622; LATENT criteria are met at 0.418. The 0.204 gap is an
       UPPER bound (it is observational - RETRIEVABLE criteria are also the more
       commonplace facts the model may already know), so a discount is applied.

Negative-point criteria are excluded from the upside and counted as a separate
downside: surfacing the fact makes the model likelier to STATE it, and those
criteria penalise stating it.
"""
import glob
import json

CATS = {f.split("cat_")[1][:-5]: json.load(open(f))
        for f in glob.glob("scripts/self_evolving/kb/probe_out/cat_*.json")}
L = CATS["LATENT"]
TOTAL_ABS = sum(abs(r["points"]) for c in CATS.values() for r in c)

MET_RETRIEVABLE = 0.622   # mean met_rate, positive-point RETRIEVABLE criteria w/ obs
MET_LATENT = 0.418


def run(r, discount, label):
    pos = [x for x in L if x["points"] > 0 and x.get("met_rate") is not None]
    # criteria never graded in val: assume the population met_rate
    unobs = [x for x in L if x["points"] > 0 and x.get("met_rate") is None]
    gain = 0.0
    for x in pos:
        lift = max(0.0, (MET_RETRIEVABLE - x["met_rate"])) * discount
        gain += x["points"] * r * lift
    gain_unobs = sum(x["points"] for x in unobs) * r * (MET_RETRIEVABLE - MET_LATENT) * discount
    neg = [x for x in L if x["points"] < 0]
    # downside: a negative criterion not currently incurred may now be incurred
    dn = 0.0
    for x in neg:
        m = x.get("met_rate")
        m = MET_LATENT if m is None else m
        dn += abs(x["points"]) * r * max(0.0, (MET_RETRIEVABLE - m)) * discount
    net = gain + gain_unobs - dn
    print(f"{label:34s} r={r:.3f} disc={discount:.2f}  "
          f"+{gain:6.1f} (obs) +{gain_unobs:5.1f} (unobs) -{dn:5.1f} (neg) = "
          f"NET {net:6.1f} pts = {net/TOTAL_ABS:.2%} of the 8992-pt val total, "
          f"{net/2940:.1%} of LATENT's 2940")


if __name__ == "__main__":
    print(f"total abs points {TOTAL_ABS:.0f}; LATENT abs {sum(abs(x['points']) for x in L):.0f}")
    for r, lab in [(0.06, "k5->k12 only, same single query"),
                   (0.15, "2 sub-queries"),
                   (0.225, "4 sub-queries k4 (measured)"),
                   (0.30, "6 sub-queries, tuned merge (target)")]:
        for d in (1.0, 0.6, 0.4):
            run(r, d, lab)
        print()
