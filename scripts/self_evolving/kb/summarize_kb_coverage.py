"""Aggregate probe_kb_coverage.jsonl into the numbers a retrieval decision needs.

The headline is deliberately points-weighted and restricted to knowledge-dependent
criteria: HealthBench scores by points, and NOT_KNOWLEDGE criteria cannot move
under retrieval either way, so leaving them in the denominator understates every
coverage rate.

The number that decides whether to re-introduce retrieval is `upside`: points the
model currently MISSES on criteria whose fact retrieval can already supply
(RETRIEVABLE) or could supply after a policy fix (LATENT). ABSENT points are the
ingestion backlog; CONFLICT points are the risk retrieval adds.

    python scripts/self_evolving/kb/summarize_kb_coverage.py \
        --probe scripts/self_evolving/kb/probe_out/kb_coverage.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict

CATS = ["RETRIEVABLE", "LATENT", "ABSENT", "CONFLICT", "NOT_KNOWLEDGE", "UNPARSED"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--top_sources", type=int, default=25)
    args = ap.parse_args()

    rows, errors = [], 0
    for line in open(args.probe):
        r = json.loads(line)
        if r.get("error"):
            errors += 1
            continue
        for c in r["criteria"]:
            c["question_id"] = r["question_id"]
            c["data_source"] = r["data_source"]
            rows.append(c)

    n_cat = Counter(c["category"] for c in rows)
    # Points weighting uses |points| so negative criteria (things the answer must
    # NOT do) count as the same size of grading target they are.
    pts_cat = defaultdict(float)
    for c in rows:
        pts_cat[c["category"]] += abs(c["points"])
    total_pts = sum(pts_cat.values())

    know = [c for c in rows if c["category"] not in ("NOT_KNOWLEDGE", "UNPARSED")]
    know_pts = sum(abs(c["points"]) for c in know)

    print(f"tasks parsed: {len({c['question_id'] for c in rows})}  "
          f"criteria: {len(rows)}  judge errors: {errors}")
    print(f"\n{'category':<15}{'n':>6}{'n %':>8}{'points':>10}{'pts %':>8}"
          f"{'% of knowledge pts':>20}")
    for k in CATS:
        if not n_cat[k]:
            continue
        kp = pts_cat[k]
        share = f"{100 * kp / know_pts:>19.1f}" if k in (
            "RETRIEVABLE", "LATENT", "ABSENT", "CONFLICT") else " " * 19
        print(f"{k:<15}{n_cat[k]:>6}{100 * n_cat[k] / len(rows):>7.1f}%"
              f"{kp:>10.0f}{100 * kp / total_pts:>7.1f}%{share}")

    lat = Counter(c.get("latent_cause") for c in rows if c["category"] == "LATENT")
    print(f"\nLATENT cause: {dict(lat)}   "
          f"(query = natural query missed it; policy = our filter/rerank dropped it)")

    # ---- what retrieval could actually buy -----------------------------------
    # Restricted to criteria the model is observed to miss on val (met_rate < 1).
    obs = [c for c in know if c.get("n_val_obs")]
    missed_pts = sum(abs(c["points"]) * (1 - c["met_rate"]) for c in obs)
    print(f"\nobserved on val: {len(obs)} knowledge criteria, "
          f"{sum(abs(c['points']) for c in obs):.0f} pts total, "
          f"{missed_pts:.0f} pts currently missed")
    print(f"{'category':<15}{'missed pts':>12}{'% of missed':>14}{'mean met':>10}")
    for k in ("RETRIEVABLE", "LATENT", "ABSENT", "CONFLICT"):
        sub = [c for c in obs if c["category"] == k]
        if not sub:
            continue
        mp = sum(abs(c["points"]) * (1 - c["met_rate"]) for c in sub)
        mm = sum(c["met_rate"] for c in sub) / len(sub)
        print(f"{k:<15}{mp:>12.0f}{100 * mp / missed_pts:>13.1f}%{mm:>10.2f}")

    upside = sum(abs(c["points"]) * (1 - c["met_rate"]) for c in obs
                 if c["category"] in ("RETRIEVABLE", "LATENT"))
    print(f"\nRETRIEVAL-ADDRESSABLE UPSIDE: {upside:.0f} / {missed_pts:.0f} missed pts "
          f"= {100 * upside / missed_pts:.1f}% of what the model loses on\n"
          f"knowledge criteria is a fact the KB already holds.")

    # ---- ingestion backlog ---------------------------------------------------
    absent = [c for c in rows if c["category"] == "ABSENT" and c.get("suggested_source")]
    src = Counter()
    for c in absent:
        s = c["suggested_source"].strip()
        key = s.split(":")[0].split("(")[0].split(" - ")[0].strip().lower()[:48]
        src[key] += 1
    print(f"\nABSENT: {len(absent)} criteria name a source. Top requested:")
    for s, n in src.most_common(args.top_sources):
        print(f"  {n:>4}  {s}")

    by_split = defaultdict(Counter)
    for c in know:
        by_split[c["data_source"]][c["category"]] += 1
    print("\nby split (knowledge criteria only):")
    for k, v in sorted(by_split.items()):
        tot = sum(v.values())
        print(f"  {k:<38}" + "  ".join(
            f"{c}={100 * v[c] / tot:.0f}%" for c in
            ("RETRIEVABLE", "LATENT", "ABSENT", "CONFLICT") if v[c]))

    if args.out_json:
        json.dump({"n_criteria": len(rows), "by_category": dict(n_cat),
                   "points_by_category": dict(pts_cat), "latent_cause": dict(lat),
                   "missed_pts": missed_pts, "upside_pts": upside,
                   "sources_requested": src.most_common(200)},
                  open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
