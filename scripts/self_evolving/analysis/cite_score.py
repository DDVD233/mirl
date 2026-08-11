#!/usr/bin/env python3
"""Track the CITE SCORE: how often the model satisfies criteria that demand a source.

WHY THIS METRIC. The step-165 val analysis of hb9b_specgap_full_rewrite put 34% of the
remaining unmet positive mass on criteria needing a specific fact the 9B does not carry --
guideline editions, trial identities, citations, exact codes and thresholds. Rubric
criteria that explicitly demand a citation or name a study are the cleanest observable
slice of that: 14.3% of unmet positive mass, and their met rate was FLAT across 165 steps
(0.396 -> 0.406) while everything else moved. If retrieval works, this is the number that
must rise; overall accuracy is too diluted to show it early.

Bucketing is decided from the criterion TEXT (keyword match), which is a heuristic and is
reported as such -- it selects the criteria that ask for a source, not the criteria the
model happened to fail.

AND THE HEADLINE `cite` NUMBER IS TOO BROAD TO TRUST ALONE. Audited over the 525 val tasks,
its 118 criteria split three ways:

  81 (69%)  generic "study/trial/reference" about a study ALREADY IN THE PROMPT -- e.g.
            "Notes that the study compared IV PPI use vs placebo". Needs no retrieval.
  22 (19%)  names an issuing ORGANISATION. Reachable: orgs appear in KB prose.
  15 (13%)  names a specific WORK (trial acronym, author, journal). These were
            UNREACHABLE -- the KB stores those abstracts with the bibliography stripped,
            so "ACORN", "SOAP" and "Lau" appear in no passage. kb/build_pubmed_titles.py
            joins the titles back; `work` is the column that tests whether it worked.

So watch `org` and `work`, not `cite`: mixing in the 69% dilutes any real effect ~3x.

Usage:
  python3 cite_score.py --run hb9b_specgap_full_retrieval
  python3 cite_score.py --run hb9b_specgap_full_rewrite     # pre-retrieval baseline
"""

import argparse
import glob
import json
import os
import re

# Criterion mentions sourcing AT ALL. Kept for continuity with the earlier series, but it
# is far too broad to be the headline: audited over the 525-task val set it matches 118
# positive criteria, of which only 37 (31%) ask for an identifiable source. The other 81 say
# "study"/"trial"/"reference" while asking about a study ALREADY DESCRIBED IN THE PROMPT --
# e.g. "Notes that the study compared IV PPI use vs placebo" -- which needs no retrieval at
# all. Reporting only this number dilutes any retrieval effect roughly threefold.
CITE_PAT = re.compile(
    r"\b(cite|citation|citing|reference[sd]?|source[sd]?|per the|according to|"
    r"guideline|guidance|consensus statement|trial|study|studies|RCT|cohort|"
    r"meta-analys|systematic review|society|ACC|AHA|ESC|NICE|WHO|USPSTF|IDSA|ASCO|NCCN)\b",
    re.I,
)
# The criterion names an ISSUING ORGANISATION. These are REACHABLE: organisations survive in
# the KB because they appear in prose ("the American College of Gastroenterology updated its
# guidelines"). 22 criteria.
ORG_PAT = re.compile(
    r"\b(ACC|AHA|ESC|EACTS|NICE|WHO|USPSTF|IDSA|ASCO|NCCN|AUA|SUFU|ACG|ADA|KDIGO|GOLD|"
    r"CDC|FDA|AAP|ACOG|ACR|ATS|ISTH|American \w+|European \w+|National \w+|World Health)\b"
)
# The criterion names a SPECIFIC WORK -- trial acronym, author, journal. 15 criteria, and
# they were UNREACHABLE before kb/build_pubmed_titles.py: the abstracts are in the KB with
# their bibliography stripped, so "ACORN", "SOAP" and "Lau" appear nowhere in any passage.
# This bucket is the direct test of whether the title join worked.
WORK_PAT = re.compile(
    r"\b([A-Z]{3,}[- ]?(?:II|III|\d)?\s+(?:trial|study|RCT)|\w+ et al\.?|"
    r"NEJM|New England Journal|Lancet|JAMA|BMJ|Annals of \w+|Circulation)\b"
)
# Criterion demands a concrete quantity (the other half of the knowledge-shaped gap).
NUM_PAT = re.compile(r"\d")


def load_criteria(parquet):
    import pandas as pd
    d = pd.read_parquet(parquet)
    rows = []
    for ei in d["extra_info"]:
        # `ei or {}` raises on a parquet struct: numpy arrays have no scalar truthiness.
        # Same for `items or []` when rubric_items comes back as an ndarray.
        ei = ei if hasattr(ei, "get") else {}
        items = ei.get("rubric_items")
        items = [] if items is None else list(items)
        rows.append([
            {"text": str(it.get("criterion_text") or it.get("criterion") or ""),
             "points": float(it.get("points") or 0.0)}
            for it in items if hasattr(it, "get")
        ])
    return rows


def met_map(rubric_met):
    """`rubric_met` -> {criterion_text: met}. Tolerates str or list encodings."""
    out = {}
    v = rubric_met
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            return out
    if isinstance(v, dict):
        for k, m in v.items():
            out[str(k)] = bool(m)
    elif isinstance(v, list):
        for it in v:
            if isinstance(it, dict):
                t = it.get("criterion_text") or it.get("criterion") or it.get("text")
                if t is not None:
                    out[str(t)] = bool(it.get("met"))
    return out


def score_step(path, criteria):
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if len(rows) != len(criteria):
        return None
    # cite/numeric/other partition the criteria (each lands in exactly one); org and work
    # are OVERLAPPING sub-buckets of cite, reported separately because they have different
    # ceilings against this corpus.
    agg = {k: [0, 0] for k in ("cite", "numeric", "other", "org", "work")}  # [met, total]
    for row, crits in zip(rows, criteria):
        mm = met_map(row.get("rubric_met"))
        for c in crits:
            if c["points"] <= 0:
                continue
            met = mm.get(c["text"])
            if met is None:
                continue
            t = c["text"]
            is_cite = bool(CITE_PAT.search(t))
            bucket = "cite" if is_cite else ("numeric" if NUM_PAT.search(t) else "other")
            agg[bucket][0] += int(met)
            agg[bucket][1] += 1
            if is_cite:
                if WORK_PAT.search(t):
                    agg["work"][0] += int(met)
                    agg["work"][1] += 1
                elif ORG_PAT.search(t):
                    agg["org"][0] += int(met)
                    agg["org"][1] += 1
    out = {"n_rows": len(rows)}
    for k, (m, t) in agg.items():
        out[f"{k}_met"] = (m / t) if t else None
        out[f"{k}_n"] = t
    # retrieval telemetry, when the arm has it
    for f in ("retrieval_used", "retrieval_coverage", "n_search", "titled_frac"):
        vals = [r[f] for r in rows if isinstance(r.get(f), (int, float))]
        out[f] = (sum(vals) / len(vals)) if vals else None

    # CONTENT vs LENGTH, always reported together.
    #
    # acc_len_adj_signed is the official metric and it is two-sided around a 2000-char
    # centre, so a shorter answer earns credit with no reference to content. That is fine
    # WITHIN one arm, and misleading ACROSS arms whose answer lengths differ for
    # mechanical reasons. Measured at step 0 of the retrieval arm against the same base
    # model without retrieval: the official metric rose +0.0671 while acc_raw_signed FELL
    # -0.0119, because the multi-turn answer budget cut graded length 5954 -> 3265 chars
    # and the length term alone moved +0.0791 (t=-10.84). The entire headline gain was
    # brevity. Since acc_len_adj_signed = acc_raw_signed - length_term by construction,
    # their per-row difference IS the term -- no field-name guessing required.
    for f in ("acc_raw_signed", "acc_len_adj_signed"):
        vals = [r[f] for r in rows if isinstance(r.get(f), (int, float))]
        out[f] = (sum(vals) / len(vals)) if vals else None
    lt = [r["acc_raw_signed"] - r["acc_len_adj_signed"] for r in rows
          if isinstance(r.get("acc_raw_signed"), (int, float))
          and isinstance(r.get("acc_len_adj_signed"), (int, float))]
    out["len_term"] = (sum(lt) / len(lt)) if lt else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default="/scratch/sheng/self_evolving")
    a = ap.parse_args()

    criteria = load_criteria(os.path.join(a.root, "healthbench_pro_val.parquet"))
    d = os.path.join(a.root, "logs_hb9b", "val_generations", a.run)
    files = []
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        m = re.search(r"(\d+)\.jsonl$", os.path.basename(f))
        if m:
            files.append((int(m.group(1)), f))
    files.sort()
    if not files:
        print(f"no val dumps under {d}")
        return

    print(f"run: {a.run}   (cite = criteria whose text demands a named source)")
    print("  content = acc_raw_signed; len_term is what the official metric adds for "
          "brevity.\n  Compare ARMS on content: len_term differs mechanically when answer "
          "budgets differ.")
    print("  org  = criterion names an issuing body (REACHABLE: orgs appear in KB prose)")
    print("  work = criterion names a trial/author/journal (needs the title join)")
    print(f"{'step':>5} {'cite':>7} {'org':>7} {'work':>7} {'numeric':>8} {'other':>7} "
          f"{'retr':>6} {'titled':>7} {'content':>8} {'official':>9}")
    first = True
    for step, f in files:
        s = score_step(f, criteria)
        if not s:
            continue
        if first:
            print(f"{'':5} {'n=' + str(s['cite_n']):>7} {'n=' + str(s['org_n']):>7} "
                  f"{'n=' + str(s['work_n']):>7} {'n=' + str(s['numeric_n']):>8} "
                  f"{'n=' + str(s['other_n']):>7}")
            first = False
        def fmt(x, nd=4):
            return "n/a" if x is None else f"{x:.{nd}f}"
        print(f"{step:>5} {fmt(s['cite_met']):>7} {fmt(s['org_met']):>7} "
              f"{fmt(s['work_met']):>7} {fmt(s['numeric_met']):>8} {fmt(s['other_met']):>7} "
              f"{fmt(s['retrieval_used'],2):>6} {fmt(s.get('titled_frac'),2):>7} "
              f"{fmt(s['acc_raw_signed']):>8} {fmt(s['acc_len_adj_signed']):>9}")


if __name__ == "__main__":
    main()
