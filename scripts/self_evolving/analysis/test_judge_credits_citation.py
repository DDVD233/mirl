#!/usr/bin/env python3
"""Does the JUDGE credit the citation form the KB can actually produce?

THE LAST LINK IN THE REWARD CHAIN. The KB now labels passages, /retrieve appends a source
list, and the policy is told to name what a passage names. None of it pays unless the grader
marks the criterion MET when the answer cites correctly -- and the grader is the verbatim
official HealthBench template, which is strict by design: "If a rubric item has multiple
sentences or criteria, you should consider all of them. If any of the criteria is not met,
the answer should be false."

Two silent failure modes:

  1. The judge ignores attribution and grades only the clinical claim, so citation work is
     unrewarded and the metric cannot move.
  2. The judge insists on the criterion's OWN wording ("the 2000 NEJM trial by Lau et al.")
     and rejects the equivalent the KB supplies ("Lau JY et al., N Engl J Med 2000"), so the
     capability exists but is unreachable.

CURATED, NOT SAMPLED, and deliberately so. An earlier version scanned the val set and
scored 0/4 for two reasons that had nothing to do with attribution: it stripped the
criterion's leading clause to build the "claim", deleting the substantive content the
criterion also requires, and it cited one hardcoded paper against criteria about three other
works -- a wrong citation, not a differently-formatted one. A judge explanation said so
outright: "identifies the study as the 2000 NEJM trial by Lau et al., but it does not state
that this [is foundational evidence for ...]".

So each case below pairs a REAL criterion with a citation form CONFIRMED against esummary
and the built database, holds the clinical content constant, and varies only how the source
is named. The three arms are then interpretable:

  bare      content, source named only as "the published evidence"
  kb        content, source named the way retrieval.py renders it
  verbatim  content, source named exactly as the criterion words it   (reachability ceiling)

  python3 test_judge_credits_citation.py
"""

import argparse
import ast
import asyncio
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HP = os.path.join(REPO, "verl", "utils", "reward_score", "healthbench_pro.py")

# (label, criterion verbatim from the val set, points, task, {arm: answer})
#
# The KB forms are what retrieval.py would put in a passage header for the work each
# criterion names, verified against the published database:
#   10922420 -> ('N Engl J Med', '2000', 'Lau JY', 12)
#   pubmed23n0859_11771 -> title 'Prospective Comparison of Acute Kidney Injury During
#                          Treatment With the Combination of Piperacillin...'  (no "ACORN")
CASES = [
    dict(
        label="work / trial named by author+journal+year",
        crit=("References the 2000 NEJM trial by Lau et al. as foundational evidence for "
              "high-dose IV PPI after endoscopic hemostasis in bleeding peptic ulcers."),
        pts=8,
        task=("A 68-year-old had endoscopic hemostasis for a bleeding gastric ulcer with a "
              "visible vessel. What PPI regimen should follow, and what evidence supports "
              "it?"),
        answers={
            "bare": ("After endoscopic hemostasis for a high-risk bleeding peptic ulcer, "
                     "give high-dose IV PPI (80 mg omeprazole bolus then 8 mg/h for 72 h). "
                     "The published evidence is the foundational support for this."),
            "kb": ("After endoscopic hemostasis for a high-risk bleeding peptic ulcer, give "
                   "high-dose IV PPI (80 mg omeprazole bolus then 8 mg/h for 72 h). The "
                   "foundational evidence for this is Lau JY et al., N Engl J Med 2000 "
                   "(PMID 10922420), which showed high-dose IV omeprazole after endoscopic "
                   "treatment substantially reduces recurrent bleeding."),
            "verbatim": ("After endoscopic hemostasis for a high-risk bleeding peptic ulcer, "
                         "give high-dose IV PPI (80 mg omeprazole bolus then 8 mg/h for "
                         "72 h). The 2000 NEJM trial by Lau et al. is the foundational "
                         "evidence for high-dose IV PPI after endoscopic hemostasis in "
                         "bleeding peptic ulcers."),
        },
    ),
    dict(
        # THE HARD ONE. The KB holds this trial's abstract but its title does NOT contain
        # "ACORN", so the best the pipeline can produce is the paper's identity without the
        # acronym. If the judge needs the acronym, this criterion stays unreachable and the
        # honest conclusion is that titles+metadata are not sufficient for the acronym
        # subset.
        label="work / trial whose KB title lacks the acronym",
        crit=("Mentions that the ACORN study compared AKI in septic patients receiving "
              "piperacillin-tazobactam or cefepime"),
        pts=7,
        task=("For a septic adult needing empiric gram-negative coverage, does "
              "piperacillin-tazobactam cause more AKI than cefepime? What compared them?"),
        answers={
            "bare": ("A randomized trial compared these regimens and found no significant "
                     "difference in acute kidney injury in septic patients."),
            "kb": ("A randomized trial — 'Prospective Comparison of Acute Kidney Injury "
                   "During Treatment With the Combination of Piperacillin-Tazobactam and "
                   "Vancomycin' — compared acute kidney injury in septic patients receiving "
                   "piperacillin-tazobactam versus cefepime."),
            "verbatim": ("The ACORN study compared AKI in septic patients receiving "
                         "piperacillin-tazobactam or cefepime, finding no significant "
                         "difference in kidney injury."),
        },
    ),
    dict(
        # Organisations survive in KB prose, so here the KB form and the criterion's wording
        # are nearly the same. This arm checks that the year matters.
        label="org / guideline named with a year",
        crit=("Mentions recommendations based on the 2022 American College of "
              "Gastroenterology Clinical Guideline for the Diagnosis and Management of "
              "Gastroesophageal Reflux Disease"),
        pts=8,
        task=("An adult with typical heartburn twice weekly asks how GERD should be "
              "diagnosed and managed. What do current guidelines recommend?"),
        answers={
            "bare": ("Current guidance recommends an 8-week empiric PPI trial for typical "
                     "GERD symptoms, with endoscopy reserved for alarm features."),
            "kb": ("Per the American College of Gastroenterology clinical guideline on the "
                   "diagnosis and management of gastroesophageal reflux disease, an 8-week "
                   "empiric PPI trial is recommended for typical symptoms, with endoscopy "
                   "reserved for alarm features."),
            "verbatim": ("Per the 2022 American College of Gastroenterology Clinical "
                         "Guideline for the Diagnosis and Management of Gastroesophageal "
                         "Reflux Disease, an 8-week empiric PPI trial is recommended for "
                         "typical symptoms, with endoscopy reserved for alarm features."),
        },
    ),
]


def grader_template() -> str:
    """The REAL template, read out of the reward module -- not a copy that could drift."""
    tree = ast.parse(open(HP).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "GRADER_TEMPLATE" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("FATAL: GRADER_TEMPLATE not found")


def judge(base, key, model, conversation, rubric_item, tmpl):
    """Grade one (conversation, rubric item) pair through the project's OWN API client.

    Not a hand-rolled request: posting plain JSON returned HTTP 400 because the TRAPI path
    needs handling _call_api already implements. Reusing it also means a client-level
    regression surfaces here.
    """
    sys.path.insert(0, REPO)
    from verl.utils.reward_score.self_evolving import _call_api

    prompt = (tmpl.replace("<<conversation>>", conversation)
                  .replace("<<rubric_item>>", rubric_item))
    txt = asyncio.run(_call_api(base, key, model, "", prompt,
                                max_tokens=700, provider="trapi", timeout_s=180)) or ""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None, txt[:200]
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, txt[:200]
    return bool(d.get("criteria_met")), (d.get("explanation") or "")[:260]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="http://point.dd.works:18890/v1")
    ap.add_argument("--model", default="gpt-chat-latest_2026-05-28")
    ap.add_argument("--key_file", default="/scratch/sheng/self_evolving/.trapi_key")
    ap.add_argument("--votes", type=int, default=1,
                    help="Grade each arm N times; the judge is not deterministic.")
    a = ap.parse_args()
    key = open(a.key_file).read().strip()
    tmpl = grader_template()

    print(f"grader: {a.model}   votes: {a.votes}\n")
    tally = {k: [0, 0] for k in ("bare", "kb", "verbatim")}
    for i, c in enumerate(CASES, 1):
        print(f"--- [{i}/{len(CASES)}] {c['label']}")
        print(f"    criterion: {c['crit'][:150]}")
        for arm in ("bare", "kb", "verbatim"):
            conv = f"user: {c['task']}\n\nassistant: {c['answers'][arm]}"
            mets = []
            why = ""
            for _ in range(a.votes):
                met, expl = judge(a.api_base, key, a.model, conv,
                                  f"[{c['pts']}] {c['crit']}", tmpl)
                mets.append(bool(met))
                why = expl or why
            k = sum(mets)
            tally[arm][0] += len(mets)
            tally[arm][1] += k
            print(f"      {arm:9} met={k}/{len(mets)}  {why[:150]}")
        print()

    print("=== criteria marked MET (across cases x votes) ===")
    for arm, (n, k) in tally.items():
        print(f"  {arm:9} {k}/{n}")
    bare, kb, vb = (tally[x][1] / max(1, tally[x][0]) for x in ("bare", "kb", "verbatim"))
    print()
    if kb > bare and kb >= 0.5 * vb:
        print("  PASS: the KB's citation form is credited, and materially more often than no "
              "citation. The reward can see what retrieval now supplies.")
        return 0
    if vb > bare:
        print("  PARTIAL: these criteria ARE reachable, but only with their own wording. The "
              "KB form is not enough -- the citation must carry the study/guideline NAME, "
              "not only author-journal-year. Titles lacking the acronym stay unreachable.")
        return 2
    print("  FAIL: attribution does not move the grade at all; the judge is scoring the "
          "clinical claim only, so no citation work can move this metric.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
