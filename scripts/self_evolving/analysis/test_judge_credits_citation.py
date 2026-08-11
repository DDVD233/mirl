#!/usr/bin/env python3
"""Does the JUDGE credit a citation the solver could actually produce?

THE LAST LINK IN THE REWARD CHAIN. The KB now labels passages, /retrieve appends a source
list, and the policy is told to name what a passage names. All of that pays exactly nothing
unless the grader marks the criterion MET when the answer cites correctly. And the grader is
the verbatim official HealthBench template, which is deliberately strict: "If a rubric item
has multiple sentences or criteria, you should consider all of them. If any of the criteria is
not met, the answer should be false."

So there are two ways this fails silently:

  1. The judge ignores the citation and grades only the clinical claim -- in which case the
     whole citation project is unrewarded and the metric cannot move.
  2. The judge demands the criterion's OWN phrasing ("the 2000 NEJM trial by Lau et al.")
     and rejects the equivalent reference the KB supplies ("Lau JY et al., N Engl J Med
     2000") -- in which case the capability exists but is unreachable in practice.

Three answer variants per real criterion separate those cases:

  bare    states the clinical fact, no attribution        -> expect NOT met
  kb      cites in the form the KB produces              -> expect MET  (the thing at stake)
  verbatim cites using the criterion's own wording        -> upper bound; if even this fails,
                                                            the criterion is unreachable and
                                                            the problem is not our citation

Criteria are drawn from the actual 525-task val set, matched by the same WORK_PAT/ORG_PAT
cite_score.py uses, so this measures the graded population rather than invented examples.

  python3 test_judge_credits_citation.py --n 6
"""

import argparse
import ast
import asyncio
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "self_evolving", "analysis"))

VAL = "/scratch/sheng/self_evolving/healthbench_pro_val.parquet"
HP = os.path.join(REPO, "verl", "utils", "reward_score", "healthbench_pro.py")


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

    Deliberately not a hand-rolled request: the first version of this posted plain JSON and
    got HTTP 400, because the TRAPI path needs provider-specific handling that
    _call_api already implements. Reusing it also means this test exercises the same client
    the reward does, so a client-level regression shows up here too.
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
    return bool(d.get("criteria_met")), (d.get("explanation") or "")[:220]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_base", default="http://point.dd.works:18890/v1")
    ap.add_argument("--model", default="gpt-chat-latest_2026-05-28")
    ap.add_argument("--key_file", default="/scratch/sheng/self_evolving/.trapi_key")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    key = open(a.key_file).read().strip()
    tmpl = grader_template()

    from cite_score import ORG_PAT, WORK_PAT
    import pandas as pd
    d = pd.read_parquet(VAL)

    # Real criteria that name a source, paired with their own task text.
    cases = []
    for _, row in d.iterrows():
        ei = row["extra_info"]
        ei = ei if hasattr(ei, "get") else {}
        items = ei.get("rubric_items")
        # `prompt` is a chat list (an ndarray of message dicts once through parquet), so
        # `or` on it raises "truth value of an array ... is ambiguous". Pull the user turn
        # explicitly rather than relying on truthiness.
        task = ""
        pr = row["prompt"] if "prompt" in row else None
        try:
            msgs = list(pr) if pr is not None else []
            task = " ".join(str(m.get("content") or "") for m in msgs
                            if hasattr(m, "get") and m.get("role") != "system").strip()
        except Exception:  # noqa: BLE001
            task = ""
        if not task:
            task = str(ei.get("task") or "")
        for it in ([] if items is None else list(items)):
            if not hasattr(it, "get"):
                continue
            t = str(it.get("criterion_text") or it.get("criterion") or "")
            p = float(it.get("points") or 0.0)
            if p <= 0 or not t:
                continue
            kind = ("work" if WORK_PAT.search(t) else "org" if ORG_PAT.search(t) else None)
            if kind and task:
                cases.append((kind, t, p, task[:1500]))
        if len(cases) >= a.n * 3:
            break
    cases = cases[:a.n]
    if not cases:
        print("FATAL: found no source-naming criteria in the val set"); return 1
    print(f"testing {len(cases)} real criteria against the official grader "
          f"({a.model})\n")

    tally = {"bare": [0, 0], "kb": [0, 0], "verbatim": [0, 0]}
    for i, (kind, crit, pts, task) in enumerate(cases, 1):
        # The clinical claim, stripped of attribution: everything after the naming clause.
        claim = re.sub(r"^(mentions|references|states|notes|includes|cites)\b[^,]*,?\s*",
                       "", crit, flags=re.I)
        # A source the KB could actually supply for this topic. Deliberately NOT the
        # criterion's wording -- that is the `verbatim` arm.
        kb_ref = "Lau JY et al., N Engl J Med 2000 (PMID 10922420)"
        variants = {
            "bare": f"{claim} This is well established in the clinical literature.",
            "kb": f"{claim} This comes from {kb_ref}.",
            "verbatim": f"{claim} Specifically, this is supported by {crit[:180]}.",
        }
        print(f"--- [{i}/{len(cases)}] ({kind}, {pts:+.0f}) {crit[:110]}")
        for name, answer in variants.items():
            conv = f"user: {task}\n\nassistant: {answer}"
            met, why = judge(a.api_base, key, a.model, conv, f"[{pts}] {crit}", tmpl)
            tally[name][0] += 1
            tally[name][1] += int(bool(met))
            print(f"      {name:9} met={str(met):5}  {why[:120]}")
        print()

    print("=== SUMMARY: criteria marked MET ===")
    for name, (n, k) in tally.items():
        print(f"  {name:9} {k}/{n}")
    bare_r = tally["bare"][1] / max(1, tally["bare"][0])
    kb_r = tally["kb"][1] / max(1, tally["kb"][0])
    vb_r = tally["verbatim"][1] / max(1, tally["verbatim"][0])
    print()
    if kb_r > bare_r:
        print("  PASS: a KB-style citation is credited more often than no citation, so the "
              "reward can actually see the difference.")
        return 0
    if vb_r > bare_r:
        print("  PARTIAL: the criterion IS reachable, but only with its own wording -- the "
              "KB's reference form is not being credited. The citation needs to carry the "
              "study/guideline NAME, not just author-journal-year.")
        return 2
    print("  FAIL: citing changes nothing. The judge is grading the clinical claim only, so "
          "no citation work can move this metric.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
