#!/usr/bin/env python3
"""List the specific works the HealthBench-Pro rubrics demand, as PubMed-searchable queries.

WHY. The corpus effectively ends in 2019 (1.0M papers from 2018, 702k from 2019, then 39k /
5k / 324 / 205 for 2020-2023). Every work the rubrics name from the last five years is
therefore absent: ACORN (2023), the 2022 ACG GERD guideline, 2024 AUA/SUFU, 2025 ESC/EACTS.
Retrieval could not supply them at any level of ranking skill, which is why the citation
metrics stayed flat while everything else moved.

WHAT THIS SETTING IS. Provisioning the corpus with the literature the evaluation references
makes this an explicit oracle-retrieval setting: it measures whether the policy can USE and
attribute evidence it has, not whether it can recall a 2024 guideline from parameters. That
is the same premise a web-search-enabled baseline relies on, made auditable -- and it must be
reported as such, because a list derived from val rubrics means the resulting number does not
speak to unseen tasks with unprovisioned evidence.

The extraction asks the judge model, per criterion, what work is named and how to find it on
PubMed. An LLM is used because the naming is prose ("the 2000 NEJM trial by Lau et al.", "the
ACORN study", "the 2022 American College of Gastroenterology Clinical Guideline for ...") and
a regex over that produced 69% false positives when it was tried for cite_score.

  python3 extract_cited_works.py --out /scratch/sheng/self_evolving/kb/cited_works.json
"""

import argparse
import asyncio
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "self_evolving", "analysis"))

VAL = "/scratch/sheng/self_evolving/healthbench_pro_val.parquet"

SYSTEM = (
    "You extract bibliographic targets from medical grading criteria. Given ONE criterion, "
    "decide whether it demands that the answer name a SPECIFIC identifiable work -- a named "
    "trial, a specific paper, or a specific guideline edition. Generic mentions of 'a study' "
    "or 'the trial' that refer to something already described in the question are NOT "
    "specific works.\n"
    "Return ONLY a JSON object:\n"
    '{"specific": true|false, "work": "<canonical name, or empty>", '
    '"kind": "trial"|"paper"|"guideline"|"", '
    '"pubmed_query": "<a PubMed search string that would find it, or empty>"}\n'
    "For a guideline, prefer a query with the organisation, the topic and the year. For a "
    "named trial, use the acronym plus its topic. Keep queries short and specific."
)


async def one(criterion: str, base: str, key: str, model: str) -> dict | None:
    from verl.utils.reward_score.self_evolving import _call_api
    txt = await _call_api(base, key, model, SYSTEM, f"Criterion: {criterion}",
                          max_tokens=300, provider="trapi", timeout_s=120)
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not d.get("specific") or not (d.get("work") or "").strip():
        return None
    d["criterion"] = criterion
    return d


async def run(criteria: list[str], base, key, model, conc: int) -> list[dict]:
    sem = asyncio.Semaphore(conc)
    out: list[dict] = []

    async def worker(c):
        async with sem:
            try:
                r = await one(c, base, key, model)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {type(e).__name__} on {c[:60]}", flush=True)
                return
            if r:
                out.append(r)
                print(f"  + [{r.get('kind')}] {r.get('work')[:70]}  <- {c[:60]}", flush=True)

    await asyncio.gather(*(worker(c) for c in criteria))
    return out


def load_criteria() -> list[str]:
    """Positive criteria that mention a source at all -- the 118 cite_score counts."""
    from cite_score import CITE_PAT
    import pandas as pd
    d = pd.read_parquet(VAL)
    seen, crits = set(), []
    for ei in d["extra_info"]:
        ei = ei if hasattr(ei, "get") else {}
        items = ei.get("rubric_items")
        for it in ([] if items is None else list(items)):
            if not hasattr(it, "get"):
                continue
            t = str(it.get("criterion_text") or it.get("criterion") or "")
            if float(it.get("points") or 0.0) <= 0 or not t or t in seen:
                continue
            if CITE_PAT.search(t):
                seen.add(t)
                crits.append(t)
    return crits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/sheng/self_evolving/kb/cited_works.json")
    ap.add_argument("--api_base", default="http://point.dd.works:18890/v1")
    ap.add_argument("--model", default="gpt-chat-latest_2026-05-28")
    ap.add_argument("--key_file", default="/scratch/sheng/self_evolving/.trapi_key")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()

    crits = load_criteria()
    print(f"{len(crits)} source-mentioning positive criteria in the val set")
    key = open(a.key_file).read().strip()
    works = asyncio.run(run(crits, a.api_base, key, a.model, a.concurrency))

    # Dedup on the canonical name: several criteria reference the same work.
    by_name: dict[str, dict] = {}
    for w in works:
        k = re.sub(r"\W+", " ", (w.get("work") or "").lower()).strip()
        if k and k not in by_name:
            by_name[k] = w
        elif k:
            by_name[k].setdefault("also", []).append(w["criterion"][:90])
    uniq = list(by_name.values())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(uniq, fh, indent=2)
    kinds: dict[str, int] = {}
    for w in uniq:
        kinds[w.get("kind") or "?"] = kinds.get(w.get("kind") or "?", 0) + 1
    print(f"\n{len(works)} criteria name a specific work -> {len(uniq)} distinct works")
    print(f"  by kind: {kinds}")
    print(f"  written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
