"""Download public medical QA corpora and normalise them into STYLE EXEMPLARS.

These are seeds for *how a clinician writes*, never training questions. The
generated training tasks that RL sees must be new clinical scenarios; a seed's
job is only to carry the register — length, informality, typos, multi-turn shape.
`seed_guard.py` enforces that at generation time; this script just builds the pool.

Why these four (measured against the 525 HealthBench-Professional val items):
  HB-Pro prompts are median 258 chars, 61% under 200, 28% start lowercase, ~9%
  non-English, 22% multi-turn. Our generated tasks were median 761 chars, 0%
  short, 0% informal, 0.03% multi-turn — polished vignettes against terse real
  messages. Each corpus below supplies a register we lack:

    K-QA                      real patient questions + physician-written ATOMIC
                              statements — closest thing to HB-Pro's short lenient
                              criteria, so it seeds rubric phrasing too
    HealthSearchQA            terse consumer health queries (brevity)
    MedQuAD                   NIH consumer QA, broad specialty spread
    augmented-clinical-notes  note/documentation material for the `writing` 27%

Output: one JSONL of {text, source, kind, chars, turns} at --out.

    python scripts/self_evolving/kb/fetch_seed_corpora.py \
        --out /scratch/dvd/seed_corpora/style_exemplars.jsonl
"""

import argparse
import json
import os
import random
import re

# (hf_id, config, split, field candidates, kind)
SOURCES = [
    ("katielink/healthsearchqa", "all_data", "train", ["question", "Question", "text"], "consumer_query"),
    ("lavita/MedQuAD", None, "train", ["question", "Question"], "consumer_query"),
    ("AGBonnet/augmented-clinical-notes", None, "train", ["question", "note", "summary", "full_note"], "clinical_note"),
]

# K-QA ships plain JSONL rather than a loadable config, and it is the most valuable
# of the four: `Must_have` is a list of physician-written ATOMIC statements, which is
# structurally what a HealthBench-Pro positive criterion looks like (one checkable
# fact, leniently worded). So it seeds BOTH the question register and the rubric
# phrasing — the two things our generator got most wrong.
KQA_URL = "https://huggingface.co/datasets/Itaykhealth/K-QA/resolve/main/questions_w_answers.jsonl"


def fetch_kqa(out_f, min_chars, max_chars):
    import urllib.request
    kept = 0
    with urllib.request.urlopen(KQA_URL, timeout=90) as r:
        for line in r.read().decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            q = _clean(d.get("Question", ""))
            if not (min_chars <= len(q) <= max_chars):
                continue
            out_f.write(json.dumps({
                "text": q,
                "source": "Itaykhealth/K-QA",
                "kind": "patient_question",
                "chars": len(q),
                "starts_lower": q[:1].islower(),
                "has_question_mark": q.rstrip().endswith("?"),
                # criterion-style exemplars: short atomic clinical statements
                "criterion_exemplars": [
                    _clean(s) for s in (d.get("Must_have") or [])[:6] if isinstance(s, str)
                ],
            }) + "\n")
            kept += 1
    return kept


def _first_field(row, candidates):
    for c in candidates:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _clean(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/dvd/seed_corpora/style_exemplars.jsonl")
    ap.add_argument("--per_source", type=int, default=4000)
    ap.add_argument("--min_chars", type=int, default=30)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)
    written, per_src = 0, {}
    with open(args.out, "w", encoding="utf-8") as f:
        try:
            k = fetch_kqa(f, args.min_chars, args.max_chars)
            per_src["Itaykhealth/K-QA"] = k
            written += k
            print(f"Itaykhealth/K-QA: kept {k:,} (with criterion exemplars)")
        except Exception as e:
            print(f"SKIP K-QA: {type(e).__name__}: {e}")

        for hf_id, cfg, split, fields, kind in SOURCES:
            try:
                ds = load_dataset(hf_id, cfg, split=split)
            except Exception as e:
                print(f"SKIP {hf_id}: {type(e).__name__}: {e}")
                continue
            n = len(ds)
            idx = list(range(n))
            rng.shuffle(idx)
            kept = 0
            for i in idx:
                if kept >= args.per_source:
                    break
                row = ds[i]
                text = _clean(_first_field(row, fields))
                if not (args.min_chars <= len(text) <= args.max_chars):
                    continue
                f.write(json.dumps({
                    "text": text,
                    "source": hf_id,
                    "kind": kind,
                    "chars": len(text),
                    # crude register signals the proposer is asked to imitate
                    "starts_lower": text[:1].islower(),
                    "has_question_mark": text.rstrip().endswith("?"),
                }) + "\n")
                kept += 1
            per_src[hf_id] = kept
            written += kept
            print(f"{hf_id}: {n:,} rows -> kept {kept:,}")
    print(f"\nwrote {written:,} style exemplars -> {args.out}")
    print(json.dumps(per_src, indent=1))


if __name__ == "__main__":
    main()
