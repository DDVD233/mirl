"""Build a held-out ProfBench val parquet for in-loop validation.

Same row schema as scripts/self_evolving/eval/preprocess_healthbench_professional.py
(graded by the rubric scorer with the val judge; bare single-turn prompt, no
system message). Reads the raw ProfBench test.jsonl from NFS -- no HF `datasets`.

Official ProfBench weights: Critical=4, Major=3, Minor=2, Additional=1, all
positive; the official score is the weighted fulfilment fraction, which equals
our grader's sum(met points)/sum(positive points).

Example:
    python scripts/self_evolving/eval/preprocess_profbench.py \
        --out /scratch/sheng/self_evolving/profbench_val.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAW = "/scratch/sheng/self_evolving/bench_raw/ProfBench/test.jsonl"
WEIGHTS = {"Critical": 4, "Major": 3, "Minor": 2, "Additional": 1}


def _to_verl_rows(raw_rows: list[dict], stats: dict):
    for i, ex in enumerate(raw_rows):
        domain = str(ex["domain"])
        slug = domain.lower().replace(" ", "_")
        messages = [{"role": "user", "content": str(ex["prompt"])}]
        rubric, ctypes = [], []
        for it in ex.get("rubrics", []):
            w = WEIGHTS.get(it.get("criterion_weight"))
            if w is None:
                stats["skipped_no_weight"] += 1
                continue
            rubric.append({"criterion_text": it["criterion_description"], "points": w})
            ctypes.append(list(it.get("criterion_type") or []))
            stats["pos"] += 1
        assert sum(r["points"] for r in rubric) > 0, f"{ex['task_id']}: no positive rubric points"
        yield {
            "data_source": f"profbench/{slug}",
            "prompt": messages,
            "reward_model": {"style": "rubric", "ground_truth": ""},
            "extra_info": {
                "question_id": str(ex["task_id"]),
                "index": str(i),
                "split": "test",
                "rubric_items": rubric,
                "conversation": messages,
                "use_case": slug,
                "type": "report",
                "difficulty": "expert",
                "specialty": domain,
                "source": "profbench",
                "criterion_types": ctypes,
            },
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default="/scratch/sheng/self_evolving/profbench_val.parquet")
    args = ap.parse_args()

    with open(args.raw, encoding="utf-8") as f:
        raw_rows = [json.loads(line) for line in f if line.strip()]

    stats = {"skipped_no_weight": 0, "pos": 0, "neg": 0}
    rows = list(_to_verl_rows(raw_rows, stats))
    if not rows:
        raise SystemExit(f"no valid rows in {args.raw}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)

    print(f"wrote {len(rows)} rows -> {out}")
    print("per data_source:", pd.Series([r["data_source"] for r in rows]).value_counts().to_dict())
    n_crit = [len(r["extra_info"]["rubric_items"]) for r in rows]
    print(f"criteria/task mean: {sum(n_crit) / len(rows):.2f}")
    print(f"positive criteria: {stats['pos']}  negative criteria: {stats['neg']}  "
          f"skipped (no weight): {stats['skipped_no_weight']}")
    conv_chars = [sum(len(m["content"]) for m in r["prompt"]) for r in rows]
    print(f"mean conversation chars: {sum(conv_chars) / len(rows):.0f}")


if __name__ == "__main__":
    main()
