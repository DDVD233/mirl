"""Convert HealthBench-Pro generated annotations -> verl training parquet.

Input: annotation JSONL from healthbench_gen.py (HuggingFace HealthBench schema:
conversation{messages}, rubric_items[{criterion_text, points}], use_case, type,
difficulty, specialty).

Output: a verl-shape parquet whose rows are graded by
verl/utils/reward_score/healthbench_pro.py (data_source "healthbench_pro"):
  - prompt:       [{system}, <clinician conversation turns>]
  - reward_model: {style: "rubric", ground_truth: ""}
  - extra_info:   {question_id, rubric_items, conversation, use_case, type,
                   difficulty, specialty, split}

The rubric reward reads extra_info["rubric_items"] (and reconstructs the
conversation from extra_info["conversation"]) to score the rollout. No single
ground-truth answer — the rubric IS the supervision.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "You are a careful, safety-conscious clinical assistant helping a physician. "
    "Answer the clinician's request accurately and concisely. Flag missing context, "
    "uncertainty, red flags, and unsafe assumptions where appropriate."
)


def _rows(path: str, split: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            conv = ex.get("conversation")
            messages = conv["messages"] if isinstance(conv, dict) else conv
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            rubric = [
                {"criterion_text": it.get("criterion_text", it.get("criterion")),
                 "points": int(it["points"])}
                for it in ex.get("rubric_items", [])
                if (it.get("criterion_text") or it.get("criterion")) is not None
            ]
            if not messages or not rubric:
                continue
            qid = ex.get("id") or uuid.uuid4().hex
            prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            yield {
                "data_source": "healthbench_pro",
                "prompt": prompt,
                "reward_model": {"style": "rubric", "ground_truth": ""},
                "extra_info": {
                    "question_id": qid,
                    "index": qid,
                    "split": split,
                    "rubric_items": rubric,
                    "conversation": messages,
                    "use_case": ex.get("use_case"),
                    "type": ex.get("type"),
                    "difficulty": ex.get("difficulty"),
                    "specialty": ex.get("specialty"),
                    "source": "healthbench_gen",
                },
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="annotation JSONL from healthbench_gen.py")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    rows = list(_rows(args.inp, args.split))
    if not rows:
        raise SystemExit(f"no valid annotations in {args.inp}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
