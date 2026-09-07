#!/usr/bin/env python3
"""Build a verl val parquet from the ORIGINAL HealthBench release (OpenAI, 2025):
the Hard subset (1,000 examples), the Consensus subset (3,671), or the full set (5,000).

Held-out transfer benchmark for the stage-2 paper: the HB-Pro-trained policies never
saw these conversations. Same row schema as preprocess_healthbench_professional.py so
the in-loop validator (verl/utils/reward_score/healthbench_pro.py, official per-criterion
grader template) scores them unchanged. Differences from HB-Pro that matter:

  - rubric items are under `rubrics` with keys `criterion` / `points` / `tags`;
  - there is NO length adjustment in the original HealthBench protocol, so validate with
    HB_VAL_LENGTH_PENALTY_PER_500=0 and report `acc_raw` (per-example clipped to [0,1],
    averaged), which is the benchmark's official score;
  - `example_tags` carry one `theme:<name>` tag; it is written to `use_case` and to the
    data_source suffix so verl reports per-theme means.

Source files (public blob, also under $S/bench_raw/healthbench/):
  hard      https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl
  consensus https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl
  full      https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl

Example:
    python scripts/self_evolving/eval/preprocess_healthbench_hard.py --subset hard \
        --out /scratch/sheng/self_evolving/healthbench_hard_val.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = "/scratch/sheng/self_evolving/bench_raw/healthbench"
FILES = {
    "hard": "hard_2025-05-08-21-00-10.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
    "full": "2025-05-07-06-14-12_oss_eval.jsonl",
}


def to_rows(path: Path, subset: str):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            messages = [{"role": m["role"], "content": m["content"]} for m in ex["prompt"]]
            rubric = [{"criterion_text": r["criterion"], "points": int(r["points"])}
                      for r in ex.get("rubrics", []) if r.get("criterion") is not None]
            if not messages or not rubric or messages[-1]["role"] != "user":
                continue
            theme = next((t.split(":", 1)[1] for t in ex.get("example_tags", []) if t.startswith("theme:")), "other")
            qid = ex.get("prompt_id")
            yield {
                "data_source": f"healthbench_{subset}/{theme}",
                "prompt": messages,
                "reward_model": {"style": "rubric", "ground_truth": ""},
                "extra_info": {
                    "question_id": qid, "index": qid, "split": "test",
                    "rubric_items": rubric, "conversation": messages,
                    "use_case": theme, "type": None, "difficulty": None, "specialty": None,
                    "source": f"healthbench_{subset}",
                },
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=sorted(FILES), default="hard")
    ap.add_argument("--raw_dir", default=RAW_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = list(to_rows(Path(a.raw_dir) / FILES[a.subset], a.subset))
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    themes = pd.Series([r["extra_info"]["use_case"] for r in rows]).value_counts().to_dict()
    n_items = sum(len(r["extra_info"]["rubric_items"]) for r in rows)
    print(f"wrote {len(rows)} rows ({n_items} criteria) -> {out}; themes={themes}")


if __name__ == "__main__":
    main()
