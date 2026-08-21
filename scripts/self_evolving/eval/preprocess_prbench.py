"""Build a held-out PRBench (finance/legal) val parquet for in-loop validation.

Same row schema as scripts/self_evolving/eval/preprocess_healthbench_professional.py
(graded by the rubric scorer with the val judge; bare conversation, no system
message). Reads the raw PRBench parquets from NFS -- no HF `datasets` needed.

PRBench rubric weights: important/critically/slightly weights are positive
(1..10), detrimental weights are negative. Exactly one of the six is non-null per
criterion; that value becomes `points` (so the grader's sum(met)/sum(positive)
matches PRBench's weighted scoring).

Example:
    python scripts/self_evolving/eval/preprocess_prbench.py --subset hard \
        --out /scratch/sheng/self_evolving/prbench_hard_val.parquet
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

RAW_DIR = "/scratch/sheng/self_evolving/bench_raw/PRBench/data"
SUBSET_FILES = {
    "hard": ["finance_hard", "legal_hard"],
    "full": ["finance", "legal"],
}
WEIGHT_KEYS = (
    "critically_important_weight",
    "important_weight",
    "slightly_important_weight",
    "critically_detrimental_weight",
    "detrimental_weight",
    "slightly_detrimental_weight",
)


def _is_null(v) -> bool:
    if v is None:
        return True
    try:
        return isinstance(v, float) and math.isnan(v)
    except TypeError:
        return False


def _as_list(v) -> list:
    # parquet list columns come back as numpy arrays; NaN for unused slots.
    if _is_null(v):
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    return list(v)


def _criterion_points(ann: dict):
    for k in WEIGHT_KEYS:
        v = ann.get(k)
        if not _is_null(v):
            return int(round(float(v)))
    return None


def _conversation(ex: dict) -> list[dict]:
    turns = int(ex["turns"])
    msgs = []
    for t in range(turns):
        user = ex.get(f"prompt_{t}")
        if _is_null(user):
            raise ValueError(f"{ex['task']}: prompt_{t} missing for turns={turns}")
        refs = [r for r in _as_list(ex.get(f"reference_texts_{t}")) if isinstance(r, str) and r.strip()]
        content = str(user)
        if refs:
            content += "\n\nReference material:\n" + "\n\n".join(refs)
        msgs.append({"role": "user", "content": content})
        if t < turns - 1:
            resp = ex.get(f"response_{t}")
            if _is_null(resp):
                raise ValueError(f"{ex['task']}: response_{t} missing for turns={turns}")
            msgs.append({"role": "assistant", "content": str(resp)})
    return msgs


def _to_verl_rows(df: pd.DataFrame, subset: str, stats: dict):
    for i, ex in enumerate(df.to_dict("records")):
        field = str(ex["field"]).lower()
        messages = _conversation(ex)
        rubric, categories = [], []
        for it in _as_list(ex.get("rubric")):
            ann = it.get("annotations") or {}
            pts = _criterion_points(ann)
            if pts is None:
                stats["skipped_no_weight"] += 1
                continue
            rubric.append({"criterion_text": it["title"], "points": pts})
            categories.append(ann.get("criteria_category"))
            stats["pos" if pts > 0 else "neg"] += 1
        pos_total = sum(r["points"] for r in rubric if r["points"] > 0)
        assert pos_total > 0, f"{ex['task']}: no positive rubric points"
        yield {
            "data_source": f"prbench/{field}_{subset}",
            "prompt": messages,
            "reward_model": {"style": "rubric", "ground_truth": ""},
            "extra_info": {
                "question_id": str(ex["task"]),
                "index": str(i),
                "split": "test",
                "rubric_items": rubric,
                "conversation": messages,
                "use_case": field,
                "type": None if _is_null(ex.get("decision_type")) else str(ex["decision_type"]),
                "difficulty": subset,
                "specialty": None if _is_null(ex.get("topic")) else str(ex["topic"]),
                "source": "prbench",
                "categories": categories,
                "economic_pathway": None if _is_null(ex.get("economic_pathway")) else str(ex["economic_pathway"]),
            },
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default=RAW_DIR)
    ap.add_argument("--out", default="/scratch/sheng/self_evolving/prbench_hard_val.parquet")
    ap.add_argument("--subset", choices=sorted(SUBSET_FILES), default="hard")
    args = ap.parse_args()

    frames = [pd.read_parquet(Path(args.raw_dir) / f"{n}-00000-of-00001.parquet") for n in SUBSET_FILES[args.subset]]
    df = pd.concat(frames, ignore_index=True)
    # scratchpad = expert notes; never let it leak into the output rows.
    df = df.drop(columns=[c for c in ("scratchpad",) if c in df.columns])

    stats = {"skipped_no_weight": 0, "pos": 0, "neg": 0}
    rows = list(_to_verl_rows(df, args.subset, stats))
    if not rows:
        raise SystemExit(f"no valid rows under {args.raw_dir}")
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
