"""Build the official HealthBench Professional val parquet for in-loop validation.

Pulls the held-out `openai/healthbench-professional` benchmark (525 physician-
authored clinician-AI conversations + rubrics) and writes a verl-shape parquet
whose rows are graded by verl/utils/reward_score/healthbench_pro.py exactly like
the training rubric rows — EXCEPT the trainer marks val batches with
`_is_validation`, so the rubric scorer grades them with the gpt-chat-latest judge
instead of self (see healthbench_pro.compute_score / the train script's val_* args).

Row schema (matches scripts/self_evolving/preprocess_healthbench_gen.py):
  - data_source:  "healthbench_professional"   (startswith "healthbench" -> rubric reward)
  - prompt:       [{system}, <clinician conversation turns ending on user>]
  - reward_model: {style: "rubric", ground_truth: ""}
  - extra_info:   {question_id, rubric_items, conversation, use_case, type,
                   difficulty, specialty, split}

This is the SAME data and rubric format the standalone
scripts/self_evolving/eval/healthbench_professional_eval.py uses for the official
length-adjusted score; the standalone runner remains the source of the
per-category breakdown + the official "comparable" number.

Example:
    python scripts/self_evolving/eval/preprocess_healthbench_professional.py \
        --out /scratch/sheng/self_evolving/healthbench_pro_val.parquet
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import pandas as pd

HF_DATASET = "openai/healthbench-professional"

# Keep this IN SYNC with RUBRIC_SOLVER_SYSTEM in
# scripts/self_evolving/generation_server.py so val uses the same solver framing
# the policy is trained under (apples-to-apples).
SYSTEM_PROMPT = (
    "You are a knowledgeable, careful medical AI assistant helping a clinician. Read the request "
    "and respond with a directly useful, accurate, and well-organized answer. Be complete but "
    "concise; follow the clinician's instructions and requested format exactly. Ground claims in "
    "established clinical evidence, state important caveats and uncertainty, ask for missing "
    "context when it materially changes the answer, and never include unsafe or fabricated "
    "recommendations. Prioritize patient safety."
)


def _download_hf_rows(repo_id: str, revision: str | None) -> list[dict]:
    """Raw HealthBench Professional rows from HuggingFace (datasets, parquet fallback)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(repo_id, split="test", revision=revision)
        return [dict(r) for r in ds]
    except Exception as e_ds:
        print(f"[data] `datasets` path failed ({e_ds}); trying raw parquet")
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq

        local = snapshot_download(repo_id, repo_type="dataset", revision=revision)
        parquets = sorted(Path(local).rglob("*.parquet"))
        if not parquets:
            raise RuntimeError(f"no parquet files under {local}")
        rows: list[dict] = []
        for p in parquets:
            rows.extend(pq.read_table(p).to_pylist())
        return rows


def _to_verl_rows(hf_rows: list[dict], split: str):
    for ex in hf_rows:
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
        # BARE prompt (match official eval: conversation as-is, no system message).
        prompt = messages
        # Per-use_case data_source so verl reports each task type separately in val
        # (val-core/healthbench_professional/<use_case>/...). Falls back to "other".
        use_case = (ex.get("use_case") or "other")
        yield {
            "data_source": f"healthbench_professional/{use_case}",
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
                "source": "healthbench_professional",
            },
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--repo_id", default=HF_DATASET)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all 525) for quick val")
    args = ap.parse_args()

    hf_rows = _download_hf_rows(args.repo_id, args.revision)
    rows = list(_to_verl_rows(hf_rows, args.split))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no valid rows from {args.repo_id}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
