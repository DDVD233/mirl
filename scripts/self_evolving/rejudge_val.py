"""Re-judge a saved val_generations dump with a chosen judge, and report the
OFFICIAL HealthBench-Professional metric (length-adjusted, negatives subtracted,
UNCLIPPED = acc_len_adj_signed) plus the other variants.

Reuses the EXACT in-loop scorer (verl.utils.reward_score.healthbench_pro.compute_score)
on the saved model answers, so grading/prompt/length-adjustment match training verbatim
— only the judge (and the fact we grade a fixed set of answers) changes. GPU-free.

Rubrics + conversation come from the official val parquet (index-aligned to the dump);
the answer is the dump's `output` (full text; the scorer strips think/tool spans itself).

Usage:
  REWARD_JUDGE_CONCURRENCY=24 python scripts/self_evolving/rejudge_val.py \
    --dump  /scratch/.../val_generations/<exp>/0.jsonl \
    --parquet /scratch/sheng/self_evolving/healthbench_pro_val.parquet \
    --val_model gpt-chat-latest_2026-05-28 --tag <exp>
"""
import argparse, asyncio, json, os, sys
import numpy as np
import pandas as pd


def _to_py(x):
    """Deep-convert numpy arrays/scalars (from parquet) to plain Python so the
    scorer's truthiness / list ops work (rubric_items is stored as an np.array)."""
    if isinstance(x, np.ndarray):
        return [_to_py(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x

sys.path.insert(0, "/scratch/sheng/self_evolving/verl_healthbench")
from verl.utils.reward_score import healthbench_pro as HB  # noqa: E402

VAL_BASE = os.environ.get("VAL_JUDGE_BASE", "http://point.dd.works:18890/v1")
VAL_KEY = os.environ.get("VAL_JUDGE_KEY", "sk-xMByFeWLKB87wZ")


async def score_row(i, ei, answer, val_model):
    ei = dict(ei)
    ei["_is_validation"] = True
    try:
        return await HB.compute_score(
            data_source="healthbench_professional",
            solution_str=answer or "",
            ground_truth="",
            extra_info=ei,
            val_api_base=VAL_BASE, val_api_key=VAL_KEY,
            val_model_name=val_model, val_provider="trapi",
        )
    except Exception as e:
        print(f"row {i} error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


async def main(a):
    df = pd.read_parquet(a.parquet)
    dump = [json.loads(l) for l in open(a.dump)]
    n = min(len(df), len(dump))
    print(f"[{a.tag}] parquet={len(df)} dump={len(dump)} -> scoring {n} rows with {a.val_model}", flush=True)

    async def one(i):
        ei = _to_py(df.iloc[i]["extra_info"])
        return await score_row(i, ei, dump[i].get("output", ""), a.val_model)

    results = await asyncio.gather(*[one(i) for i in range(n)])
    results = [r for r in results if r is not None]
    if not results:
        print("no results"); return
    import statistics as st
    def m(k): return st.mean(float(r.get(k, 0.0)) for r in results)
    print(f"\n===== {a.tag}  (judge={a.val_model}, N={len(results)}) =====")
    print(f"  acc_len_adj_signed (OFFICIAL primary)  = {m('acc_len_adj_signed'):.4f}")
    print(f"  acc_len_adj        (len-adj, clipped)  = {m('acc_len_adj'):.4f}")
    print(f"  acc_raw_signed     (raw, signed)       = {m('acc_raw_signed'):.4f}")
    print(f"  acc_raw            (raw, clipped)      = {m('acc_raw'):.4f}")
    print(f"  format_ok = {m('format_ok'):.3f}  think_chars = {m('think_chars'):.0f}")
    out = {"tag": a.tag, "judge": a.val_model, "n": len(results),
           "acc_len_adj_signed": m("acc_len_adj_signed"), "acc_len_adj": m("acc_len_adj"),
           "acc_raw_signed": m("acc_raw_signed"), "acc_raw": m("acc_raw")}
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"appended -> {a.out}", flush=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--parquet", default="/scratch/sheng/self_evolving/healthbench_pro_val.parquet")
    p.add_argument("--val_model", default="gpt-chat-latest_2026-05-28")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", default="/scratch/sheng/self_evolving/rejudge_results.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse()))
