"""Re-judge an existing eval jsonl's responses with a chosen judge, to isolate the
JUDGE effect from the model/sampling. Joins each row to the test set by hadm_id to
recover the question, then calls self_evolving.judge_accuracy_lenient with the new
judge. Reports OLD (in-file) vs NEW lenient acc on the SAME responses.

    N=400 JUDGE=gpt-5.3-chat_2026-03-03 /usr/local/bin/python rejudge_gpt55.py
"""
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, "/scratch/sheng/self_evolving/verl")
os.environ.setdefault("CHAT_PROVIDER", "trapi")
from verl.utils.reward_score.self_evolving import judge_accuracy_lenient  # noqa: E402

SRC = os.environ.get("SRC", "/scratch/sheng/self_evolving/eval_hf/eval_openai_gpt-5.5.jsonl")
TEST = "/scratch/sheng/self_evolving/mimiciv_rare/test.jsonl"
JUDGE = os.environ.get("JUDGE", "gpt-5.3-chat_2026-03-03")
BASE = os.environ.get("JUDGE_BASE", "http://point.dd.works:18890/v1")
KEY = open("/scratch/sheng/self_evolving/.trapi_key").read().strip()
N = int(os.environ.get("N", "400"))
CONC = int(os.environ.get("CONC", "12"))

g = [json.loads(l) for l in open(SRC) if l.strip()]
test = {}
for l in open(TEST):
    if l.strip():
        t = json.loads(l)
        ei = t.get("extra_info") or {}
        test[ei.get("hadm_id")] = (ei.get("question", ""),
                                   ei.get("options", {}) if isinstance(ei.get("options"), dict) else {},
                                   (t.get("data_source") or "?"))

random.seed(int(os.environ.get("SEED", "1")))
sample = random.sample(g, min(N, len(g)))
sem = asyncio.Semaphore(CONC)


async def one(r):
    h = r.get("hadm_id")
    q, opts, ds = test.get(h, ("", {}, "?"))
    async with sem:
        try:
            v = await judge_accuracy_lenient(
                api_base=BASE, api_key=KEY, model_name=JUDGE,
                question=q, ground_truth=r.get("ground_truth", "") or "",
                extracted_answer=r.get("extracted_answer", "") or "", options=opts)
        except Exception as e:
            v = None
    return {"new": v, "old": r.get("judge_acc_lenient"), "ds": str(ds).split("/")[-1]}


async def main():
    res = [x for x in await asyncio.gather(*[one(r) for r in sample]) if x["new"] is not None]
    def mean(xs): xs = [x for x in xs if isinstance(x, (int, float))]; return sum(xs) / len(xs) if xs else 0
    print(f"re-judged n={len(res)} (of {len(sample)} sampled)")
    print(f"  OLD in-file judge  lenient acc = {mean([x['old'] for x in res]):.4f}")
    print(f"  NEW {JUDGE} lenient acc = {mean([x['new'] for x in res]):.4f}")
    # agreement
    both = [(x['old'], x['new']) for x in res if isinstance(x['old'], (int, float))]
    agree = sum(1 for o, n in both if (o > 0.5) == (n > 0.5)) / len(both) if both else 0
    print(f"  judge agreement (same verdict): {agree:.1%}")


asyncio.run(main())
