"""Measure the val-accuracy lift from the broadened answer extractor + the more
lenient judge prompt, on an EXISTING val_generations dump, holding the judge model
constant (so the comparison is apples-to-apples vs the dump's recorded `acc`).

For a random sample of rows it recomputes the primary metric (judge_acc_lenient)
under the NEW pipeline (new extract_final_answer + new JUDGE_ACCURACY_LENIENT_PROMPT)
and compares against the per-row `acc` already stored in the dump (old pipeline).

Usage:
  API_BASE=http://point.dd.works:18890/v1 API_KEY=$(cat .trapi_key) \
  MODEL_NAME=gpt-5.3-chat_2026-03-03 SE_MODULE_PATH=/path/to/self_evolving.py \
  python rejudge_lift.py <dump.jsonl> [sample_n] [seed]
"""
import json, os, sys, asyncio, random, importlib.util

DUMP = sys.argv[1]
SAMPLE_N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
API_BASE = os.environ["API_BASE"]
API_KEY = os.environ["API_KEY"]
MODEL = os.environ["MODEL_NAME"]
SE_PATH = os.environ.get("SE_MODULE_PATH", "/tmp/server_self_evolving.py")

spec = importlib.util.spec_from_file_location("se", SE_PATH)
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)

rows = [json.loads(l) for l in open(DUMP)]
random.seed(SEED)
sample = rows if len(rows) <= SAMPLE_N else random.sample(rows, SAMPLE_N)
n = len(sample)
old_acc = sum(r["acc"] for r in sample) / n

for r in sample:
    r["_new_ext"] = se.extract_final_answer(r["output"])
empty_new = sum(1 for r in sample if not r["_new_ext"])

async def judge_one(r):
    if not r["_new_ext"]:
        return 0.0
    try:
        v = await se.judge_accuracy_lenient(
            api_base=API_BASE, api_key=API_KEY, model_name=MODEL,
            question=r["input"][-1800:], ground_truth=r["gts"],
            extracted_answer=r["_new_ext"], options={},
        )
        return float(v)
    except Exception:
        return 0.0

async def main():
    sem = asyncio.Semaphore(int(os.environ.get("REWARD_JUDGE_CONCURRENCY", "8")))
    async def bound(r):
        async with sem:
            return await judge_one(r)
    verdicts = await asyncio.gather(*[bound(r) for r in sample])
    new_acc = sum(1 for v in verdicts if v >= 0.5) / n
    print(f"dump={DUMP}")
    print(f"sample n={n} (seed={SEED}); empty under new extractor: {empty_new}")
    print(f"OLD acc (dump, boxed-only + old prompt): {old_acc:.3f}")
    print(f"NEW acc (broadened extractor + lenient prompt): {new_acc:.3f}   (+{(new_acc-old_acc)*100:.1f} pts)")
    # show cases NEW=correct but OLD=wrong (recovered) and any NEW=wrong but OLD=correct (regressions)
    rec = [(r, v) for r, v in zip(sample, verdicts) if v >= 0.5 and r["acc"] != 1.0]
    reg = [(r, v) for r, v in zip(sample, verdicts) if v < 0.5 and r["acc"] == 1.0]
    print(f"\nrecovered (old wrong -> new correct): {len(rec)};  regressions (old correct -> new wrong): {len(reg)}")
    for r, v in rec[:15]:
        print(f"  +GT={r['gts']!r}  ||  EXT={str(r['_new_ext'])[:80]!r}")

asyncio.run(main())
