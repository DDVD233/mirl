"""Passage-budget-MATCHED control: is the gain from more QUERIES or more PASSAGES?

Both arms return ~24 passages and cost ~the same prompt tokens. The only
difference is whether those 24 come from one query's top-24 or from six
sub-queries' top-4 each.
"""
import argparse, asyncio, json, os, sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.multiquery_probe import (MILVUS_URI, MILVUS_TOKEN, run_task)  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=70)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/mq_matched.json")
    args = ap.parse_args()
    import random
    from collections import Counter
    from pymilvus import MilvusClient

    recs = json.load(open("scripts/self_evolving/kb/probe_out/cat_LATENT.json"))
    bytask = {}
    for r in recs:
        bytask.setdefault(r["question_id"], []).append(r)
    qids = sorted(bytask)
    random.Random(args.seed).shuffle(qids)
    qids = qids[: args.n_tasks]

    arms = {
        "M_1q_t24": dict(n_queries=1, k=24, total=24, merge_per_source=8, policy_per_source=8, fetch_mult=16),
        "M_2q_t24": dict(n_queries=2, k=12, total=24, merge_per_source=8, policy_per_source=8, fetch_mult=16),
        "M_6q_t24": dict(n_queries=6, k=4, total=24, merge_per_source=8, policy_per_source=4, fetch_mult=16),
        "M_6q_t24_nodedup": dict(n_queries=6, k=4, total=24, merge_per_source=8, policy_per_source=4,
                                 fetch_mult=16, drop_stubs=False),
    }
    mc = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        done = await asyncio.gather(*[run_task(client, mc, sem, bytask[q][0]["task"], bytask[q], arms, 6)
                                      for q in qids], return_exceptions=True)
    out = []
    for qid, res in zip(qids, done):
        if isinstance(res, Exception) or res is None:
            continue
        out.append({"question_id": qid, "records": [{"criterion_text": r["criterion_text"], "points": r["points"],
                                                     "met_rate": r["met_rate"]} for r in bytask[qid]], **res})
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"tasks {len(out)}  criteria {sum(len(o['records']) for o in out)}")
    for name in arms:
        sup = tot = 0; chars = []; nps = []
        for o in out:
            a = o["arms"][name]; chars.append(a["chars"]); nps.append(len(a["passages"]))
            jm = {j.get("idx"): j.get("supplied") for j in a["judged"] if isinstance(j, dict)}
            for i in range(len(o["records"])):
                tot += 1; sup += 1 if jm.get(i) else 0
        print(f"{name:18s} {sup:3d}/{tot} = {sup/max(1,tot):.3f}   passages {sum(nps)/len(nps):.1f}  "
              f"~tok {sum(chars)/len(chars)/4.41:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
