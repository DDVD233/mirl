"""Mechanical (no-LLM) measurement of what the shipped retrieval policy wastes.

Runs every distinct `query_a` in a probe category file through the shipped policy
at several k / per-source-cap settings and counts: under-fill (fewer passages
returned than requested), heading/TOC stubs, exact and near duplicates, and the
source mix. No judging - this is purely "how many of the model's passage slots
carry no new information".
"""
import argparse, asyncio, json, os, sys
from collections import Counter

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.retrieval import RetrieveConfig, rank_hits  # noqa: E402
from kb.multiquery_probe import (EMBED_BASE, EMBED_MODEL, MILVUS_URI, MILVUS_TOKEN,  # noqa: E402
                                 embed, milvus_search, is_stub, shingles, norm)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="scripts/self_evolving/kb/probe_out/cat_LATENT.json")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    from pymilvus import MilvusClient

    recs = json.load(open(args.cat))
    qs, seen = [], set()
    for r in recs:
        q = r.get("query_a")
        if q and q not in seen:
            seen.add(q); qs.append(q)
    qs = qs[: args.limit]
    print(f"{len(qs)} distinct query_a")

    mc = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    async with httpx.AsyncClient() as client:
        vecs = []
        for i in range(0, len(qs), 32):
            vecs += await embed(client, qs[i:i + 32])

    for k, cap in [(5, 3), (5, 6), (12, 3), (12, 4), (12, 6)]:
        rc = RetrieveConfig(); rc.max_per_source = cap
        under = stub = dup_e = dup_n = 0
        tot = 0; chars = 0
        srcs = Counter(); nret = []
        for i in range(0, len(vecs), 16):
            batch = vecs[i:i + 16]
            hs = await asyncio.to_thread(milvus_search, mc, batch, k * rc.fetch_mult, rc.exclude_expr)
            for hl in hs:
                ps = rank_hits(hl, k, rc)
                nret.append(len(ps))
                if len(ps) < k:
                    under += 1
                seen_sh, seen_txt = [], set()
                for p in ps:
                    tot += 1; chars += len(p["text"]); srcs[p["source"]] += 1
                    if is_stub(p["text"]):
                        stub += 1
                    n = norm(p["text"])[:300]
                    if n in seen_txt:
                        dup_e += 1
                    else:
                        seen_txt.add(n)
                        sh = shingles(p["text"])
                        if any(len(sh & pr) / max(1, min(len(sh), len(pr))) >= 0.6 for pr in seen_sh):
                            dup_n += 1
                        seen_sh.append(sh)
        n = len(qs)
        print(f"k={k:2d} cap={cap}: underfilled {under}/{n} ({under/n:.1%})  "
              f"avg returned {sum(nret)/n:.2f}  stubs {stub}/{tot} ({stub/max(1,tot):.1%})  "
              f"exact-dup {dup_e}/{tot} ({dup_e/max(1,tot):.1%})  near-dup {dup_n}/{tot} ({dup_n/max(1,tot):.1%})  "
              f"chars/query {chars/n:.0f} (~{chars/n/4.41:.0f} tok)")
        print("     sources:", srcs.most_common())


if __name__ == "__main__":
    asyncio.run(main())
