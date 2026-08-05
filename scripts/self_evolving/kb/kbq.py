"""One-line CLI over the medical KB, so an auditor can check a claim directly.

    python scripts/self_evolving/kb/kbq.py "amoxicillin dose strep pharyngitis"
    python scripts/self_evolving/kb/kbq.py --raw -k 15 "PEACH score cutoffs"

--raw drops the source filter and the re-ranking (pure dense cosine over all
57M rows) — the difference between the two modes is exactly what separates a
fact our policy discards from a fact the KB does not hold.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.retrieval import RetrieveConfig, local_search, rank_hits  # noqa: E402


def raw_search(query: str, k: int) -> list[dict]:
    import httpx
    from pymilvus import MilvusClient

    r = httpx.post("http://localhost:18001/v1/embeddings",
                   json={"model": "Qwen/Qwen3-VL-Embedding-2B", "input": [query[:2000]]},
                   headers={"Authorization": "Bearer EMPTY"}, timeout=60)
    r.raise_for_status()
    vec = r.json()["data"][0]["embedding"]
    mc = MilvusClient(uri="http://localhost:19531", token="root:Milvus")
    res = mc.search(collection_name="medical_knowledge_v2", data=[vec], limit=k,
                    output_fields=["source_dataset", "content_type", "text_content",
                                   "question", "answer", "entry_id"])
    out = []
    for hl in res:
        for h in hl:
            e = h["entity"]
            out.append({"source": e.get("source_dataset", ""),
                        "content_type": e.get("content_type", ""),
                        "text": (e.get("text_content") or e.get("answer")
                                 or e.get("question") or ""),
                        "score": h["distance"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--raw", action="store_true",
                    help="no source filter, no re-ranking (pure dense)")
    ap.add_argument("--chars", type=int, default=900)
    args = ap.parse_args()

    if args.raw:
        passages = raw_search(args.query, args.k)
    else:
        passages, _ = local_search(args.query, top_k=args.k, cfg=RetrieveConfig())
    if not passages:
        print("NO PASSAGES")
        return
    for i, p in enumerate(passages):
        print(f"\n[{i + 1} | source={p['source']} | cos={p['score']:.3f}]")
        print(p["text"][:args.chars].strip())


if __name__ == "__main__":
    main()
