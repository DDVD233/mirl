"""Verify WikiDoc title rows are ingested and retrievable from
`medical_knowledge_v2`, using the exact embed+search path the generation
server uses (mib:18001 embed -> Milvus ANN search)."""

import json
import os
import urllib.request

from pymilvus import MilvusClient

COLLECTION = "medical_knowledge_v2"
EMBED_BASE = os.environ.get("EMBED_API_BASE", "http://localhost:18001/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
EMBED_KEY = os.environ.get("EMBED_API_KEY", "EMPTY")
URI = os.environ.get("MILVUS_URI", "http://localhost:19531")
TOKEN = os.environ.get("MILVUS_TOKEN", "root:Milvus")


def embed(text):
    body = json.dumps({"model": EMBED_MODEL, "input": [text[:2000]]}).encode()
    req = urllib.request.Request(
        f"{EMBED_BASE}/embeddings", data=body,
        headers={"Authorization": f"Bearer {EMBED_KEY}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["data"][0]["embedding"]


def main():
    c = MilvusClient(uri=URI, token=TOKEN)

    total = int(c.query(COLLECTION, filter='source_dataset == "wikidoc"',
                        output_fields=["count(*)"])[0]["count(*)"])
    print(f"wikidoc rows in {COLLECTION}: {total:,}\n")

    # Queries phrased like real user questions (not exact titles) to prove
    # semantic retrieval, restricted to wikidoc rows to confirm they surface.
    queries = [
        "what causes myasthenia gravis muscle weakness",
        "management of acute pancreatitis",
        "differential diagnosis of chest pain",
        "treatment options for atrial fibrillation",
    ]
    for q in queries:
        vec = embed(q)
        hits = c.search(
            collection_name=COLLECTION, data=[vec], limit=5,
            filter='source_dataset == "wikidoc"',
            output_fields=["entry_id", "source_dataset", "content_type", "text_content"],
        )[0]
        print(f"Q: {q}")
        for h in hits:
            e = h["entity"]
            print(f"   {h['distance']:.3f}  [{e['content_type']}] {e['text_content']}  ({e['entry_id']})")
        print()

    # Also confirm wikidoc rows compete in an UNFILTERED search (i.e. they're
    # genuinely retrievable alongside the other 56M rows, not just when forced).
    vec = embed("Wolff-Parkinson-White syndrome")
    hits = c.search(collection_name=COLLECTION, data=[vec], limit=10,
                    output_fields=["source_dataset", "content_type", "text_content"])[0]
    print("Unfiltered top-10 for 'Wolff-Parkinson-White syndrome':")
    for h in hits:
        e = h["entity"]
        print(f"   {h['distance']:.3f}  [{e['source_dataset']}/{e['content_type']}] {e['text_content'][:90]}")


if __name__ == "__main__":
    main()
