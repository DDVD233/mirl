"""
Add WikiDoc (medical wiki) article titles to the `medical_knowledge_v2`
Milvus collection so they're retrievable alongside the existing QA / textbook
knowledge.

Input
-----
`wikidoc_all_titles.json` at the repo root: a flat JSON array of ~200k
article-title strings scraped from WikiDoc. Each surviving title becomes one
text row whose `text_content` is the title itself, embedded with the same
local embedding server (mib:18001, Qwen3-VL-Embedding-2B, dim 2048, COSINE)
used to build the rest of the collection, so the vectors live in the same
space and rank against the same queries.

Row schema (matches build_medical_knowledge_v2.py exactly)
----------------------------------------------------------
  entry_id       "wikidoc_<n>"      (n = index into the *filtered* title list)
  source_dataset "wikidoc"
  modality       "text"
  content_type   "title"
  text_content   the article title  (<=2000)
  question       ""
  answer         ""
  image_path     ""
  embedding      2048-d float vector

Filtering
---------
The raw dump includes non-article MediaWiki namespace pages (Sandbox:, Template:,
Portal:, Wikipedia:, Widget:, author bios in ''' bold markup, etc.). Those carry
no medical content and would only pollute retrieval, so they're dropped. Real
article titles — including the useful "DDx:" (differential diagnosis) pages —
are kept. Duplicates (case-insensitive) are collapsed.

Idempotency / resume
--------------------
entry_ids already present in the collection (or in the append-only checkpoint
file) are skipped, so the script can be killed and re-run. entry_ids are keyed
on position in the *filtered+deduped* list, which is deterministic given the
same input file + filter, so re-runs line up.

Run (from repo root on MIB, where the embed server + Milvus are local)
----------------------------------------------------------------------
  /home/dvd/miniconda3/envs/new2/bin/python \
    scripts/self_evolving/add_wikidoc_titles.py \
    --titles_json wikidoc_all_titles.json 2>&1 | tee /tmp/wikidoc_ingest.log

Add --dry_run to see filter/dedupe counts and a sample without touching Milvus.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter

logger = logging.getLogger("wikidoc")

COLLECTION = "medical_knowledge_v2"
EMBED_DIM = 2048
SOURCE_DATASET = "wikidoc"

# MediaWiki namespaces (text before the first ':') that are not real articles.
DROP_NAMESPACES = {
    "sandbox", "sanbox", "template", "portal", "wikipedia", "wikipedia talk",
    "wp", "w", "wikidoc news", "tweetbook", "wikipatient", "wikipatient talk",
    "widget", "file", "image", "user", "user talk", "category", "help",
    "media", "mediawiki", "mediawiki talk", "talk", "special", "draft",
    "book", "module", "gadget", "topic", "help talk", "category talk",
    "template talk", "portal talk", "draft talk",
}


def clean_title(raw: str) -> str:
    """Normalise a raw title for storage. Returns '' if it should be dropped."""
    t = (raw or "").strip()
    if len(t) < 2:
        return ""
    # Author-bio / markup dumps begin with wiki bold/italic markup.
    if t.startswith("'''") or t.startswith("''"):
        return ""
    # Namespace check on the text before the first colon (only when the colon
    # appears early, i.e. it's a namespace prefix rather than a subtitle).
    if ":" in t:
        head = t.split(":", 1)[0].strip().strip('"').strip("'").lower()
        if head in DROP_NAMESPACES:
            return ""
    return t[:2000]


def load_titles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} is not a JSON array")
    counts = Counter()
    counts["raw"] = len(raw)
    seen = set()
    kept = []
    for item in raw:
        if not isinstance(item, str):
            counts["non_string"] += 1
            continue
        t = clean_title(item)
        if not t:
            counts["filtered"] += 1
            continue
        key = t.lower()
        if key in seen:
            counts["dup"] += 1
            continue
        seen.add(key)
        kept.append(t)
    counts["kept"] = len(kept)
    return kept, counts


def load_existing_entry_ids(client, checkpoint_file):
    """entry_ids already inserted — from the local checkpoint first (cheap),
    else by scanning the collection for source_dataset == wikidoc."""
    seen = set()
    if checkpoint_file and os.path.isfile(checkpoint_file):
        with open(checkpoint_file) as f:
            for line in f:
                eid = line.strip()
                if eid:
                    seen.add(eid)
        if seen:
            logger.info(f"checkpoint {checkpoint_file}: {len(seen):,} entry_ids")
            return seen
    try:
        it = client.query_iterator(
            collection_name=COLLECTION,
            filter=f'source_dataset == "{SOURCE_DATASET}"',
            batch_size=10_000,
            output_fields=["entry_id"],
        )
        while True:
            rows = it.next()
            if not rows:
                break
            for r in rows:
                seen.add(r["entry_id"])
        it.close()
    except Exception as e:
        logger.warning(f"scan for existing wikidoc entry_ids failed: {e}")
    return seen


async def embed_batch(http, args, texts):
    """Embed a list of texts in one request; returns list of vectors."""
    resp = await http.post(
        f"{args.embed_api_base}/embeddings",
        json={"model": args.embed_model, "input": [t[:2000] for t in texts]},
        headers={"Authorization": f"Bearer {args.embed_api_key}"},
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # OpenAI-style APIs return items with an `index`; sort to be safe.
    data = sorted(data, key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


async def run(args):
    import httpx

    titles, fcounts = load_titles(args.titles_json)
    logger.info("title stats: " + ", ".join(f"{k}={v:,}" for k, v in fcounts.items()))
    logger.info("sample kept: " + " | ".join(titles[:8]))

    if args.dry_run:
        logger.info("--dry_run: not connecting to Milvus")
        return

    from pymilvus import MilvusClient
    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
    if not client.has_collection(COLLECTION):
        logger.error(f"{COLLECTION} does not exist")
        sys.exit(1)

    existing = load_existing_entry_ids(client, args.checkpoint_file)
    logger.info(f"resume: {len(existing):,} wikidoc entry_ids already present — will skip")

    # Build the work list of (entry_id, title) not yet inserted.
    work = [
        (f"wikidoc_{i}", t)
        for i, t in enumerate(titles)
        if f"wikidoc_{i}" not in existing
    ]
    logger.info(f"{len(work):,} rows to embed+insert")
    if not work:
        logger.info("nothing to do")
        return

    sem = asyncio.Semaphore(args.embed_concurrency)
    limits = httpx.Limits(max_connections=max(args.embed_concurrency * 4, 64))
    counts = Counter()
    t0 = time.time()
    ckpt = open(args.checkpoint_file, "a") if args.checkpoint_file else None

    # Chunk work into embed batches; run embed_concurrency batches in parallel,
    # then insert everything from that wave in one Milvus call.
    def batches(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(120.0)) as http:
        embed_batches = list(batches(work, args.embed_batch))
        for wave_start in range(0, len(embed_batches), args.embed_concurrency):
            wave = embed_batches[wave_start:wave_start + args.embed_concurrency]

            async def do_one(batch):
                async with sem:
                    try:
                        vecs = await embed_batch(http, args, [t for _, t in batch])
                        if len(vecs) != len(batch):
                            raise ValueError(
                                f"embed count mismatch {len(vecs)} != {len(batch)}")
                        return [
                            {
                                "entry_id": eid,
                                "source_dataset": SOURCE_DATASET,
                                "modality": "text",
                                "content_type": "title",
                                "text_content": title,
                                "question": "",
                                "answer": "",
                                "image_path": "",
                                "embedding": v,
                            }
                            for (eid, title), v in zip(batch, vecs)
                        ]
                    except Exception as e:
                        counts["embed_err"] += len(batch)
                        logger.debug(f"embed err: {type(e).__name__}: {e}")
                        return []

            wave_rows = await asyncio.gather(*(do_one(b) for b in wave))
            rows = [r for batch_rows in wave_rows for r in batch_rows]
            if not rows:
                continue
            # Insert in insert_batch chunks.
            for chunk in batches(rows, args.insert_batch):
                try:
                    client.insert(collection_name=COLLECTION, data=chunk)
                    counts["inserted"] += len(chunk)
                    if ckpt:
                        for r in chunk:
                            ckpt.write(r["entry_id"] + "\n")
                        ckpt.flush()
                except Exception as e:
                    counts["insert_err"] += len(chunk)
                    logger.warning(f"insert err: {type(e).__name__}: {e}")
            done = counts["inserted"]
            if done and done % (args.insert_batch * 10) < args.insert_batch:
                rate = done / max(time.time() - t0, 1)
                logger.info(f"  inserted {done:,}/{len(work):,} ({rate:,.0f}/s)")

    if ckpt:
        ckpt.close()
    logger.info("== summary ==")
    for k, v in sorted(counts.items()):
        logger.info(f"  {k}: {v:,}")
    logger.info(f"done in {time.time() - t0:,.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles_json", default="wikidoc_all_titles.json")
    ap.add_argument("--dry_run", action="store_true",
                    help="print filter/dedupe stats + sample, don't touch Milvus")
    ap.add_argument("--milvus_uri",
                    default=os.environ.get("MILVUS_URI", "http://localhost:19531"))
    ap.add_argument("--milvus_token",
                    default=os.environ.get("MILVUS_TOKEN", "root:Milvus"))
    ap.add_argument("--embed_api_base",
                    default=os.environ.get("EMBED_API_BASE", "http://localhost:18001/v1"))
    ap.add_argument("--embed_api_key",
                    default=os.environ.get("EMBED_API_KEY", "EMPTY"))
    ap.add_argument("--embed_model",
                    default=os.environ.get("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))
    ap.add_argument("--embed_batch", type=int, default=32,
                    help="titles per embedding request")
    ap.add_argument("--embed_concurrency", type=int, default=16,
                    help="in-flight embedding requests")
    ap.add_argument("--insert_batch", type=int, default=500,
                    help="rows per Milvus insert call")
    ap.add_argument("--checkpoint_file", default="/tmp/wikidoc_done_entry_ids.txt",
                    help="append-only entry_id log for resume; '' to disable")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
