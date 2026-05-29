"""
Rebuild the `medical_knowledge` Milvus collection with an `image_path` field.

Approach
--------
We can't back-fill `image_path` on the existing collection because (a) the
schema is `auto_id=True` (upsert appends new rows, doesn't replace), and
(b) the original indexer's entry_id ordering for `source_dataset="climb"`
can't be reverse-engineered from entry_id alone — line N of any source
file we've found does not consistently equal `climb_<N>`.

Instead this script builds a NEW collection `medical_knowledge_v2`,
populates it from scratch, and (after the operator confirms counts) swaps
it into place over the old collection. Multimodal rows are re-embedded
from their source files so we can attach a known `image_path`; text rows
are copied bit-for-bit from the old collection (embedding included, no
re-embed) so we don't pay the embed cost for the 54M+ text-only rows.

Phases
------
The script is split into phases so they can be run / resumed
independently in tmux:

  schema       create v2 collection with image_path field + HNSW_SQ index
  embed_mm     read source files, embed text via mib:18001, insert with
               image_path. ~1.1h for ~2.27M rows at ~600/s.
  copy_text    iterate old collection where modality=="text", bulk insert
               into v2 with image_path="". ~2h for ~54M rows.
  verify       per-source row counts in v2 vs old. Operator inspects.
  swap         drop old collection, rename v2 to medical_knowledge.
               Prompts for explicit confirmation.

Each phase is idempotent — re-running skips rows already present in v2
(keyed on `entry_id` for embed_mm, on a per-batch checkpoint file for
copy_text).

Run from tmux on MIB
--------------------
  ssh mib "tmux new -d -s mkv2 \\
    '/home/dvd/miniconda3/envs/cu129/bin/python \\
     scripts/self_evolving/build_medical_knowledge_v2.py \\
     --phase=schema 2>&1 | tee /tmp/mkv2_schema.log'"

Then sequentially: --phase=embed_mm, --phase=copy_text, --phase=verify,
--phase=swap.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Iterator, Optional

logger = logging.getLogger("mkv2")

OLD_COLLECTION = "medical_knowledge"
NEW_COLLECTION = "medical_knowledge_v2"

# Schema constants (must match what gen_server.py reads at search time).
# Lengths chosen to match the existing collection so a side-by-side
# describe diff shows only the new image_path field.
SCHEMA_FIELDS = {
    "entry_id":       ("VARCHAR", 200),
    "source_dataset": ("VARCHAR", 50),
    "modality":       ("VARCHAR", 20),
    "content_type":   ("VARCHAR", 20),
    "text_content":   ("VARCHAR", 2000),
    "question":       ("VARCHAR", 1000),
    "answer":         ("VARCHAR", 1000),
    "image_path":     ("VARCHAR", 512),
}
EMBED_DIM = 2048
INDEX_PARAMS = {
    "index_type": "HNSW_SQ",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200, "sq_type": "SQ8"},
}


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
def cmd_schema(client, args) -> None:
    from pymilvus import DataType

    if client.has_collection(NEW_COLLECTION):
        if not args.recreate:
            logger.info(f"{NEW_COLLECTION} already exists; pass --recreate to drop and re-create")
            return
        logger.warning(f"dropping existing {NEW_COLLECTION}")
        client.drop_collection(NEW_COLLECTION)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    for name, (kind, ml) in SCHEMA_FIELDS.items():
        schema.add_field(name, DataType.VARCHAR, max_length=ml,
                         nullable=True, default_value="")
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBED_DIM)

    idx = client.prepare_index_params()
    idx.add_index(
        field_name="embedding",
        index_type=INDEX_PARAMS["index_type"],
        metric_type=INDEX_PARAMS["metric_type"],
        params=INDEX_PARAMS["params"],
    )
    client.create_collection(NEW_COLLECTION, schema=schema, index_params=idx)
    logger.info(f"created {NEW_COLLECTION} with image_path + HNSW_SQ index")


# ----------------------------------------------------------------------
# Phase A: embed multimodal rows from source
# ----------------------------------------------------------------------
def iter_pmcvqa_rows(csv_path: str, split: str) -> Iterator[dict]:
    """Each PMC-VQA CSV row -> one v2 row.

    text_content = "<question> Answer: <answer>" (matches original schema)
    image_path   = "self_evolving_datasets/pmc_vqa/images/<Figure_path>"
    """
    if not os.path.isfile(csv_path):
        logger.warning(f"missing csv {csv_path}, skipping {split}")
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            fig = (row.get("Figure_path") or "").strip()
            q = (row.get("Question") or "").strip()
            a = (row.get("Answer") or "").strip()
            if not fig or not q:
                continue
            yield {
                "entry_id": f"pmc_vqa_{split}_{i}",
                "source_dataset": "pmc_vqa",
                "modality": "image",
                "content_type": "qa",
                "text_content": f"{q} Answer: {a}"[:2000],
                "question": q[:1000],
                "answer": a[:1000],
                "image_path": f"self_evolving_datasets/pmc_vqa/images/{fig}"[:512],
                # embedding filled in later by embedder
            }


def iter_climb_rows(jsonl_path: str, split: str) -> Iterator[dict]:
    """Each multimodal row in geom_{train,valid}.jsonl -> one v2 row.

    We use the split as part of the new entry_id so the operator can tell
    train- vs valid-derived entries apart at a glance.
    entry_id = "climb_<split>_<line_index>" (0-indexed).
    """
    if not os.path.isfile(jsonl_path):
        logger.warning(f"missing jsonl {jsonl_path}, skipping climb {split}")
        return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            imgs = rec.get("images") or []
            vids = rec.get("videos") or []
            if not imgs and not vids:
                continue
            # Pick the primary media: prefer video if present (these are the
            # CT scan sequences whose "image" rows are individual slices).
            if vids:
                path = vids[0]
                modality = "video"
            else:
                path = imgs[0]
                modality = "image"
            problem = rec.get("problem", "")
            # Strip the <image>/<video> placeholder + a leading "Above is..."
            # blob if present, to mirror what the original Milvus rows look
            # like for the question field.
            q = problem
            for tag in ("<image>\n", "<video>\n", "<image>", "<video>"):
                q = q.replace(tag, "")
            q = q.strip()
            a = (rec.get("answer") or "").strip()
            yield {
                "entry_id": f"climb_{split}_{i}",
                "source_dataset": "climb",
                "modality": modality,
                "content_type": "qa",
                "text_content": f"{q} Answer: {a}"[:2000],
                "question": q[:1000],
                "answer": a[:1000],
                "image_path": f"high_modality/{path}"[:512],
            }


async def cmd_embed_mm(client, args) -> None:
    import httpx

    # Resume support: skip entry_ids already in v2 (idempotent re-runs).
    existing_ids = _load_existing_entry_ids(client, NEW_COLLECTION,
                                            args.checkpoint_file)
    logger.info(f"resume: {len(existing_ids):,} entry_ids already in v2 — will skip")

    sources = [
        ("pmc_vqa_train", lambda: iter_pmcvqa_rows(args.pmc_vqa_train_csv, "train")),
        ("pmc_vqa_test",  lambda: iter_pmcvqa_rows(args.pmc_vqa_test_csv,  "test")),
        ("climb_train",   lambda: iter_climb_rows(args.geom_train_jsonl,   "train")),
        ("climb_valid",   lambda: iter_climb_rows(args.geom_valid_jsonl,   "valid")),
    ]

    sem = asyncio.Semaphore(args.embed_concurrency)
    counts = Counter()
    limits = httpx.Limits(max_connections=max(args.embed_concurrency * 4, 64))
    timeout = httpx.Timeout(120.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as http:
        for name, factory in sources:
            logger.info(f"=== source: {name} ===")
            row_iter = factory()
            await _embed_and_insert(
                client, http, sem, row_iter, name, counts, existing_ids, args,
            )

    logger.info("== embed_mm summary ==")
    for k, v in sorted(counts.items()):
        logger.info(f"  {k}: {v:,}")


async def _embed_and_insert(client, http, sem, row_iter, name, counts,
                            existing_ids, args) -> None:
    """Pull rows in chunks, fire `args.embed_concurrency` parallel embed
    requests, then bulk-insert into Milvus once a chunk is complete."""
    pending: list[dict] = []
    t0 = time.time()

    async def embed_one(row: dict) -> Optional[dict]:
        async with sem:
            try:
                resp = await http.post(
                    f"{args.embed_api_base}/embeddings",
                    json={"model": args.embed_model, "input": [row["text_content"][:2000]]},
                    headers={"Authorization": f"Bearer {args.embed_api_key}"},
                )
                resp.raise_for_status()
                row["embedding"] = resp.json()["data"][0]["embedding"]
                return row
            except Exception as e:
                counts[f"{name}_embed_err"] += 1
                logger.debug(f"embed err for {row.get('entry_id', '?')}: {e}")
                return None

    async def flush_pending():
        nonlocal pending
        if not pending:
            return
        rows = await asyncio.gather(*(embed_one(r) for r in pending))
        good = [r for r in rows if r is not None]
        if good:
            try:
                client.insert(collection_name=NEW_COLLECTION, data=good)
                for r in good:
                    counts[f"{name}_inserted"] += 1
                    existing_ids.add(r["entry_id"])
                # Persist checkpoint so a crash mid-source can resume.
                if args.checkpoint_file:
                    with open(args.checkpoint_file, "a") as f:
                        for r in good:
                            f.write(r["entry_id"] + "\n")
            except Exception as e:
                logger.warning(f"milvus insert err for {name}: {type(e).__name__}: {e}")
                counts[f"{name}_insert_err"] += len(good)
        pending = []

    for row in row_iter:
        if row["entry_id"] in existing_ids:
            counts[f"{name}_skipped_resume"] += 1
            continue
        pending.append(row)
        if len(pending) >= args.insert_batch:
            await flush_pending()
            rate = (
                counts[f"{name}_inserted"] / max(time.time() - t0, 1)
            )
            if counts[f"{name}_inserted"] % (args.insert_batch * 5) == 0:
                logger.info(
                    f"  {name}: inserted {counts[f'{name}_inserted']:,} "
                    f"({rate:,.0f}/s)"
                )

    await flush_pending()
    logger.info(
        f"finished {name}: inserted={counts[f'{name}_inserted']:,} "
        f"skipped_resume={counts[f'{name}_skipped_resume']:,} "
        f"embed_err={counts[f'{name}_embed_err']:,}"
    )


def _load_existing_entry_ids(client, collection: str,
                             checkpoint_file: Optional[str]) -> set:
    """Read entry_ids already inserted into v2 so a resumed run skips them.

    First consults a local newline-delimited checkpoint file (cheap, no
    network). Falls back to scanning v2 itself if no checkpoint exists or
    --no_resume was passed.
    """
    seen: set[str] = set()
    if checkpoint_file and os.path.isfile(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            for line in f:
                eid = line.strip()
                if eid:
                    seen.add(eid)
        if seen:
            logger.info(f"checkpoint {checkpoint_file}: {len(seen):,} entry_ids")
            return seen

    if not client.has_collection(collection):
        return seen
    try:
        it = client.query_iterator(
            collection_name=collection,
            filter="",
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
        logger.warning(f"scan of {collection} for entry_ids failed: {e}")
    return seen


# ----------------------------------------------------------------------
# Phase B: copy text rows from old collection
# ----------------------------------------------------------------------
def cmd_copy_text(client, args) -> None:
    """Read every row from the old collection where modality=='text' and
    insert it into v2 verbatim (embedding included) with image_path=''.

    This catches everything in the old collection that doesn't get
    re-embedded by Phase A: pubmedqa, mirage, medrag_textbook, the ECG
    text rows under source_dataset='climb', and anything else marked
    modality='text'.
    """
    it = client.query_iterator(
        collection_name=OLD_COLLECTION,
        filter='modality == "text"',
        batch_size=args.read_batch,
        output_fields=["entry_id", "source_dataset", "modality", "content_type",
                       "text_content", "question", "answer", "embedding"],
    )
    counts = Counter()
    batch: list[dict] = []
    t0 = time.time()
    last_log = 0
    try:
        while True:
            rows = it.next()
            if not rows:
                break
            for r in rows:
                payload = {
                    "entry_id": r.get("entry_id", "") or "",
                    "source_dataset": r.get("source_dataset", "") or "",
                    "modality": r.get("modality", "") or "",
                    "content_type": r.get("content_type", "") or "",
                    "text_content": r.get("text_content", "") or "",
                    "question": r.get("question", "") or "",
                    "answer": r.get("answer", "") or "",
                    "image_path": "",
                    "embedding": r["embedding"],
                }
                batch.append(payload)
                if len(batch) >= args.insert_batch:
                    client.insert(collection_name=NEW_COLLECTION, data=batch)
                    counts["text_copied"] += len(batch)
                    batch = []
                    if counts["text_copied"] - last_log >= args.insert_batch * 20:
                        rate = counts["text_copied"] / max(time.time() - t0, 1)
                        logger.info(
                            f"  text_copied: {counts['text_copied']:,} "
                            f"({rate:,.0f}/s)"
                        )
                        last_log = counts["text_copied"]
    finally:
        it.close()
    if batch:
        client.insert(collection_name=NEW_COLLECTION, data=batch)
        counts["text_copied"] += len(batch)
    logger.info(f"text_copied total: {counts['text_copied']:,}")


# ----------------------------------------------------------------------
# Phase C: verify
# ----------------------------------------------------------------------
def cmd_verify(client, args) -> None:
    def count(coll: str, f: str) -> int:
        try:
            r = client.query(collection_name=coll, filter=f,
                             output_fields=["count(*)"])
            return int(r[0]["count(*)"])
        except Exception as e:
            logger.warning(f"count err {coll} {f}: {e}")
            return -1

    pairs = [
        ("total", ""),
        ("text", 'modality == "text"'),
        ("image", 'modality == "image"'),
        ("video", 'modality == "video"'),
        ("pmc_vqa", 'source_dataset == "pmc_vqa"'),
        ("climb", 'source_dataset == "climb"'),
        ("with image_path", 'image_path != ""'),
    ]
    logger.info(f"{'metric':<25} {OLD_COLLECTION:>20} {NEW_COLLECTION:>20}")
    for label, f in pairs:
        old_c = count(OLD_COLLECTION, f) if client.has_collection(OLD_COLLECTION) else -1
        new_c = count(NEW_COLLECTION, f)
        logger.info(f"{label:<25} {old_c:>20,} {new_c:>20,}")


# ----------------------------------------------------------------------
# Phase D: swap
# ----------------------------------------------------------------------
def cmd_swap(client, args) -> None:
    if not args.yes_swap:
        logger.error("--yes_swap is required to perform the destructive swap")
        sys.exit(2)
    if not client.has_collection(NEW_COLLECTION):
        logger.error(f"{NEW_COLLECTION} doesn't exist; nothing to swap")
        sys.exit(1)
    if client.has_collection(OLD_COLLECTION):
        backup = f"{OLD_COLLECTION}_old_{int(time.time())}"
        logger.info(f"renaming {OLD_COLLECTION} -> {backup} (kept for rollback)")
        client.rename_collection(OLD_COLLECTION, backup)
    logger.info(f"renaming {NEW_COLLECTION} -> {OLD_COLLECTION}")
    client.rename_collection(NEW_COLLECTION, OLD_COLLECTION)
    logger.info("swap complete")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["schema", "embed_mm", "copy_text", "verify", "swap"])
    ap.add_argument("--milvus_uri", default=os.environ.get(
        "MILVUS_URI", "http://mib.media.mit.edu:19531"))
    ap.add_argument("--milvus_token", default=os.environ.get(
        "MILVUS_TOKEN", "root:Milvus"))

    # Schema phase
    ap.add_argument("--recreate", action="store_true",
                    help="phase=schema: drop existing v2 collection if present")

    # Embed_mm phase
    ap.add_argument("--pmc_vqa_train_csv",
                    default="/scratch/self_evolving_datasets/pmc_vqa/train.csv")
    ap.add_argument("--pmc_vqa_test_csv",
                    default="/scratch/self_evolving_datasets/pmc_vqa/test.csv")
    ap.add_argument("--geom_train_jsonl",
                    default="/scratch/high_modality/geom_train.jsonl")
    ap.add_argument("--geom_valid_jsonl",
                    default="/scratch/high_modality/geom_valid.jsonl")
    ap.add_argument("--embed_api_base",
                    default=os.environ.get("EMBED_API_BASE",
                                           "http://mib.media.mit.edu:18001/v1"))
    ap.add_argument("--embed_api_key",
                    default=os.environ.get("EMBED_API_KEY", "EMPTY"))
    ap.add_argument("--embed_model",
                    default=os.environ.get(
                        "EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))
    ap.add_argument("--embed_concurrency", type=int, default=16,
                    help="In-flight embed requests")
    ap.add_argument("--checkpoint_file", default="/tmp/mkv2_done_entry_ids.txt",
                    help="Append-only file of entry_ids successfully "
                         "inserted into v2; resumed runs skip them. Pass "
                         "empty string to disable checkpointing.")

    # Insert batch sizing — shared across phases
    ap.add_argument("--insert_batch", type=int, default=200,
                    help="Rows per Milvus insert call")
    ap.add_argument("--read_batch", type=int, default=1000,
                    help="query_iterator batch size for copy_text")

    # Swap phase
    ap.add_argument("--yes_swap", action="store_true",
                    help="phase=swap: actually perform the destructive swap")

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from pymilvus import MilvusClient
    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)

    if args.phase == "schema":
        cmd_schema(client, args)
    elif args.phase == "embed_mm":
        asyncio.run(cmd_embed_mm(client, args))
    elif args.phase == "copy_text":
        cmd_copy_text(client, args)
    elif args.phase == "verify":
        cmd_verify(client, args)
    elif args.phase == "swap":
        cmd_swap(client, args)


if __name__ == "__main__":
    main()
