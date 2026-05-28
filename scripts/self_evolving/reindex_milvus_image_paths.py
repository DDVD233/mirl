"""
Re-index the `medical_knowledge` Milvus collection with an `image_path` field.

Background
----------
The collection stores 56M+ rows across text/image/video modalities, but the
schema only ever recorded `entry_id` (e.g. `pmc_vqa_train_42`, `climb_14`)
for multimodal rows — no path to the underlying file. The downstream image
server therefore couldn't resolve an entry_id to bytes without consulting
the source files (CSVs, JSONLs) on disk.

This script:
  1. adds an `image_path` VARCHAR(512) field to the collection in place
     (Milvus 2.6 `add_collection_field`, no recreate / no re-embedding);
  2. iterates multimodal rows in batches via `query_iterator`;
  3. resolves each row's entry_id to a relative path via the source files
     (PMC-VQA CSVs and CLIMB JSONLs); and
  4. **full-row upserts** (id + entry_id + source_dataset + modality +
     content_type + text_content + question + answer + embedding +
     image_path), because `partial_update=True` does not actually persist
     the changed field in Milvus 2.6.15 (verified empirically on a test
     collection — silent no-op).

Why full-row upsert is OK
-------------------------
The collection is auto_id=True, so each upsert replaces the row at its
existing primary key with a fresh auto-assigned id. Downstream code keys
on `entry_id` (not `id`), so the new auto-ids don't break anything. The
data is preserved atomically: old row deleted, new row inserted with the
same scalar values + the new `image_path`. The embedding vector is
re-sent unchanged.

Network cost: ~12KB/row × 2.4M multimodal rows ≈ ~30GB read + 30GB write.
On a local Milvus instance this is ~10 min; over WAN it would be ~hours.
**Run from a tmux session on MIB**, not from a dev laptop.

Path format
-----------
Relative paths under `/scratch` on MIB:
  pmc_vqa_train_<N> -> self_evolving_datasets/pmc_vqa/images/<Figure_path>
  pmc_vqa_test_<N>  -> self_evolving_datasets/pmc_vqa/images/<Figure_path>
  climb_<N> (image) -> high_modality/<images[0] from geom_train_images.jsonl[N]>
  climb_<N> (video) -> high_modality/<videos[0] from geom_train_videos.jsonl[N]>

Idempotency
-----------
Rows whose image_path is already non-empty are skipped (`--force` to
override). So a crashed run can be resumed simply by re-running — only
unset rows get re-upserted.

Run
---
  tmux new -d -s reindex \
    'python scripts/self_evolving/reindex_milvus_image_paths.py 2>&1 | tee /tmp/reindex.log'

Then `tmux attach -t reindex` from another SSH session to watch progress.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter

logger = logging.getLogger("reindex")


# ----------------------------------------------------------------------
# Source-file readers
# ----------------------------------------------------------------------
def load_pmcvqa_paths(csv_path: str, split: str) -> dict[str, str]:
    """`pmc_vqa_<split>_<N>` -> `self_evolving_datasets/pmc_vqa/images/<Figure_path>`.

    `N` is the 0-indexed row position after the header. `Figure_path` is the
    `Figure_path` column from the CSV (no directory prefix in the source —
    we prepend `self_evolving_datasets/pmc_vqa/images/` so the path lines
    up with the image server's `<root>/self_evolving_datasets/pmc_vqa/...`
    resolution and the `images/` directory inside `images.zip`.
    """
    out: dict[str, str] = {}
    if not os.path.isfile(csv_path):
        logger.warning(f"pmc_vqa csv missing: {csv_path}")
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            fp = (row.get("Figure_path") or "").strip()
            if not fp:
                continue
            out[f"pmc_vqa_{split}_{i}"] = (
                f"self_evolving_datasets/pmc_vqa/images/{fp}"
            )
    logger.info(f"loaded {len(out):,} pmc_vqa_{split}_* paths from {csv_path}")
    return out


def load_climb_paths(jsonl_path: str, key: str, modality: str) -> dict[tuple[str, str], str]:
    """`(climb_<N>, modality)` -> `high_modality/<images[0] or videos[0]>`.

    `N` is the 0-indexed line position. We key on (entry_id, modality)
    because Milvus uses the same `climb_<N>` namespace for both images
    and videos drawn from different source files — modality is the
    disambiguator.
    """
    out: dict[tuple[str, str], str] = {}
    if not os.path.isfile(jsonl_path):
        logger.warning(f"climb jsonl missing: {jsonl_path}")
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            paths = rec.get(key) or []
            if not paths:
                continue
            out[(f"climb_{i}", modality)] = f"high_modality/{paths[0]}"
    logger.info(f"loaded {len(out):,} climb {modality} paths from {jsonl_path}")
    return out


# ----------------------------------------------------------------------
# Milvus reindexer
# ----------------------------------------------------------------------
def ensure_image_path_field(client, collection: str, max_length: int) -> None:
    from pymilvus import DataType

    desc = client.describe_collection(collection)
    fields = {f["name"] for f in desc["fields"]}
    if "image_path" in fields:
        logger.info("image_path field already present — skipping schema change")
        return
    logger.info("adding image_path field to schema...")
    client.add_collection_field(
        collection_name=collection,
        field_name="image_path",
        data_type=DataType.VARCHAR,
        nullable=True,
        max_length=max_length,
        default_value="",
    )
    logger.info("image_path field added")


# Every scalar field that exists on the collection, in the order we expect.
# Used to build the full-row payload for upsert. We refresh from
# describe_collection at runtime to stay schema-agnostic, but document
# the expected set here for the reader.
EXPECTED_SCALAR_FIELDS = (
    "entry_id", "source_dataset", "modality", "content_type",
    "text_content", "question", "answer", "image_path",
)


def _collection_fields(client, collection: str) -> tuple[list[str], str, str]:
    """Return (all_field_names_excluding_pk, pk_name, vector_field_name)."""
    desc = client.describe_collection(collection)
    pk = None
    vec = None
    others: list[str] = []
    for f in desc["fields"]:
        n = f["name"]
        t = f["type"]
        if f.get("is_primary"):
            pk = n
            continue
        if t == 101:  # FLOAT_VECTOR
            vec = n
            others.append(n)
            continue
        others.append(n)
    if pk is None or vec is None:
        raise RuntimeError(f"could not infer schema: pk={pk} vec={vec}")
    return others, pk, vec


def reindex(args) -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
    ensure_image_path_field(client, args.collection, args.path_max_length)

    other_fields, pk_field, vec_field = _collection_fields(client, args.collection)
    logger.info(f"schema: pk={pk_field}  vector={vec_field}  others={other_fields}")
    output_fields = [pk_field, *other_fields]

    # Build path lookup tables from source files.
    pmc_train = load_pmcvqa_paths(args.pmc_vqa_train_csv, "train")
    pmc_test = load_pmcvqa_paths(args.pmc_vqa_test_csv, "test")
    climb_images = load_climb_paths(args.climb_images_jsonl, "images", "image")
    climb_videos = load_climb_paths(args.climb_videos_jsonl, "videos", "video")

    def resolve(entry_id: str, source_dataset: str, modality: str) -> str:
        if source_dataset == "pmc_vqa":
            if entry_id.startswith("pmc_vqa_train_"):
                return pmc_train.get(entry_id, "")
            if entry_id.startswith("pmc_vqa_test_"):
                return pmc_test.get(entry_id, "")
            return ""
        if source_dataset == "climb":
            if modality == "image":
                return climb_images.get((entry_id, "image"), "")
            if modality == "video":
                return climb_videos.get((entry_id, "video"), "")
        return ""

    counts: Counter = Counter()
    miss_log_path = args.unmatched_log
    miss_f = open(miss_log_path, "w") if miss_log_path else None

    work_units = [
        ("pmc_vqa", "image"),
        ("climb", "image"),
        ("climb", "video"),
    ]

    for source_dataset, modality in work_units:
        filt = f'source_dataset == "{source_dataset}" and modality == "{modality}"'
        logger.info(f"=== iterating {filt} ===")
        it = client.query_iterator(
            collection_name=args.collection,
            filter=filt,
            batch_size=args.batch_size,
            output_fields=output_fields,
        )
        batch_upserts: list[dict] = []
        t0 = time.time()
        processed_total = 0
        last_log_at = 0
        try:
            stop = False
            while True:
                rows = it.next()
                if not rows:
                    break
                for r in rows:
                    eid = r["entry_id"]
                    existing = r.get("image_path") or ""
                    if existing and not args.force:
                        counts["already_set"] += 1
                        continue
                    path = resolve(eid, source_dataset, modality)
                    if not path:
                        counts["unresolved"] += 1
                        if miss_f:
                            miss_f.write(json.dumps({
                                "entry_id": eid,
                                "source_dataset": source_dataset,
                                "modality": modality,
                            }) + "\n")
                        continue
                    # Build the full-row payload. We re-send every field
                    # we got back from query, with image_path replaced.
                    payload = {f: r.get(f, "") for f in output_fields}
                    payload[pk_field] = r[pk_field]
                    payload[vec_field] = r[vec_field]
                    payload["image_path"] = path
                    batch_upserts.append(payload)
                    counts[f"resolved_{source_dataset}_{modality}"] += 1

                    if len(batch_upserts) >= args.upsert_batch:
                        if not args.dry_run:
                            client.upsert(
                                collection_name=args.collection,
                                data=batch_upserts,
                            )
                        processed_total += len(batch_upserts)
                        batch_upserts = []
                        if processed_total - last_log_at >= args.upsert_batch * 10:
                            rate = processed_total / max(time.time() - t0, 1)
                            logger.info(
                                f"  {source_dataset}/{modality}: "
                                f"upserted {processed_total:,} "
                                f"({rate:,.0f}/s)"
                            )
                            last_log_at = processed_total

                if args.dry_run and (
                    counts[f"resolved_{source_dataset}_{modality}"] >= args.dry_run_limit
                ):
                    logger.info(
                        f"dry_run: stopping after "
                        f"{counts[f'resolved_{source_dataset}_{modality}']} resolved rows"
                    )
                    stop = True
                    break
                if stop:
                    break
        finally:
            it.close()
        if batch_upserts and not args.dry_run:
            client.upsert(collection_name=args.collection, data=batch_upserts)
            processed_total += len(batch_upserts)
            batch_upserts = []
        logger.info(
            f"finished {source_dataset}/{modality}: "
            f"resolved={counts[f'resolved_{source_dataset}_{modality}']:,} "
            f"upserted={processed_total:,}"
        )

    if miss_f:
        miss_f.close()
        logger.info(f"unmatched ids written to {miss_log_path}")

    logger.info("== summary ==")
    for k, v in sorted(counts.items()):
        logger.info(f"  {k}: {v:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milvus_uri", default=os.environ.get(
        "MILVUS_URI", "http://mib.media.mit.edu:19531"))
    ap.add_argument("--milvus_token", default=os.environ.get(
        "MILVUS_TOKEN", "root:Milvus"))
    ap.add_argument("--collection", default="medical_knowledge")
    ap.add_argument(
        "--pmc_vqa_train_csv",
        default="/scratch/self_evolving_datasets/pmc_vqa/train.csv",
        help="Source CSV for PMC-VQA train (entry_id pmc_vqa_train_<N> -> row N).",
    )
    ap.add_argument(
        "--pmc_vqa_test_csv",
        default="/scratch/self_evolving_datasets/pmc_vqa/test.csv",
    )
    ap.add_argument(
        "--climb_images_jsonl",
        default="/scratch/high_modality/geom_train_images.jsonl",
        help="Source JSONL for CLIMB image entries (climb_<N> -> line N).",
    )
    ap.add_argument(
        "--climb_videos_jsonl",
        default="/scratch/high_modality/geom_train_videos.jsonl",
    )
    ap.add_argument("--batch_size", type=int, default=1000,
                    help="Milvus query_iterator batch size (read).")
    ap.add_argument("--upsert_batch", type=int, default=200,
                    help="Number of rows per upsert call (write). Smaller = "
                         "more network round-trips but lower memory; larger = "
                         "faster but risks 32MB gRPC payload limit on Milvus.")
    ap.add_argument("--path_max_length", type=int, default=512)
    ap.add_argument("--unmatched_log", default="/tmp/reindex_unmatched.jsonl")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite image_path even if already set.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Iterate but skip upserts; stop after --dry_run_limit "
                         "rows per (source, modality).")
    ap.add_argument("--dry_run_limit", type=int, default=1000)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    reindex(args)


if __name__ == "__main__":
    main()
