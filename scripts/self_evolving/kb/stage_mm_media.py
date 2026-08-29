"""Stage a bounded, balanced slice of the CLIMB image corpus for multimodal minting.

WHY STAGING AND NOT A FILE SERVER. The images live only on mib
(`/scratch/high_modality`, 2.1 TB). The GPU pods mount a different `/scratch` (the
shared NFS PVC) and cannot see mib's disk, and mib's firewall only exposes 18001 /
19531 / 22 to them -- the CLIMB media file server on 18080 is unreachable, verified
2026-08-29. Rather than run a tunnel that has to be babysat, we copy the images we
actually intend to mint from onto the shared NFS ONCE. Then every consumer -- gen
server, trainer, judge, referee -- reads an ordinary absolute path, exactly like the
MedXpertQA MM rows that are already proven to work end to end. No runtime dependency
on mib, nothing to keep alive, nothing to fail mid-run.

WHAT IT WRITES. Under `--out_dir` (staged locally on mib, then rsynced to the pods):
  * `images/<relpath>`  -- the image, downscaled to `--max_pixels` and re-encoded,
    which is what keeps a 2.1 TB corpus down to a few GB and matches the resolution
    the student actually trains at;
  * `manifest.jsonl`    -- one row per staged image: relpath, the clinical question
    CLIMB ships with it, the ground-truth finding, modality and dataset. The gen
    server samples THIS file, so minting never has to query Milvus for an image and
    can never name a file that was not copied.

Balanced by `data_source` (the clinical modality) because the corpus is not: chest
X-ray alone is 1.1 TB of the 2.1 TB, and an unbalanced sample would mint almost
nothing but chest films.

Run on mib:
    python scripts/self_evolving/kb/stage_mm_media.py \
        --out_dir /scratch/climb_mm_stage --per_modality 3000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SRC = "/scratch/high_modality"
DEFAULT_MANIFEST = "geom_train.jsonl"


def _reservoir_sample_by_modality(src_jsonl: Path, per_modality: int, seed: int,
                                  max_scan: int) -> dict[str, list[dict]]:
    """One streaming pass, reservoir sampling per modality.

    The manifest is ~1 GB / 2M rows, so it is read once and never held in memory.
    Reservoir (not head-N) because the file is grouped by dataset: taking the first
    N of each modality would take one dataset per modality and call it diversity.
    """
    rng = random.Random(seed)
    seen: Counter = Counter()
    keep: dict[str, list[dict]] = defaultdict(list)
    with src_jsonl.open() as fh:
        for i, line in enumerate(fh):
            if max_scan and i >= max_scan:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            imgs = row.get("images") or []
            if not imgs or row.get("videos"):
                continue  # images only; video rows are a different pipeline
            mod = str(row.get("data_source") or "other")
            seen[mod] += 1
            bucket = keep[mod]
            if len(bucket) < per_modality:
                bucket.append(row)
            else:
                j = rng.randrange(seen[mod])
                if j < per_modality:
                    bucket[j] = row
    print(f"[scan] rows by modality: {dict(seen.most_common())}", flush=True)
    return keep


def _stage_one(src_root: Path, out_images: Path, rel: str, max_pixels: int,
               quality: int) -> tuple[bool, str]:
    """Copy one image, downscaled. Returns (ok, reason)."""
    from PIL import Image

    src = src_root / rel
    if not src.is_file():
        return False, "missing"
    dst = out_images / rel
    if dst.is_file() and dst.stat().st_size > 0:
        return True, "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w * h > max_pixels:
                scale = (max_pixels / float(w * h)) ** 0.5
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.BICUBIC)
            # Always re-encode to JPEG: the corpus mixes png/jpg/dcm-derived files and
            # a uniform encoding keeps the staged tree predictable in size.
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            im.save(tmp, format="JPEG", quality=quality)
            os.replace(tmp, dst)
    except Exception as e:
        return False, f"{type(e).__name__}"
    return True, "staged"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_root", default=DEFAULT_SRC)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="CLIMB jsonl under src_root to sample from")
    ap.add_argument("--out_dir", required=True, help="staging dir on THIS machine")
    ap.add_argument("--per_modality", type=int, default=3000)
    ap.add_argument("--max_pixels", type=int, default=1048576, help="1024x1024")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_scan", type=int, default=0, help="cap rows scanned (0=all)")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    src_jsonl = src_root / args.manifest
    if not src_jsonl.is_file():
        raise SystemExit(f"no such manifest: {src_jsonl}")

    out_dir = Path(args.out_dir)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    picked = _reservoir_sample_by_modality(src_jsonl, args.per_modality, args.seed,
                                           args.max_scan)

    rows_out, stats = [], Counter()
    for mod, rows in sorted(picked.items()):
        for row in rows:
            rels = [str(r) for r in (row.get("images") or [])]
            ok_all, reason = True, ""
            for rel in rels:
                ok, reason = _stage_one(src_root, out_images, rel, args.max_pixels,
                                        args.quality)
                if not ok:
                    ok_all = False
                    break
            stats[reason if not ok_all else "ok"] += 1
            if not ok_all:
                continue
            rows_out.append({
                "images": rels,
                "question": row.get("problem", ""),
                "answer": row.get("answer", ""),
                "modality": mod,
                "dataset": row.get("dataset", ""),
            })
        print(f"[stage] {mod}: kept {sum(1 for r in rows_out if r['modality'] == mod)}",
              flush=True)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")

    total_bytes = sum(p.stat().st_size for p in out_images.rglob("*") if p.is_file())
    print(f"\nstaged {len(rows_out)} rows -> {manifest_path}")
    print(f"outcomes: {dict(stats)}")
    print(f"images on disk: {total_bytes / 1e9:.2f} GB under {out_images}")
    by_mod = Counter(r["modality"] for r in rows_out)
    print(f"by modality: {dict(by_mod.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
