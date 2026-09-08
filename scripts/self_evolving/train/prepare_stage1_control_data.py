"""Resolve historical image roots in isolated copies; fail on missing image files."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, image_root):
    old = Path("/scratch/self_evolving_datasets/mimiciv_rare")
    manifest = {}
    for split, expected in (("train", 5719), ("test", 2452)):
        path = source / (split + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != expected or len({r["extra_info"]["hadm_id"] for r in rows}) != expected:
            raise ValueError(f"Incomplete {split} dataset")
        paths, changed = set(), 0
        for row in rows:
            for item in row.get("images", []):
                image = Path(item["image"])
                if image.is_relative_to(old):
                    image = image_root / image.relative_to(old)
                    item["image"] = str(image)
                    changed += 1
                paths.add(image)
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise ValueError(f"{split}: {len(missing)} missing images; first: {missing[0]}")
        target = output / path.name
        rendered = "".join(json.dumps(row) + "\n" for row in rows)
        if target.exists() and target.read_text() != rendered:
            raise ValueError("Prepared input changed; use a new output directory")
        target.write_text(rendered)
        manifest[split] = {
            "n": expected,
            "source_sha256": sha(path),
            "prepared_sha256": sha(target),
            "remapped_image_references": changed,
            "unique_images": len(paths),
            "missing_images": 0,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/scratch/sheng/self_evolving/mimiciv_rare"))
    parser.add_argument(
        "--output", type=Path, default=Path("/scratch/sheng/self_evolving/logs_stage1_9b_control/data_images_fixed")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.source, args.output, args.source)


if __name__ == "__main__":
    main()
