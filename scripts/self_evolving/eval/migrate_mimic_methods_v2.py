"""Reuse unaffected successful generations in a separately versioned recovery run.

Planning methods are never migrated: they must rerun with constrained JSON.
Original files stay untouched; each copied row records its old configuration
and original record hash. The only permissible configuration difference is code.
"""

import argparse
import json
from pathlib import Path

import mimic_method_baselines as baseline


def migrate(source, target, args):
    manifest = json.loads((source / "manifest.json").read_text())
    old_config = manifest["config"]
    new_config = baseline.configuration(args)
    if baseline.digest(old_config) != manifest["config_sha256"]:
        raise ValueError("Source manifest hash mismatch")
    for key in set(old_config) | set(new_config):
        if key != "code_sha256" and old_config.get(key) != new_config.get(key):
            raise ValueError(f"Incompatible migration: {key}")
    if target.exists():
        raise ValueError("Recovery destination must be new")
    target.mkdir(parents=True)
    new_hash = baseline.digest(new_config)
    reused = {}
    for method in ("direct", "medrag", "self_consistency", "self_refine"):
        rows = baseline.read_rows(source / f"{method}.jsonl")
        if any(row["config_sha256"] != manifest["config_sha256"] for row in rows):
            raise ValueError("Source row/manifest mismatch")
        with (target / f"{method}.jsonl").open("w") as output:
            for row in rows:
                origin = {
                    "file": str(source / f"{method}.jsonl"),
                    "config_sha256": row["config_sha256"],
                    "row_sha256": baseline.digest(row),
                }
                row = dict(row, config_sha256=new_hash, image_exceptions=[], reused_from=origin)
                output.write(json.dumps(row) + "\n")
        reused[method] = len(rows)
    inventory = source / "retrieval_inventory.json"
    if inventory.exists():
        (target / inventory.name).write_bytes(inventory.read_bytes())
    baseline.write_json(
        target / "manifest.json",
        {
            "config": new_config,
            "config_sha256": new_hash,
            "migration": {
                "source": str(source),
                "reused": reused,
                "reason": "Unchanged successful calls; failed images audited; planners rerun",
            },
        },
    )
    print(json.dumps({"target": str(target), "reused": reused}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    migration, remaining = parser.parse_known_args()
    import sys

    sys.argv = [sys.argv[0]] + remaining
    args = baseline.parse_args()
    migrate(migration.source, Path(args.out_dir), args)
