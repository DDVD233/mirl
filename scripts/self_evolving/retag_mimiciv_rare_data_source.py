"""Re-tag the `data_source` of an existing MIMIC-IV rare-disease annotation file
into top-level ICD-10 chapter groups, so verl reports set-wise validation
metrics (one accuracy per group) instead of a single global number.

This avoids re-running the full (expensive, image-rendering) preprocessing in
preprocess_mimiciv_rare.py: it only rewrites the `data_source` field and adds
`extra_info.icd_category`, deriving the category from the already-present
`extra_info.primary_icd10`. The categorization logic is imported from
preprocess_mimiciv_rare.icd10_category so the two stay in sync.

verl groups validation metrics by `data_source` (see
process_validation_metrics + RayPPOTrainer._val_metrics_update), and the
reward router dispatches on `data_source.startswith("mimic")`, so the new
"mimic_rare/<category>" values keep scoring through the same medical-text
scorer while splitting metrics per chapter.

Usage:
  python scripts/self_evolving/retag_mimiciv_rare_data_source.py \
      /scratch/self_evolving_datasets/mimiciv_rare/train.jsonl \
      /scratch/self_evolving_datasets/mimiciv_rare/test.jsonl

Each input is rewritten in place after a one-time ".before_catsplit" backup
(skipped if a backup already exists, so re-runs are idempotent).
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_mimiciv_rare import icd10_category  # noqa: E402


def retag_file(path: str) -> Counter:
    backup = path + ".before_catsplit"
    if not os.path.exists(backup):
        os.replace(path, backup)
        read_from = backup
    else:
        # Backup already present (re-run): re-derive from the pristine original.
        read_from = backup

    counts: Counter = Counter()
    n_missing = 0
    tmp = path + ".tmp"
    with open(read_from) as fin, open(tmp, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            ei = e.get("extra_info") or {}
            code = ei.get("primary_icd10")
            if not code:
                n_missing += 1
            cat = icd10_category(code)
            e["data_source"] = cat
            ei["icd_category"] = cat
            e["extra_info"] = ei
            counts[cat] += 1
            fout.write(json.dumps(e) + "\n")
    os.replace(tmp, path)

    total = sum(counts.values())
    print(f"\n{path}  (n={total}, backup={os.path.basename(backup)})")
    if n_missing:
        print(f"  WARNING: {n_missing} rows had no extra_info.primary_icd10 -> other")
    for cat, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {c:5d} ({100 * c / total:5.1f}%)  {cat}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="annotation jsonl file(s) to re-tag in place")
    args = ap.parse_args()
    grand: Counter = Counter()
    for f in args.files:
        grand.update(retag_file(f))
    if len(args.files) > 1:
        total = sum(grand.values())
        print(f"\n=== combined (n={total}) ===")
        for cat, c in sorted(grand.items(), key=lambda x: -x[1]):
            print(f"  {c:5d} ({100 * c / total:5.1f}%)  {cat}")


if __name__ == "__main__":
    main()
