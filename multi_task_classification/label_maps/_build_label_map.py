#!/usr/bin/env python3
# multi_task_classification/build_label_map.py
# Build a dataset-specific label mapping and save to JSON.

import json
import os
import re
import sys
import gzip
from collections import defaultdict

# ------------------
# Hardcoded paths SCRATCH
# ------------------
# INPUT_JSONLS = [
#     "/scratch/keane/human_behaviour/human_behaviour_data/cleaned_full_train.jsonl"
# ]
# OUTPUT_JSON = "/home/keaneong/human-behavior/verl/multi_task_classification/cleaned_full_label_map.json"


# Hardcoded paths LOCAL
INPUT_JSONLS = [
    "/Users/keane/Desktop/research/human-behavior/data/unified_scheme/unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.jsonl"
]
OUTPUT_JSON = "/Users/keane/Desktop/research/human-behavior/data/unified_scheme/label_maps/v2_non_unified_scheme_splitmmpsy_binarymmpsy_no_vptd_chalearn_lmvd_esconv.json"


# ------------------------------------


# ------------------
# Helpers
# ------------------
SAFE_RE = re.compile(r"[^a-z0-9_]+")
def open_maybe_gzip(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

def read_pairs(paths):
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] File not found: {p}", file=sys.stderr)
            continue
        with open_maybe_gzip(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ds = obj.get("dataset")
                ans = obj.get("answer")
                if ds is None or ans is None:
                    continue
                yield ds, ans

# ------------------
# Main
# ------------------
def main():
    pairs = list(read_pairs(INPUT_JSONLS))
    if not pairs:
        print("[ERROR] No valid pairs found.", file=sys.stderr)
        sys.exit(1)

    # collect unique per-dataset answers
    per_ds = defaultdict(list)
    seen = set()
    ordered = []
    for ds, ans in pairs:
        if (ds, ans) not in seen:
            seen.add((ds, ans))
            per_ds[ds].append(ans)
            ordered.append((ds, ans))

    # sort within each dataset for consistency
    for ds in per_ds:
        per_ds[ds] = sorted(per_ds[ds])

    # build mapping
    label_mapping = {}
    idx = 0
    for ds, ans in sorted(ordered, key=lambda x: (x[0], x[1])):
        label = f"{ds}_{ans}"
        if label not in label_mapping:
            label_mapping[label] = idx
            idx += 1

    output = {
        "label_mapping": label_mapping,
        "dataset_labels": dict(per_ds),
        "num_classes": len(label_mapping),
        "datasets": sorted(list(per_ds.keys())),
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[OK] wrote {len(label_mapping)} classes to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
