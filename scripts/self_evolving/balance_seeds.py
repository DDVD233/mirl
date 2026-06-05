"""Cap a self-evolving seed JSONL to N samples per diagnosis (ICD code).

The mimiciv_rare seeds are heavily label-imbalanced — ~6% unique diagnoses, with
a few common codes (E10.10 x341, C22.0 x260, ...) dominating thousands of rows.
The gen server samples "direct" questions from these seeds, so the student
over-trains on a handful of common diagnoses and never sees enough of the rare
ones. This caps each diagnosis to --cap rows (keeping all rows of rare codes,
downsampling common ones), flattening the training distribution.

Only the TRAIN seeds should be balanced — leave the val/test set as-is so the
held-out metric still reflects the real distribution.

Usage:
    python scripts/self_evolving/balance_seeds.py \\
        --in  /scratch/sheng/self_evolving/mimiciv_rare/train.jsonl \\
        --out /scratch/sheng/self_evolving/mimiciv_rare/train_balanced.jsonl \\
        --cap 25
"""

import argparse
import json
import random
import re
from collections import defaultdict

ICD_CODE_RE = re.compile(r"([A-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?)")


def diagnosis_key(rec: dict) -> str:
    """Group key = the ICD code of the ground-truth diagnosis (text fallback)."""
    gt = str(rec.get("reward_model", {}).get("ground_truth", "") or rec.get("gts", ""))
    m = ICD_CODE_RE.search(gt)
    return m.group(1) if m else gt.strip()[:40].lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=25, help="max rows per diagnosis code")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    by_code: dict[str, list[str]] = defaultdict(list)
    total = 0
    for line in open(a.inp):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        by_code[diagnosis_key(rec)].append(line)

    out: list[str] = []
    for lines in by_code.values():
        rng.shuffle(lines)
        out.extend(lines[: a.cap])
    rng.shuffle(out)

    with open(a.out, "w") as f:
        for line in out:
            f.write(line + "\n")

    capped = sum(1 for v in by_code.values() if len(v) > a.cap)
    print(f"in: {total} rows, {len(by_code)} diagnoses  ->  out: {len(out)} rows  "
          f"(cap={a.cap}; {capped} diagnoses downsampled)")


if __name__ == "__main__":
    main()
