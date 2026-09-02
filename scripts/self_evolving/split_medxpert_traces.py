"""Combine the MedXpertQA trace shards and split them for stage-1 SFT.

Lives in the repo (on the shared NFS) rather than in /root on a pod: pod 2336 was
recreated mid-run on 2026-09-01 and took every local helper script with it, while
everything under /scratch survived untouched.

HELD OUT BY QUESTION, not by row. With more than one trace per question a random row
split puts two traces of the SAME question on both sides, and val loss then measures
memorisation of a question the model just trained on.

FILTERS the small tail of defective traces the generation gates cannot see: a stray
nested tag, or a "Final answer:" that sits INSIDE the reasoning so the visible answer
has none. Measured at 0.37% of the corpus; both teach a broken output shape.

    python scripts/self_evolving/split_medxpert_traces.py \
        --dir /scratch/sheng/self_evolving/medxpert_sft
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import sys

TH = re.compile(r"<think>(.*?)</think>", re.S | re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/scratch/sheng/self_evolving/medxpert_sft")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--min_val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    S = args.dir
    files = [os.path.join(S, "traces.jsonl")] + sorted(glob.glob(os.path.join(S, "shards", "traces_*.jsonl")))
    rows = []
    for f in files:
        if not os.path.isfile(f):
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    print(f"read {len(rows)} raw traces from {len(files)} files")

    seen, uniq = set(), []
    dropped = collections.Counter()
    for r in rows:
        key = (r["extra_info"]["question_id"], r["reference_response"][:200])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        t = r["reference_response"]
        m = TH.search(t)
        if not m:
            dropped["no_think_block"] += 1
            continue
        think, after = m.group(1), t[m.end():].strip()
        if "<think>" in think or "</think>" in think:
            dropped["nested_tags"] += 1
            continue
        if not re.search(r"final answer:", after, re.I):
            dropped["no_final_answer_after_block"] += 1
            continue
        if len(think.split()) < 50:
            dropped["think_too_short"] += 1
            continue
        uniq.append(r)

    qids = sorted({r["extra_info"]["question_id"] for r in uniq})
    rng = random.Random(args.seed)
    rng.shuffle(qids)
    n_val = max(args.min_val, int(args.val_frac * len(qids)))
    val_q = set(qids[:n_val])

    train = [r for r in uniq if r["extra_info"]["question_id"] not in val_q]
    val = [r for r in uniq if r["extra_info"]["question_id"] in val_q]
    random.Random(args.seed + 1).shuffle(train)

    for name, part in (("traces_train", train), ("traces_val", val)):
        out = os.path.join(S, f"{name}.jsonl")
        with open(out, "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(part):6d} -> {out}")

    print(f"dropped: {dict(dropped) if dropped else 'none'}")
    print(f"kept {len(uniq)} traces over {len(qids)} questions "
          f"(val holds out {len(val_q)} questions)")
    print(f"train by kind:   {dict(collections.Counter(r['extra_info']['kind'] for r in train))}")
    print(f"train by source: {dict(collections.Counter(r['data_source'] for r in train).most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
