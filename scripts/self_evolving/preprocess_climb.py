"""
Preprocess the CLIMB multimodal medical dataset into verl-shaped JSONL.

CLIMB rows (``/scratch/high_modality/geom_*.jsonl``) look like::

    {"problem": "<image>\\nAbove is a chest X-ray ... Answer with ...:\\nNo Finding\\n...",
     "answer": "Pleural Effusion, Support Devices",
     "images": ["chest_xray/chexpert_full/.../view1_frontal.jpg"],
     "videos": [],
     "data_source": "chest_xray", "dataset": "chexpert_full"}

This script emits two products:

1. **Eval set** (``--split valid``, from ``geom_valid_mini.jsonl``) →
   ``<out_dir>/val_mini.jsonl``: the static validation set the trainer scores
   each val step. ``data_source`` becomes ``climb_<modality>`` so verl groups
   per-modality metrics automatically, and the CLIMB metric reducer computes
   class-macro F1 + a macro-average across modalities.

2. **Train seeds** (``--split train``, from ``geom_train.jsonl``) →
   ``<out_dir>/train_seeds.jsonl``, **balanced per modality** (CLIMB is ~70%
   chest_xray otherwise). The generation server consumes these both as
   direct-insert real training rows and as targets for synthesizing new
   multimodal questions.

Media is referenced as ``climb://<relpath>`` handles (NOT on-disk paths): the
trainer node resolves them on demand from the local CLIMB file server (see
``verl/utils/climb.py``), so the produced JSONL is portable to the remote GPU
nodes that have no access to ``/scratch/high_modality``.
"""

import argparse
import json
import os
import random
import re
import sys
import uuid
from collections import defaultdict

# Allow running as a plain script (python scripts/self_evolving/preprocess_climb.py)
# from anywhere by putting the repo root on sys.path before importing verl.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from verl.utils.climb import CLIMB_SCHEME, strip_high_modality

# Student solver prompt for CLIMB classification. Demands the exact option
# phrase(s) in \boxed{} so both the reward scorer and the metric reducer can
# parse the prediction with `extract_boxed_answer` + `parse_label_set`.
CLIMB_SOLVER_SYSTEM_PROMPT = (
    "You are an expert medical imaging diagnostician. Carefully examine the "
    "provided medical image(s) or video and answer the question by selecting "
    "from the listed options. Some cases have MORE THAN ONE correct label — "
    "include every label that applies, separated by commas. Think through the "
    "salient findings briefly (under 300 words), commit to your reasoning "
    "without hedging, then output ONLY your final answer wrapped in \\boxed{} "
    "using the EXACT option phrase(s) as written. The boxed answer is REQUIRED. "
    "Example: \\boxed{Pleural Effusion, Support Devices}"
)

_OPTIONS_HEADER_RE = re.compile(r"following:?\s*\n", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"<image>|<video>")
_LEADIN_RE = re.compile(r"^\s*(above|here)\s+(is|are)\b.*?(?:\.|:|\n)", re.IGNORECASE)


def parse_options(problem: str) -> list:
    """Extract the answer option phrases listed after 'following:' in the
    problem text. Returns [] when the question is not a closed-option prompt."""
    m = _OPTIONS_HEADER_RE.search(problem)
    if not m:
        return []
    tail = problem[m.end():]
    opts = [ln.strip() for ln in tail.splitlines()]
    return [o for o in opts if o]


def clean_question(problem: str) -> str:
    """Strip the <image>/<video> placeholder and a leading 'Above is ...'
    boilerplate for use as a retrieval/generation query (NOT for the served
    prompt, which keeps the placeholder)."""
    q = _PLACEHOLDER_RE.sub("", problem).strip()
    q = _LEADIN_RE.sub("", q, count=1).strip()
    return q


def to_climb_refs(rels: list) -> list:
    """Turn relative media paths into portable climb:// handles."""
    return [f"{CLIMB_SCHEME}{strip_high_modality(r)}" for r in (rels or []) if r]


def build_entry(rec: dict, split: str, index: int) -> dict | None:
    problem = rec.get("problem") or ""
    answer = str(rec.get("answer") or "").strip()
    modality = rec.get("data_source") or "unknown"
    images = rec.get("images") or []
    videos = rec.get("videos") or []
    if not problem or not answer or (not images and not videos):
        return None

    n_ph = len(_PLACEHOLDER_RE.findall(problem))
    if n_ph != len(images) + len(videos):
        # Placeholder/media count must match what RLHFDataset consumes in order.
        return None

    entry = {
        "data_source": f"climb_{modality}",
        "prompt": [
            {"role": "system", "content": CLIMB_SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "images": to_climb_refs(images),
        "videos": to_climb_refs(videos),
        "reward_model": {"style": "climb", "ground_truth": answer},
        "extra_info": {
            "question_id": uuid.uuid4().hex,
            "index": index,
            "split": split,
            "source": "climb",
            "modality": modality,
            "dataset": rec.get("dataset", ""),
            "question": clean_question(problem),
            "options": parse_options(problem),
        },
    }
    return entry


def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--climb_root", default="/scratch/high_modality")
    ap.add_argument("--split", choices=["train", "valid"], required=True)
    ap.add_argument("--src", default="", help="override source jsonl path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_modality", type=int, default=2000,
                    help="train: max seeds per clinical modality (balancing cap)")
    ap.add_argument("--limit", type=int, default=-1, help="hard cap on total rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    if args.src:
        src = args.src
    elif args.split == "valid":
        src = os.path.join(args.climb_root, "geom_valid_mini.jsonl")
    else:
        src = os.path.join(args.climb_root, "geom_train.jsonl")
    if not os.path.isfile(src):
        raise SystemExit(f"source not found: {src}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = "val_mini.jsonl" if args.split == "valid" else "train_seeds.jsonl"
    out_path = os.path.join(args.out_dir, out_name)

    by_modality = defaultdict(list)
    kept = skipped = 0
    for i, rec in enumerate(iter_jsonl(src)):
        entry = build_entry(rec, args.split, i)
        if entry is None:
            skipped += 1
            continue
        by_modality[entry["extra_info"]["modality"]].append(entry)
        kept += 1

    # Balance for train seeds; keep eval set complete (no per-modality cap).
    rows = []
    for mod, entries in by_modality.items():
        if args.split == "train" and args.per_modality > 0 and len(entries) > args.per_modality:
            entries = random.sample(entries, args.per_modality)
        rows.extend(entries)
    random.shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    # Reindex after shuffling/capping.
    for j, r in enumerate(rows):
        r["extra_info"]["index"] = j

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} rows -> {out_path} (kept={kept} skipped={skipped})")
    print("per-modality counts:")
    final = defaultdict(int)
    for r in rows:
        final[r["extra_info"]["modality"]] += 1
    for mod in sorted(final):
        print(f"  climb_{mod:14s} {final[mod]}")


if __name__ == "__main__":
    main()
