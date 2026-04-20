"""
Prepare JSONL prompt file for MVSA (Multimodal Sentiment Analysis: image + text).

Task: 3-class sentiment — positive / neutral / negative.

Filtering rule (consistency):
    Each sample has 3 annotators, each giving (text_sentiment, image_sentiment) → 6 sub-labels.
    Keep only examples where the majority sentiment appears in >= CONSISTENCY_THRESHOLD of those 6.
    Default threshold = 4 (i.e. at least 4/6 annotator sub-labels agree).
    This removes all "three-way splits" (max=2), weak majorities (max=3), and 3-3 ties (max=3).

Stratification:
    After filtering, equal-sized random sampling from each sentiment class (seeded for
    reproducibility), so every class has the same N = min_class_size examples.
    This removes the strong positive-skew in the raw MVSA labels.

Usage:
    python prepare_mvsa.py \
        --data_dir /path/to/MVSA \
        --output /path/to/test_mvsa_prompts.jsonl

    # Disable stratification (keep all consistent examples):
    python prepare_mvsa.py --data_dir ... --output ... --no_stratify

    # Adjust consistency threshold:
    python prepare_mvsa.py --data_dir ... --output ... --consistency_threshold 5
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

# Sentinel: tiebreak order when equal vote counts remain after threshold filter
TIE_PRIORITY = ["positive", "neutral", "negative"]

TASK_INSTRUCTION = (
    "Based on the image and the accompanying text, determine the overall sentiment. "
    "Respond with exactly one of: positive, neutral, negative."
)


def get_label_and_count(votes: list[str]) -> tuple[str, int]:
    """
    Returns (majority_label, majority_count) from 6 sub-label votes.
    Ties are broken by TIE_PRIORITY (positive > neutral > negative).
    """
    counter = Counter(v.strip().lower() for v in votes if v.strip())
    if not counter:
        return "", 0
    top_count = counter.most_common(1)[0][1]
    # Among tied labels, pick by priority
    for pref in TIE_PRIORITY:
        if counter.get(pref, 0) == top_count:
            return pref, top_count
    return counter.most_common(1)[0]


def parse_label_row(row: str) -> tuple[str, list[str]] | None:
    """
    Parse a tab-separated label row.
    Returns (sample_id, [text_s, image_s, text_s, image_s, text_s, image_s]) or None.
    """
    parts = row.strip().split("\t")
    if len(parts) < 4:
        return None
    sample_id = parts[0].strip()
    votes = []
    for annotator_col in parts[1:4]:
        pair = annotator_col.strip().split(",")
        if len(pair) == 2:
            votes.extend([pair[0].strip(), pair[1].strip()])
        elif len(pair) == 1:
            votes.append(pair[0].strip())
    return sample_id, votes


def main(args):
    data_dir   = os.path.abspath(args.data_dir)
    label_file = os.path.join(data_dir, "labelResultAll.txt")
    img_dir    = os.path.join(data_dir, "data")

    entries: list[dict] = []
    skipped_parse   = 0
    skipped_missing = 0
    skipped_incons  = 0

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:     # skip header
        if not line.strip():
            continue

        parsed = parse_label_row(line)
        if parsed is None:
            skipped_parse += 1
            continue

        sample_id, votes = parsed
        img_path = os.path.join(img_dir, f"{sample_id}.jpg")
        txt_path = os.path.join(img_dir, f"{sample_id}.txt")

        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            skipped_missing += 1
            continue

        # ── Consistency filter ────────────────────────────────────────────
        gold_label, majority_count = get_label_and_count(votes)
        if majority_count < args.consistency_threshold:
            skipped_incons += 1
            continue

        with open(txt_path, "r", encoding="utf-8", errors="replace") as tf:
            caption = tf.read().strip()

        rel_img = os.path.relpath(img_path, os.path.dirname(data_dir))

        entries.append({
            "problem": f"<image>\nText: \"{caption}\"\n\n{TASK_INSTRUCTION}",
            "answer":  gold_label,
            "images":  [rel_img],
            "audios":  [],
            "videos":  [],
            "texts":   [caption],
            "dataset": "mvsa",
            "sample_id": sample_id,
            "majority_count": majority_count,   # useful for downstream analysis
            "modality_signature": "text_image",
            "ext_video_feats": [],
            "ext_audio_feats": [],
        })

    # ── Stratification ────────────────────────────────────────────────────
    if args.no_stratify:
        final_entries = entries
    else:
        rng = random.Random(args.seed)

        by_label: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_label[e["answer"]].append(e)

        min_size = min(len(v) for v in by_label.values())
        final_entries = []
        for label in sorted(by_label):             # deterministic order
            sampled = rng.sample(by_label[label], min_size)
            final_entries.extend(sampled)
        rng.shuffle(final_entries)

    # ── Write ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in final_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    label_counts = Counter(e["answer"] for e in final_entries)
    filtered_total = skipped_parse + skipped_missing + skipped_incons

    print(f"Written {len(final_entries)} entries → {args.output}")
    print(f"  Consistency threshold : {args.consistency_threshold}/6 sub-label votes")
    print(f"  Filtered (inconsistent): {skipped_incons}")
    print(f"  Filtered (missing files): {skipped_missing}")
    print(f"  Stratified (equal per class): {'no' if args.no_stratify else 'yes'}")
    print(f"  Seed: {args.seed}")
    print(f"Class distribution:")
    for label in TIE_PRIORITY:
        n = label_counts.get(label, 0)
        print(f"    {label:<10}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to MVSA directory")
    parser.add_argument("--output",   required=True, help="Output JSONL path")
    parser.add_argument(
        "--consistency_threshold", type=int, default=5,
        help="Min votes for majority label out of 6 (default 5). "
             "Examples with fewer are considered ambiguous and removed."
    )
    parser.add_argument(
        "--no_stratify", action="store_true",
        help="Skip equal-class sampling; keep all consistent examples."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling (default 42)."
    )
    main(parser.parse_args())
