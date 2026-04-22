"""
Prepare JSONL prompt file for Dreaddit (Reddit stress detection).

Task: binary classification — label 1 → "stress", 0 → "no stress".
Uses dreaddit-test.csv only (evaluation split).

Usage:
    python prepare_dreaddit.py \
        --data_dir /path/to/dreaddit \
        --output /path/to/test_dreaddit_prompts.jsonl
"""

import argparse
import csv
import json
import os

TASK_INSTRUCTION = (
    "Based on the following social media post, determine whether the author is experiencing stress. "
    "Respond with exactly one of: stress, no stress."
)

LABEL_MAP = {"0": "no stress", "1": "stress"}


def main(args):
    data_dir = os.path.abspath(args.data_dir)
    csv_path = os.path.join(data_dir, "dreaddit-test.csv")

    entries = []
    skipped = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text      = (row.get("text") or "").strip()
            raw_label = str(row.get("label", "")).strip()
            sample_id = row.get("id", "").strip()

            if not text or raw_label not in LABEL_MAP:
                skipped += 1
                continue

            entries.append({
                "problem":            f'{TASK_INSTRUCTION}\n\nPost: "{text}"',
                "answer":             LABEL_MAP[raw_label],
                "images":             [],
                "audios":             [],
                "videos":             [],
                "texts":              [text],
                "dataset":            "dreaddit",
                "sample_id":          sample_id,
                "modality_signature": "text",
                "ext_video_feats":    [],
                "ext_audio_feats":    [],
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stress    = sum(1 for e in entries if e["answer"] == "stress")
    no_stress = len(entries) - stress
    print(f"Written {len(entries)} entries to {args.output}  (skipped {skipped} invalid rows)")
    print(f"  stress   : {stress}")
    print(f"  no stress: {no_stress}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Path to dreaddit directory containing dreaddit-test.csv")
    parser.add_argument("--output",   required=True, help="Output JSONL path")
    main(parser.parse_args())
