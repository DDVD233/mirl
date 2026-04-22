"""
Prepare JSONL prompt file for SarcNet (multimodal sarcasm detection: image + text).

Task: binary classification — Multi_label 1 → "sarcasm", 0 → "no sarcasm".
Uses SarcNetTest.csv only (evaluation split).

Usage:
    python prepare_sarcnet.py \
        --data_dir /path/to/SarcNet\ Image-Text \
        --output /path/to/test_sarcnet_prompts.jsonl
"""

import argparse
import csv
import json
import os

TASK_INSTRUCTION = (
    "Based on the image and the accompanying text, determine whether the content is sarcastic. "
    "Respond with exactly one of: sarcasm, no sarcasm."
)

LABEL_MAP = {"0": "no sarcasm", "1": "sarcasm"}


def main(args):
    data_dir = os.path.abspath(args.data_dir)
    csv_path = os.path.join(data_dir, "SarcNetTest.csv")
    img_dir  = os.path.join(data_dir, "Image")

    entries = []
    skipped_missing = 0
    skipped_label   = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_filename = row["Imagepath"].strip()
            text         = row["Text"].strip()
            raw_label    = str(row["Multi_label"]).strip()

            if raw_label not in LABEL_MAP:
                skipped_label += 1
                continue

            img_path = os.path.join(img_dir, img_filename)
            if not os.path.exists(img_path):
                skipped_missing += 1
                continue

            rel_img = os.path.relpath(img_path, os.path.dirname(data_dir))

            entries.append({
                "problem":            f'<image>\nText: "{text}"\n\n{TASK_INSTRUCTION}',
                "answer":             LABEL_MAP[raw_label],
                "images":             [rel_img],
                "audios":             [],
                "videos":             [],
                "texts":              [text],
                "dataset":            "sarcnet",
                "sample_id":          img_filename,
                "modality_signature": "text_image",
                "ext_video_feats":    [],
                "ext_audio_feats":    [],
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    sarcasm    = sum(1 for e in entries if e["answer"] == "sarcasm")
    no_sarcasm = len(entries) - sarcasm
    print(f"Written {len(entries)} entries to {args.output}  "
          f"(skipped {skipped_missing} missing images, {skipped_label} invalid labels)")
    print(f"  sarcasm   : {sarcasm}")
    print(f"  no sarcasm: {no_sarcasm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Path to 'SarcNet Image-Text' directory")
    parser.add_argument("--output",   required=True, help="Output JSONL path")
    main(parser.parse_args())
