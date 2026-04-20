"""
Prepare JSONL prompt file for EATD-Corpus (depression detection from audio).

Task: binary classification — SDS score >= 53 → "Depressed", else "Not Depressed".
Each subject contributes one JSONL entry with three audio paths (positive, neutral, negative
prompt responses). The inference script concatenates them before passing to the model.

Usage:
    python prepare_eatd.py \
        --data_dir /path/to/EATD-Corpus \
        --output /path/to/test_eatd_prompts.jsonl
"""

import argparse
import json
import os

DEPRESSION_THRESHOLD = 53.0

TASK_PROMPT = (
    "<audio>\n<audio>\n<audio>\n"
    "The following three audio clips are speech samples from the same individual "
    "during a clinical interview:\n"
    "  (1) Response to a positive emotional prompt\n"
    "  (2) Response to a neutral prompt\n"
    "  (3) Response to a negative emotional prompt\n\n"
    "Based on the speaker's speech patterns, prosody, and content across all three recordings, "
    "determine whether the individual shows signs of clinical depression "
    "(defined as an SDS score of 53 or above).\n\n"
    "Respond with exactly one of: Depressed, Not Depressed."
)

AUDIO_CLIPS = ["positive.wav", "neutral.wav", "negative.wav"]


def process_subject(subject_dir: str, subject_id: str, data_dir: str) -> dict | None:
    label_path = os.path.join(subject_dir, "new_label.txt")
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        raw = f.read().strip()
    try:
        sds_score = float(raw)
    except ValueError:
        return None

    audio_paths = []
    for clip in AUDIO_CLIPS:
        full = os.path.join(subject_dir, clip)
        if not os.path.exists(full):
            return None
        # Store path relative to data_dir parent so it's portable
        rel = os.path.relpath(full, os.path.dirname(data_dir))
        audio_paths.append(rel)

    answer = "Depressed" if sds_score >= DEPRESSION_THRESHOLD else "Not Depressed"

    return {
        "problem": TASK_PROMPT,
        "answer": answer,
        "audios": audio_paths,
        "images": [],
        "videos": [],
        "texts": [],
        "dataset": "eatd",
        "subject_id": subject_id,
        "sds_score": sds_score,
        "modality_signature": "text_audio",
        "ext_audio_feats": [],
    }


def main(args):
    data_dir = os.path.abspath(args.data_dir)
    entries = []
    skipped = 0

    subject_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("t_")
    )

    for subject_id in subject_dirs:
        subject_dir = os.path.join(data_dir, subject_id)
        entry = process_subject(subject_dir, subject_id, data_dir)
        if entry is None:
            skipped += 1
        else:
            entries.append(entry)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    depressed = sum(1 for e in entries if e["answer"] == "Depressed")
    print(f"Written {len(entries)} entries to {args.output}  (skipped {skipped})")
    print(f"  Depressed: {depressed}  |  Not Depressed: {len(entries) - depressed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to EATD-Corpus directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    main(parser.parse_args())
