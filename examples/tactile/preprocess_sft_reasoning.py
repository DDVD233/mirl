"""Convert annotation_verl_reasoning.json (JSONL) to parquet for verl SFT training.

Produces:
  - Parquet files with `messages` + `videos` columns for MultiTurnSFTDataset
  - RL-format JSONL test files (no reasoning) for generation-based evaluation

Splits are derived from existing GRPO annotation split files (annotation_verl_split_*),
using (video_path, user_content) as the identifier to ensure consistency and no leakage.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
SPLIT_NAMES = ["date", "participant", "glove", "question", "task"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_key(sample):
    """Return (video_path, user_content) as a unique identifier for a sample."""
    video = sample["videos"][0]["video"] if sample.get("videos") else ""
    user_content = ""
    for m in sample["prompt"]:
        if m["role"] == "user":
            user_content = m["content"]
            break
    return (video, user_content)


def read_jsonl(path):
    """Read JSONL file, return list of dicts."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def read_reasoning_jsonl(path):
    """Read JSONL file, filter to reasoning_status == 'ok'."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("reasoning_status") != "ok":
                continue
            samples.append(d)
    return samples


def load_split_keys(grpo_dir, split_name):
    """Load train/test keys from existing GRPO split files."""
    train_file = grpo_dir / f"annotation_verl_split_{split_name}_train.json"
    test_file = grpo_dir / f"annotation_verl_split_{split_name}_test.json"

    train_keys = set()
    if train_file.exists():
        for s in read_jsonl(train_file):
            train_keys.add(sample_key(s))

    test_keys = set()
    if test_file.exists():
        for s in read_jsonl(test_file):
            test_keys.add(sample_key(s))

    # Also load mini test keys
    mini_file = grpo_dir / f"annotation_verl_split_{split_name}_test_mini.json"
    mini_keys = set()
    if mini_file.exists():
        for s in read_jsonl(mini_file):
            mini_keys.add(sample_key(s))

    return train_keys, test_keys, mini_keys


MAX_FRAMES = 8


def cap_max_frames(videos):
    """Cap max_frames in video metadata."""
    return [{**v, "max_frames": min(v.get("max_frames", MAX_FRAMES), MAX_FRAMES)} for v in videos]


def to_sft_record(sample):
    """Convert a reasoning sample to SFT parquet record (messages + videos)."""
    prompt_messages = sample["prompt"]  # [system, user]
    reasoning = sample["reasoning"]

    messages = list(prompt_messages) + [{"role": "assistant", "content": reasoning}]

    videos = cap_max_frames(sample.get("videos", []))

    return {"messages": messages, "videos": videos}


def to_rl_record(sample):
    """Convert a reasoning sample to RL-format JSONL record (no reasoning)."""
    return {
        "data_source": sample["data_source"],
        "prompt": sample["prompt"],
        "videos": cap_max_frames(sample.get("videos", [])),
        "reward_model": sample["reward_model"],
        "extra_info": sample["extra_info"],
    }


def save_parquet(records, path):
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), index=False)
    print(f"  {path.name}: {len(records)} samples")


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(records)} samples")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess reasoning JSONL → parquet for SFT + JSONL for eval")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "scripts" / "annotation_verl_reasoning.json"),
        help="Path to annotation_verl_reasoning.json (JSONL)",
    )
    parser.add_argument(
        "--grpo_dir",
        type=str,
        default=None,
        help="Directory containing GRPO split files (annotation_verl_split_*). Default: same as input file directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    grpo_dir = Path(args.grpo_dir) if args.grpo_dir else input_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    rng = random.Random(SEED)

    # Read and filter
    samples = read_reasoning_jsonl(input_path)
    print(f"Loaded {len(samples)} valid reasoning samples from {input_path}")

    # ---------- Write all-samples file ----------
    all_sft = [to_sft_record(s) for s in samples]
    save_parquet(all_sft, output_dir / "annotation_verl_reasoning_sft.parquet")
    write_jsonl(all_sft, output_dir / "annotation_verl_reasoning_sft.json")

    # ---------- Process each split ----------
    for split_name in SPLIT_NAMES:
        print(f"\n{split_name.capitalize()} split:")

        train_keys, test_keys, mini_keys = load_split_keys(grpo_dir, split_name)
        if not train_keys and not test_keys:
            print(f"  WARNING: No GRPO split files found for '{split_name}' in {grpo_dir}, skipping.")
            continue

        train_samples = []
        test_samples = []
        unmatched = []
        for s in samples:
            k = sample_key(s)
            if k in train_keys:
                train_samples.append(s)
            elif k in test_keys:
                test_samples.append(s)
            else:
                unmatched.append(s)

        if unmatched:
            print(f"  WARNING: {len(unmatched)} samples not found in GRPO splits, assigning to train.")
            train_samples.extend(unmatched)

        # SFT parquets + JSON mirrors
        prefix = f"annotation_verl_reasoning_sft_split_{split_name}"
        train_sft = [to_sft_record(s) for s in train_samples]
        test_sft = [to_sft_record(s) for s in test_samples]
        save_parquet(train_sft, output_dir / f"{prefix}_train.parquet")
        save_parquet(test_sft, output_dir / f"{prefix}_test.parquet")
        write_jsonl(train_sft, output_dir / f"{prefix}_train.json")
        write_jsonl(test_sft, output_dir / f"{prefix}_test.json")

        # Mini test: use the same mini keys from GRPO if available
        if mini_keys:
            mini_samples = [s for s in test_samples if sample_key(s) in mini_keys]
        elif test_samples:
            mini_samples = rng.sample(test_samples, max(1, len(test_samples) // 10))
        else:
            mini_samples = []

        if mini_samples:
            mini_sft = [to_sft_record(s) for s in mini_samples]
            save_parquet(mini_sft, output_dir / f"{prefix}_test_mini.parquet")
            write_jsonl(mini_sft, output_dir / f"{prefix}_test_mini.json")

        # RL-format JSONL for eval (test only)
        rl_prefix = f"annotation_verl_reasoning_split_{split_name}"
        write_jsonl([to_rl_record(s) for s in test_samples], output_dir / f"{rl_prefix}_test.json")

        if mini_samples:
            write_jsonl([to_rl_record(s) for s in mini_samples], output_dir / f"{rl_prefix}_test_mini.json")

        print(f"  Split '{split_name}': {len(train_samples)} train, {len(test_samples)} test, {len(mini_samples)} mini")

    print(f"\nDone. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
