"""
Preprocess PubMedQA dataset to verl JSONL annotation format.

Downloads ori_pqal.json from the PubMedQA GitHub repo and converts it
to train/test JSONL files compatible with verl's RLHFDataset.
"""

import argparse
import json
import os
import random
import urllib.request

PUBMEDQA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"

SYSTEM_PROMPT = (
    "You are a biomedical expert. Read the provided context from a PubMed abstract "
    "and answer the question. Think through the question briefly, then give your "
    "final answer. Keep your reasoning under 500 words. Commit to your reasoning — "
    "do not waver, backtrack, or use hedging phrases like \"wait\", \"actually\", "
    "\"on second thought\", or \"hmm\". Give your best answer directly. The final "
    "answer MUST BE one of: yes, no, or maybe, wrapped in \\boxed{}. The boxed "
    "answer is REQUIRED — do not omit it. Example: \\boxed{yes}"
)


def download_pubmedqa(cache_dir: str) -> dict:
    """Download ori_pqal.json from GitHub and return parsed JSON."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "ori_pqal.json")

    if os.path.exists(cache_path):
        print(f"Using cached file: {cache_path}")
    else:
        print(f"Downloading PubMedQA from {PUBMEDQA_URL}...")
        urllib.request.urlretrieve(PUBMEDQA_URL, cache_path)
        print(f"Saved to {cache_path}")

    with open(cache_path) as f:
        return json.load(f)


def convert_example(pubmed_id: str, example: dict, index: int, split: str) -> dict:
    """Convert one PubMedQA example to verl annotation format."""
    abstract = "\n".join(
        f"[{label}] {text}"
        for label, text in zip(example["LABELS"], example["CONTEXTS"])
    )
    question = example["QUESTION"]
    answer = example["final_decision"].lower()  # yes/no/maybe

    user_content = (
        f"Context:\n{abstract}\n\n"
        f"Question: {question}\n\n"
        "Based on the context above, answer yes, no, or maybe."
    )

    return {
        "data_source": "pubmedqa",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "index": index,
            "split": split,
            "pubmed_id": pubmed_id,
            "question": question,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess PubMedQA for verl")
    parser.add_argument(
        "--output_dir",
        default="/scratch/self_evolving_datasets/pubmedqa",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--cache_dir",
        default="/tmp/pubmedqa_cache",
        help="Cache directory for downloaded files",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for test set",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    raw_data = download_pubmedqa(args.cache_dir)
    print(f"Loaded {len(raw_data)} PubMedQA examples")

    # Shuffle and split
    pubmed_ids = sorted(raw_data.keys())
    random.seed(args.seed)
    random.shuffle(pubmed_ids)

    n_test = int(len(pubmed_ids) * args.test_ratio)
    test_ids = set(pubmed_ids[:n_test])
    train_ids = pubmed_ids[n_test:]

    # Convert
    train_examples = []
    for i, pid in enumerate(train_ids):
        train_examples.append(convert_example(pid, raw_data[pid], i, "train"))

    test_examples = []
    for i, pid in enumerate(sorted(test_ids)):
        test_examples.append(convert_example(pid, raw_data[pid], i, "test"))

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    for path, examples in [(train_path, train_examples), (test_path, test_examples)]:
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(examples)} examples to {path}")

    print(f"\nDone! Train: {len(train_examples)}, Test: {len(test_examples)}")


if __name__ == "__main__":
    main()
