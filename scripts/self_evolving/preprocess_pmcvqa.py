"""
Preprocess PMC-VQA for unsupervised self-evolving training.

Training split: question text only, no image context, no labels. This simulates
the unsupervised case — the dataset only knows the question; the proposer
generates new related questions; the model learns via RL reward.

Validation split: full image + question + options + ground-truth label, used to
track how training on self-generated questions improves the model's answer
on the original PMC-VQA questions.

Output JSONL format (verl-compatible):
  Training (masked):
    {data_source, prompt[system, user], reward_model{style:"rule_mcq", ground_truth:""},
     extra_info{index, split:"train", question, format:"mcq"}}

  Validation (full):
    {data_source, prompt[system, user(with image)],
     reward_model{style:"rule_mcq", ground_truth:"A/B/C/D"},
     extra_info{index, split:"test", question, options, format:"mcq",
                image_path, has_image:true}}
"""

import argparse
import csv
import json
import os
import random


PMC_VQA_SYSTEM_PROMPT_MCQ = (
    "You are a medical expert. Read the question carefully and choose the best answer. "
    "Think through the question briefly, then give your final answer. Keep your reasoning "
    "under 500 words. Commit to your reasoning — do not waver, backtrack, or use hedging "
    "phrases like \"wait\", \"actually\", \"on second thought\", or \"hmm\". Give your "
    "best answer directly. The final answer MUST BE a single letter (A, B, C, or D) "
    "wrapped in \\boxed{}. The boxed answer is REQUIRED — do not omit it. "
    "Example: \\boxed{C}"
)


def parse_csv_row(row: dict, index: int, split: str, image_root: str,
                  masked: bool) -> dict | None:
    question = (row.get("Question") or "").strip()
    if not question:
        return None

    figure_path = (row.get("Figure_path") or "").strip()
    options = {}
    for letter in ["A", "B", "C", "D"]:
        key = f"Choice {letter}"
        val = (row.get(key) or "").strip()
        # Strip optional "A:" / "A." prefix
        val = val.lstrip("ABCD: .")
        if val:
            options[letter] = val
    if len(options) < 2:
        return None

    # Answer can be in "Answer_label" (letter) or "Answer" (letter or text).
    raw_answer = (row.get("Answer_label") or row.get("Answer") or "").strip()
    answer_label = raw_answer.upper()[:1]
    if answer_label not in options:
        # Try to match by value
        raw_low = raw_answer.lower().strip().lstrip("ABCD: .")
        for k, v in options.items():
            if v.lower().strip() == raw_low:
                answer_label = k
                break
    if answer_label not in options:
        return None

    caption = (row.get("Caption") or "").strip()

    if masked:
        # Training: only question text, no image, no caption, no options, no label.
        user_content = (
            f"{question}\n\n"
            "Based on medical knowledge, answer the question. "
            "If a single concise answer applies, put it in \\boxed{}."
        )
        ground_truth = ""
        extra_format = "free"
        reward_style = "rule_free"
        options_dict = {}
    else:
        # Validation: text caption as context + question + options + label.
        # We avoid image input to keep vLLM rollout text-only during validation.
        opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        ctx_line = f"Context (figure caption):\n{caption}\n\n" if caption else ""
        user_content = (
            f"{ctx_line}"
            f"Question: {question}\n\nOptions:\n{opts_text}\n\n"
            "Choose the single best answer (A, B, C, or D)."
        )
        ground_truth = answer_label
        extra_format = "mcq"
        reward_style = "rule_mcq"
        options_dict = options

    user_msg = {"role": "user", "content": user_content}

    return {
        "data_source": "pmc_vqa",
        "prompt": [
            {"role": "system", "content": PMC_VQA_SYSTEM_PROMPT_MCQ},
            user_msg,
        ],
        "reward_model": {
            "style": reward_style,
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index": index,
            "split": split,
            "question": question,
            "options": options_dict,
            "format": extra_format,
            "image_path": figure_path,
            "masked": masked,
        },
    }


def write_jsonl(path: str, entries: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmcvqa_dir", default="/scratch/self_evolving_datasets/pmc_vqa",
                    help="Directory with train.csv / test.csv / images/")
    ap.add_argument("--output_dir", default="/scratch/self_evolving_datasets/pmc_vqa_masked",
                    help="Output dir for masked train + labeled val JSONL")
    ap.add_argument("--train_csv", default="train_2.csv")
    ap.add_argument("--test_csv", default="test_2.csv")
    ap.add_argument("--image_subdir", default="images/figures",
                    help="Subdir relative to pmcvqa_dir where images live")
    ap.add_argument("--max_train", type=int, default=-1,
                    help="Cap training examples (masked seed targets). -1 = no cap")
    ap.add_argument("--max_val", type=int, default=500,
                    help="Cap validation examples (full-label). -1 = no cap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    train_path = os.path.join(args.pmcvqa_dir, args.train_csv)
    test_path = os.path.join(args.pmcvqa_dir, args.test_csv)
    image_root = os.path.join(args.pmcvqa_dir, args.image_subdir)

    # Masked training targets (questions only, no label)
    train_entries = []
    if os.path.exists(train_path):
        with open(train_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                entry = parse_csv_row(row, i, "train", image_root, masked=True)
                if entry:
                    train_entries.append(entry)
                if args.max_train > 0 and len(train_entries) >= args.max_train:
                    break

    # Labeled validation
    val_entries = []
    if os.path.exists(test_path):
        with open(test_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                entry = parse_csv_row(row, i, "test", image_root, masked=False)
                if entry:
                    val_entries.append(entry)
                if args.max_val > 0 and len(val_entries) >= args.max_val:
                    break

    os.makedirs(args.output_dir, exist_ok=True)
    train_out = os.path.join(args.output_dir, "train.jsonl")
    val_out = os.path.join(args.output_dir, "test.jsonl")
    write_jsonl(train_out, train_entries)
    write_jsonl(val_out, val_entries)

    print(f"Wrote {len(train_entries)} masked training entries to {train_out}")
    print(f"Wrote {len(val_entries)} labeled validation entries to {val_out}")


if __name__ == "__main__":
    main()
