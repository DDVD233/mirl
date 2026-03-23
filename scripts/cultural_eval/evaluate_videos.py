"""
Evaluate generated videos against cultural questions using Qwen3-VL-8B-Instruct via vLLM.

Usage:
    python scripts/cultural_eval/evaluate_videos.py [--output results.json]
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_BASE = Path("/orcd/compute/ppliang/001/generated_videos_base_prompt/India")

SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. The reasoning process MUST BE enclosed "
    "within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
)


def load_questions(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["country"] == "India"]


def build_eval_items(rows: list[dict], questions: dict) -> list[dict]:
    """Build flat list of (video_path, prompt, question_text, category) dicts."""
    items = []
    for row in rows:
        prompt = row["prompt"]
        category = row["category"]
        video_id = row["id"]
        video_path = str(VIDEO_BASE / category / f"{video_id}.mp4")

        if prompt not in questions:
            print(f"WARNING: prompt not found in questions.json: {prompt}")
            continue

        prompt_qs = questions[prompt]
        for q_category, q_list in prompt_qs.items():
            for q in q_list:
                items.append({
                    "video_path": video_path,
                    "prompt": prompt,
                    "question": q["question"],
                    "question_category": q_category,
                    "gt_answer": q.get("answer", ""),
                    "weight": q.get("weight", 1),
                })
    return items


def parse_response(text: str) -> tuple[str, str]:
    """Extract reasoning from <think> tags and answer from \\boxed{}."""
    reasoning = ""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()

    answer = ""
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if boxed_match:
        answer = boxed_match.group(1).strip()

    return reasoning, answer


def build_messages(item: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{item['video_path']}",
                    "fps": 1.0,
                },
                {"type": "text", "text": item["question"]},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(SCRIPT_DIR / "eval_results.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")
    args = parser.parse_args()

    questions = load_questions(SCRIPT_DIR / "question.json")
    rows = load_csv(SCRIPT_DIR / "culturalframes_balanced_sample.csv")
    print(f"Loaded {len(rows)} India videos, {len(questions)} prompts in question.json")

    items = build_eval_items(rows, questions)
    print(f"Total evaluation items (video x question): {len(items)}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        max_model_len=262144,
        kv_cache_max_size="30G",
        limit_mm_per_prompt={"video": 1},
        enable_expert_parallel=True,
        async_scheduling=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=4096,
    )

    results = []
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start : batch_start + args.batch_size]
        conversations = [build_messages(item) for item in batch]

        outputs = llm.chat(conversations, sampling_params=sampling_params)

        for item, output in zip(batch, outputs):
            text = output.outputs[0].text
            reasoning, answer = parse_response(text)
            result = {
                "prompt": item["prompt"],
                "question": item["question"],
                "question_category": item["question_category"],
                "path": item["video_path"],
                "answer": answer,
                "reasoning": reasoning,
                "raw_response": text,
                "gt_answer": item["gt_answer"],
                "weight": item["weight"],
            }
            results.append(result)
            print(json.dumps(result, indent=2))

        print(f"Processed {min(batch_start + args.batch_size, len(items))}/{len(items)}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()