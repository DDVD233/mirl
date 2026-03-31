"""
Evaluate generated videos (non-India) against cultural questions using Qwen3-VL via vLLM.

Reads a single JSON file containing video IDs, prompts, countries, categories,
and evaluation questions. Videos are located at:
    <video-base>/<Country>/<category>/<id>.mp4

Uses a torch Dataset + DataLoader for async data loading with prefetching.

Usage:
    python evaluate_videos_global.py [--output results.json]
"""

import argparse
import json
import os
import re
import traceback
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

VIDEO_BASE = Path("/orcd/compute/ppliang/001/generated_videos_base_prompt")

SYSTEM_PROMPT = """\
You are an expert cultural anthropologist and visual evaluator assessing the cultural faithfulness of a generated video.

Before giving your final answer, reason through these steps:
1. **Identify the visual evidence**: Describe exactly what you observe in the video frames — specific objects, clothing details, spatial arrangements, architectural elements, lighting, and colors.
2. **Assess cultural accuracy**: Compare your observations against the culturally specific visual descriptions embedded in the question. Do not rely on implicit cultural knowledge — only evaluate what the question explicitly describes.
3. **Evaluate temporal and physical coherence** (for action questions): Examine the sequence, duration, physics, and progression of movements across frames. Note whether actions follow the temporal grounding specified in the question.
4. **Check for stereotyping or inauthenticity**: Flag if the video substitutes Western-centric defaults, hyper-exoticized elements, or generic representations in place of the specific cultural markers described in the question.

After your reasoning, provide the final answer given the {question} ONLY as "Yes" or "No".

The final answer MUST BE put in \\boxed{}. For example: \\boxed{Yes} or \\boxed{No}."""


def load_frameworks(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_eval_items(frameworks: dict, video_base: Path) -> list[dict]:
    """Build flat list of eval items from the combined frameworks JSON."""
    items = []
    for video_id, entry in frameworks.items():
        prompt = entry["prompt"]
        country = entry["country"]
        category = entry["category"]
        video_path = str(video_base / country / category / f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue

        evaluation = entry.get("evaluation", {})
        for q_category, q_list in evaluation.items():
            for q in q_list:
                items.append({
                    "video_path": video_path,
                    "prompt": prompt,
                    "country": country,
                    "question": q["question"],
                    "question_category": q_category,
                    "gt_answer": q.get("answer", ""),
                    "weight": q.get("weight", 1),
                })
    return items


class EvalDataset(Dataset):
    """Dataset that preprocesses eval items into LLM inputs with async-friendly loading."""

    def __init__(self, items: list[dict], processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{item['video_path']}",
                        "fps": 2.0,
                    },
                    {"type": "text", "text": item["question"]},
                ],
            },
        ]

        try:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(
                messages, return_video_metadata=True
            )
        except Exception:
            traceback.print_exc()
            print(f"Warning: failed to load video {item['video_path']}, using blank video")
            # Build a blank single-frame video as fallback
            blank_video = [torch.zeros(3, 1, 1, dtype=torch.uint8)]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {
                "item": item,
                "prompt": prompt_text,
                "mm_data": {"video": blank_video},
                "failed": True,
            }

        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        return {
            "item": item,
            "prompt": prompt_text,
            "mm_data": mm_data,
            "failed": False,
        }


def collate_fn(batch):
    """Pass through list of dicts without stacking."""
    return batch


def parse_response(text: str) -> tuple[str, str]:
    """Extract reasoning (everything before \\boxed) and answer (inside \\boxed{})."""
    answer = ""
    reasoning = text.strip()

    boxed_match = re.search(r"\\boxed\{(.*)\}", text, re.DOTALL)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        # Unwrap \\text{...} if present
        text_match = re.match(r"^\\text\{(.*)\}$", answer, re.DOTALL)
        if text_match:
            answer = text_match.group(1).strip()
        # Everything before \boxed is reasoning
        reasoning = text[: boxed_match.start()].strip()
        # Strip any surrounding tags (<think>, <tool_call>, etc.)
        reasoning = re.sub(r"</?(?:think|tool_call)>", "", reasoning).strip()

    return reasoning, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results_global.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for async video loading")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")
    parser.add_argument("--video-base", default=str(VIDEO_BASE),
                        help="Base directory for video files (country subdirs expected)")
    parser.add_argument("--frameworks", default="non_india_evaluation_frameworks.json",
                        help="Path to evaluation frameworks JSON file")
    parser.add_argument("--prompt-type", default=None,
                        help="Override prompt_type (default: video_base directory name)")
    args = parser.parse_args()

    frameworks = load_frameworks(Path(args.frameworks))
    print(f"Loaded {len(frameworks)} video entries from {args.frameworks}")

    items = build_eval_items(frameworks, Path(args.video_base))
    print(f"Total evaluation items (video x question): {len(items)}")

    processor = AutoProcessor.from_pretrained(args.model)

    dataset = EvalDataset(items, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2,
        shuffle=False,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        max_model_len=262144,
        limit_mm_per_prompt={"video": 1},
        enable_expert_parallel=True,
        async_scheduling=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=4096,
    )

    results = []
    processed = 0
    for batch in dataloader:
        # Filter out failed loads
        valid = [b for b in batch if not b["failed"]]
        # failed = [b for b in batch if b["failed"]]
        #
        # for b in failed:
        #     item = b["item"]
        #     result = {
        #         "prompt": item["prompt"],
        #         "country": item["country"],
        #         "question": item["question"],
        #         "question_category": item["question_category"],
        #         "path": item["video_path"],
        #         "answer": "",
        #         "reasoning": "",
        #         "raw_response": "",
        #         "gt_answer": item["gt_answer"],
        #         "weight": item["weight"],
        #         "prompt_type": args.prompt_type or Path(args.video_base).name,
        #         "error": "video_load_failed",
        #     }
        #     results.append(result)

        if valid:
            llm_inputs = [
                {"prompt": b["prompt"], "multi_modal_data": b["mm_data"]}
                for b in valid
            ]
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

            for b, output in zip(valid, outputs):
                item = b["item"]
                text = output.outputs[0].text
                reasoning, answer = parse_response(text)
                result = {
                    "prompt": item["prompt"],
                    "country": item["country"],
                    "question": item["question"],
                    "question_category": item["question_category"],
                    "path": item["video_path"],
                    "answer": answer,
                    "reasoning": reasoning,
                    "raw_response": text,
                    "gt_answer": item["gt_answer"],
                    "weight": item["weight"],
                    "prompt_type": args.prompt_type or Path(args.video_base).name,
                }
                results.append(result)
                print(json.dumps(result, indent=2))

        processed += len(batch)
        print(f"Processed {processed}/{len(items)}")

        if len(results) % 1000 < args.batch_size:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved partial results ({len(results)} items) to {args.output}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()