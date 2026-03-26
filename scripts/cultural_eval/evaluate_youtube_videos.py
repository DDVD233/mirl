"""
Evaluate YouTube reference videos against cultural questions using Qwen3-VL via vLLM.

Each parent entry has three sets of YouTube videos:
  - object_tag_youtube_video  -> evaluated with "objects" questions only
  - action_tag_youtube_video  -> evaluated with "actions" questions only
  - scene_tag_youtube_video   -> evaluated with "scene"   questions only

Videos are located at:
    <video-base>/<Country>/<youtube_id>.mp4

Uses a torch Dataset + DataLoader for async data loading with prefetching.

Usage:
    python evaluate_youtube_videos.py [--output eval_results_youtube.json]
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

VIDEO_BASE = Path("/orcd/compute/ppliang/001/downloaded_cultural_videos/videos")

# Maps each youtube-video field to the evaluation question category it should use
TAG_TO_EVAL_CATEGORY = {
    "object_tag_youtube_video": "objects",
    "action_tag_youtube_video": "actions",
    "scene_tag_youtube_video": "scene",
}

SYSTEM_PROMPT = """\
You are an expert cultural anthropologist and visual evaluator assessing the cultural faithfulness of a video.

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
    """Build flat list of eval items from the combined frameworks JSON.

    For each parent entry, iterate over the three youtube-video tag lists.
    Each youtube video is only evaluated against the matching question category.
    """
    items = []
    for parent_id, entry in frameworks.items():
        prompt = entry["prompt"]
        country = entry["country"]
        category = entry["category"]
        evaluation = entry.get("evaluation", {})

        for tag_field, eval_category in TAG_TO_EVAL_CATEGORY.items():
            questions = evaluation.get(eval_category, [])
            if not questions:
                continue

            for yt_video in entry.get(tag_field, []):
                local_path = yt_video.get("local_path", "")
                if not local_path:
                    continue
                video_path = str(video_base / local_path.removeprefix("videos/"))
                if not os.path.isfile(video_path):
                    continue

                youtube_id = Path(local_path).stem
                video_url = yt_video.get("video_url", "")
                video_title = yt_video.get("video_title", "")
                # e.g. "object_tag_youtube_video" -> "object_tag"
                tag_key = tag_field.removesuffix("_youtube_video")
                tag_value = entry.get(tag_key, "")

                for q in questions:
                    items.append({
                        "video_path": video_path,
                        "parent_id": parent_id,
                        "prompt": prompt,
                        "country": country,
                        "category": category,
                        "tag_field": tag_field,
                        "tag_key": tag_key,
                        "tag_value": tag_value,
                        "youtube_id": youtube_id,
                        "video_url": video_url,
                        "video_title": video_title,
                        "question": q["question"],
                        "question_category": eval_category,
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
                        "min_frames": 4,
                        "max_frames": 32,
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
        text_match = re.match(r"^\\text\{(.*)\}$", answer, re.DOTALL)
        if text_match:
            answer = text_match.group(1).strip()
        reasoning = text[: boxed_match.start()].strip()
        reasoning = re.sub(r"</?(?:think|tool_call)>", "", reasoning).strip()

    return reasoning, answer


def build_result(item: dict, answer: str = "", reasoning: str = "",
                 raw_response: str = "", error: str = "") -> dict:
    """Build a result dict from an item."""
    result = {
        "parent_id": item["parent_id"],
        "prompt": item["prompt"],
        "country": item["country"],
        "category": item["category"],
        "tag_field": item["tag_field"],
        item["tag_key"]: item["tag_value"],
        "youtube_id": item["youtube_id"],
        "video_url": item["video_url"],
        "video_title": item["video_title"],
        "question": item["question"],
        "question_category": item["question_category"],
        "path": item["video_path"],
        "answer": answer,
        "reasoning": reasoning,
        "raw_response": raw_response,
        "gt_answer": item["gt_answer"],
        "weight": item["weight"],
        "prompt_type": "youtube",
    }
    if error:
        result["error"] = error
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results_youtube.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers for async video loading (0 = main process)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")
    parser.add_argument("--video-base", default=str(VIDEO_BASE),
                        help="Base directory for YouTube video files")
    parser.add_argument("--frameworks", default="all_evaluation_frameworks.json",
                        help="Path to evaluation frameworks JSON file")
    args = parser.parse_args()

    frameworks = load_frameworks(Path(args.frameworks))
    print(f"Loaded {len(frameworks)} entries from {args.frameworks}")

    items = build_eval_items(frameworks, Path(args.video_base))
    print(f"Total evaluation items (youtube video x question): {len(items)}")

    # Resume from partial results if output file already exists
    results = []
    if os.path.isfile(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {args.output}")
        done = {(r["path"], r["question"]) for r in results}
        items = [it for it in items if (it["video_path"], it["question"]) not in done]
        print(f"Remaining items after skipping completed: {len(items)}")

    if not items:
        print("No items to evaluate.")
        return

    processor = AutoProcessor.from_pretrained(args.model)

    dl_kwargs = {}
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    dataset = EvalDataset(items, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        **dl_kwargs,
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

    processed = 0
    for batch in dataloader:
        valid = [b for b in batch if not b["failed"]]
        failed = [b for b in batch if b["failed"]]

        for b in failed:
            results.append(build_result(b["item"], error="video_load_failed"))

        if valid:
            llm_inputs = [
                {"prompt": b["prompt"], "multi_modal_data": b["mm_data"]}
                for b in valid
            ]
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

            for b, output in zip(valid, outputs):
                text = output.outputs[0].text
                reasoning, answer = parse_response(text)
                result = build_result(b["item"], answer=answer,
                                      reasoning=reasoning, raw_response=text)
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