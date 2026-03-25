"""
Evaluate generated videos (non-India) against cultural questions using Qwen3-VL via vLLM.

Reads a single JSON file containing video IDs, prompts, countries, categories,
and evaluation questions. Videos are located at:
    <video-base>/<Country>/<category>/<id>.mp4

Usage:
    python evaluate_videos_global.py [--output results.json]
"""

import argparse
import json
import re
from pathlib import Path

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


def build_messages(item: dict) -> list[dict]:
    return [
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


def prepare_llm_input(messages: list[dict], processor) -> dict:
    """Apply chat template and extract video data for llm.generate()."""
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(
        messages, return_video_metadata=True
    )
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    return {"prompt": prompt, "multi_modal_data": mm_data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results_global.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")
    parser.add_argument("--video-base", default=str(VIDEO_BASE),
                        help="Base directory for video files (country subdirs expected)")
    parser.add_argument("--frameworks", default="non_india_evaluation_frameworks.json",
                        help="Path to evaluation frameworks JSON file")
    args = parser.parse_args()

    frameworks = load_frameworks(Path(args.frameworks))
    print(f"Loaded {len(frameworks)} video entries from {args.frameworks}")

    items = build_eval_items(frameworks, Path(args.video_base))
    print(f"Total evaluation items (video x question): {len(items)}")

    processor = AutoProcessor.from_pretrained(args.model)

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
    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start : batch_start + args.batch_size]
        conversations = [build_messages(item) for item in batch]
        llm_inputs = [prepare_llm_input(msgs, processor) for msgs in conversations]

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

        for item, output in zip(batch, outputs):
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
                "prompt_type": Path(args.video_base).name,
            }
            results.append(result)
            print(json.dumps(result, indent=2))

        print(f"Processed {min(batch_start + args.batch_size, len(items))}/{len(items)}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()