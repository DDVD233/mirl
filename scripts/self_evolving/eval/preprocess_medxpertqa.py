"""Build held-out MedXpertQA val parquets (Text and MM) for in-loop validation.

Same row schema as scripts/self_evolving/eval/preprocess_healthbench_professional.py
so the existing rubric scorer + val judge grade these rows unchanged. Reads the raw
jsonl from NFS -- no HF `datasets` needed.

MedXpertQA IS NOT A RUBRIC BENCHMARK. It is multiple choice with a gold letter:
Text = 2450 questions x 10 options (A-J), MM = 2000 questions x 5 options (A-E)
plus one or two images each. To grade it through the rubric path without touching
the scorer, each question becomes a ONE-CRITERION rubric worth 1 point -- "the
final answer is (X)" -- so the grader's sum(met)/sum(positive) is exactly 0 or 1
and the mean over the split is exactly accuracy. The judge decides whether the
response committed to the gold option, which is more robust than a regex over
free-form reasoning and keeps validation on the gpt judge like every other arm.

The gold letter lives ONLY in `extra_info.rubric_items` (judge side) and
`extra_info.gold_label` (metrics side). It is never in the prompt.

MM rows carry absolute image paths under the `images` column and one `<image>`
placeholder per image at the head of the user turn, which is what
verl/utils/dataset/rl_dataset.py `_build_messages` expects. Paths point into the
shared /scratch NFS, so any MSR pod resolves them with no file server.

Example:
    python scripts/self_evolving/eval/preprocess_medxpertqa.py --subset text \
        --out /scratch/sheng/self_evolving/medxpertqa_text_val.parquet
    python scripts/self_evolving/eval/preprocess_medxpertqa.py --subset mm \
        --out /scratch/sheng/self_evolving/medxpertqa_mm_val.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("/scratch/sheng/self_evolving/bench_raw/MedXpertQA")
SUBSETS = {"text": ("Text", "medxpertqa_text"), "mm": ("MM", "medxpertqa_mm")}

# Appended to every question. MedXpertQA is scored on the chosen letter, so the
# response needs one unambiguous commitment for the judge to grade (and for the
# standalone exact-match runner to parse). Kept short: the solver's own system
# prompt already tells it how to reason.
ANSWER_FORMAT_INSTR = (
    "\n\nWork through the case, then end your response with a single final line of "
    "exactly this form:\nFinal answer: (X)\nwhere X is the letter of the single best option."
)


def _criterion(label: str, option_text: str) -> dict:
    """The one graded criterion: did the response commit to the gold option?

    Worded to fail the two ways a wrong-but-plausible response passes a naive
    grader: naming the right option only as one candidate among several, and
    reasoning toward it without ever choosing.
    """
    return {
        "criterion_text": (
            f"The response's single final answer is option ({label}): \"{option_text}\". "
            f"Award the point ONLY if the response commits to ({label}) as its final choice. "
            f"Award nothing if it selects a different option, if it lists ({label}) among "
            f"several candidates without settling on it, or if it never commits to one option."
        ),
        "points": 1,
    }


def _slug(s: str) -> str:
    return str(s or "other").strip().lower().replace(" ", "_")


def _to_verl_rows(raw_rows: list[dict], subset: str, split: str, images_dir: Path, stats: dict):
    _, source = SUBSETS[subset]
    for i, ex in enumerate(raw_rows):
        label = str(ex["label"]).strip()
        options = ex["options"]
        if label not in options:
            stats["skipped_bad_label"] += 1
            continue

        images: list[str] = []
        if subset == "mm":
            for rel in ex.get("images") or []:
                p = images_dir / rel
                if not p.is_file():
                    stats["missing_image"] += 1
                    images = []
                    break
                images.append(str(p))
            if not images:
                # A multimodal question without its images is a different (easier or
                # impossible) question -- never silently grade it as text.
                stats["skipped_no_image"] += 1
                continue

        content = str(ex["question"]) + ANSWER_FORMAT_INSTR
        if images:
            content = ("<image>" * len(images)) + "\n" + content
        messages = [{"role": "user", "content": content}]

        task = _slug(ex.get("medical_task"))
        row = {
            "data_source": f"{source}/{task}",
            "prompt": messages,
            "reward_model": {"style": "rubric", "ground_truth": label},
            "extra_info": {
                "question_id": str(ex["id"]),
                "index": str(i),
                "split": split,
                "rubric_items": [_criterion(label, str(options[label]))],
                "conversation": messages,
                "use_case": task,
                "type": _slug(ex.get("question_type")),
                "difficulty": "expert",
                "specialty": _slug(ex.get("body_system")),
                "source": source,
                "gold_label": label,
                "n_options": len(options),
            },
        }
        if images:
            row["images"] = images
            stats["with_images"] += 1
        stats["kept"] += 1
        yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, choices=sorted(SUBSETS))
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--raw_dir", default=str(RAW_DIR))
    ap.add_argument("--split", default="test")
    ap.add_argument("--images_dir", default=None,
                    help="dir holding the unzipped image files (default <raw_dir>/images)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0 = all) for a quick smoke")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    src_dir, _ = SUBSETS[args.subset]
    raw_path = raw_dir / src_dir / f"{args.split}.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]

    # images.zip unpacks to <raw_dir>/images/<file>, while the jsonl `images` field
    # holds the bare filename ("MM-0-a.jpeg").
    images_dir = Path(args.images_dir) if args.images_dir else raw_dir / "images"
    stats = {"kept": 0, "skipped_bad_label": 0, "skipped_no_image": 0,
             "missing_image": 0, "with_images": 0}
    rows = list(_to_verl_rows(raw_rows, args.subset, args.split, images_dir, stats))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no valid rows from {raw_path}")
    if args.subset == "mm" and stats["with_images"] != stats["kept"]:
        raise SystemExit(f"mm subset: {stats['kept'] - stats['with_images']} rows lost their images")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
