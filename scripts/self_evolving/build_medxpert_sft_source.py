"""Assemble the stage-1 SFT source set for the MedXpertQA arm.

WHY THIS EXISTS. MedXpertQA ships no train split -- only test (2450 text / 2000 mm)
and a 5-item dev -- and the test sets ARE our held-out validation. So unlike the
MIMIC-Rare recipe, which distilled from a real train.jsonl, the SFT corpus has to be
built from public sources that are not the benchmark. Two, chosen for what each
teaches:

  * MedQA (USMLE, 10,178 public train items) -- multi-step clinical reasoning ending
    in ONE committed option. The same board-examination genre MedXpertQA draws from,
    which is what the text split (the weaker half, and the one that decayed) needs.
  * CLIMB (21,581 staged images with ground-truth findings) -- reading a medical
    image and naming what is in it. Closed-set classification, so it teaches
    image-grounding and commitment rather than clinical inference; that division of
    labour is deliberate.

Rows come out in the RLHF shape the trace generator consumes: prompt / images /
reward_model.ground_truth / extra_info.

MODALITY WEIGHTING. CLIMB is staged balanced (3000 per modality), but MedXpertQA MM
is not: Skeletal 491, Cardiovascular 320, Nervous 231, Integumentary 225,
Respiratory 192, Digestive 168, ... Sampling CLIMB flat would over-train derm,
fundus and mammo relative to the benchmark. The default weights below map each CLIMB
modality onto the benchmark share it plausibly serves. Note the honest gap: CLIMB has
no musculoskeletal set, so Skeletal -- the LARGEST MM category at 24.5% -- is
essentially uncovered, and no weighting fixes that.

    python scripts/self_evolving/build_medxpert_sft_source.py \
        --out /scratch/sheng/self_evolving/medxpert_sft/source.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
from collections import Counter

# Keep IN SYNC with eval/preprocess_medxpertqa.py so SFT teaches the exact
# instruction and final-answer shape validation asks for.
ANSWER_FORMAT_INSTR = (
    "\n\nWork through the case, then end your response with a single final line of "
    "exactly this form:\nFinal answer: (X)\nwhere X is the letter of the single best option."
)
# CLIMB is open-set (a finding, not a letter), so the committed line differs.
CLIMB_FORMAT_INSTR = (
    "\n\nWork through what the image shows, then end your response with a single "
    "final line of exactly this form:\nFinal answer: <answer>\nwhere <answer> is "
    "chosen from the options listed above."
)

# CLIMB modality -> share of the multimodal half. Derived from MedXpertQA MM's own
# body-system distribution (see the module docstring), not from what CLIMB happens
# to hold. Chest imaging carries Respiratory + Cardiovascular; mri and pathology are
# the only routes to Nervous/Skeletal/Digestive, so they are weighted up despite
# CLIMB covering them thinly.
DEFAULT_MODALITY_WEIGHTS = {
    "chest_xray": 0.20,
    "ct": 0.14,
    "mri": 0.20,
    "pathology": 0.15,
    "derm": 0.11,
    "fundus": 0.08,
    "mammo": 0.05,
    "ultrasound": 0.07,
}


def _medqa_rows(path: str, limit: int) -> list[dict]:
    """MedQA train -> RLHF rows whose prompt matches the MedXpertQA val format."""
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            opts = ex.get("options")
            if isinstance(opts, str):
                try:
                    opts = ast.literal_eval(opts)
                except (ValueError, SyntaxError):
                    continue
            if not isinstance(opts, dict) or not opts:
                continue
            letter = str(ex.get("answer_idx") or "").strip().upper()
            if letter not in opts:
                continue
            choices = " ".join(f"({k}) {v}" for k, v in sorted(opts.items()))
            content = (f"{ex['question']}\nAnswer Choices: {choices}"
                       + ANSWER_FORMAT_INSTR)
            out.append({
                "data_source": "medqa_usmle",
                "prompt": [{"role": "user", "content": content}],
                "reward_model": {"style": "mcq", "ground_truth": letter},
                "extra_info": {
                    "question_id": f"medqa_{i}",
                    "source": "medqa_usmle",
                    "answer_text": str(opts[letter]),
                    "n_options": len(opts),
                    "meta_info": ex.get("meta_info", ""),
                    "kind": "mcq",
                },
            })
            if limit and len(out) >= limit:
                break
    return out


def _climb_rows(manifest: str, images_root: str, total: int, weights: dict,
                seed: int, skip_multilabel: bool) -> list[dict]:
    """CLIMB manifest -> RLHF rows, sampled to the benchmark's modality shape."""
    rng = random.Random(seed)
    by_mod: dict[str, list] = {}
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ans = str(r.get("answer") or "").strip()
            if not ans:
                continue
            # CheXpert-style multi-label rows ("A, B, C") are NLP-extracted from
            # reports and are the noisiest labels in the corpus; a wrong label yields
            # a fluent, confident, WRONG trace, and the ground-truth gate cannot see
            # that because it only checks agreement with the label.
            if skip_multilabel and "," in ans:
                continue
            by_mod.setdefault(str(r.get("modality") or "other"), []).append(r)

    out = []
    for mod, rows in sorted(by_mod.items()):
        want = int(round(total * float(weights.get(mod, 0.0))))
        if want <= 0:
            continue
        rng.shuffle(rows)
        for r in rows[:want]:
            paths = [os.path.join(images_root, rel) for rel in (r.get("images") or [])]
            if not paths or not all(os.path.isfile(p) for p in paths):
                continue
            q = str(r.get("question") or "")
            # The manifest question already carries its own <image> placeholder.
            if "<image>" not in q:
                q = "<image>\n" + q
            out.append({
                "data_source": f"climb_{mod}",
                "prompt": [{"role": "user", "content": q + CLIMB_FORMAT_INSTR}],
                "images": paths,
                "reward_model": {"style": "open", "ground_truth": str(r.get("answer"))},
                "extra_info": {
                    "question_id": f"climb_{mod}_{len(out)}",
                    "source": "climb",
                    "modality": mod,
                    "dataset": r.get("dataset", ""),
                    "kind": "open",
                },
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--medqa_jsonl", default="/scratch/sheng/self_evolving/bench_raw/MedQA/train.jsonl")
    ap.add_argument("--climb_manifest", default="/scratch/sheng/self_evolving/mm_media/manifest.jsonl")
    ap.add_argument("--climb_images_root", default="/scratch/sheng/self_evolving/mm_media/images")
    ap.add_argument("--n_text", type=int, default=6000, help="MedQA questions to keep")
    ap.add_argument("--n_mm", type=int, default=4000, help="CLIMB questions to keep")
    ap.add_argument("--keep_multilabel", action="store_true",
                    help="keep CheXpert-style comma-separated multi-label rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows: list[dict] = []
    if args.n_text > 0:
        if not os.path.isfile(args.medqa_jsonl):
            raise SystemExit(f"MedQA not found at {args.medqa_jsonl}")
        rows += _medqa_rows(args.medqa_jsonl, args.n_text)
    if args.n_mm > 0:
        rows += _climb_rows(args.climb_manifest, args.climb_images_root, args.n_mm,
                            DEFAULT_MODALITY_WEIGHTS, args.seed, not args.keep_multilabel)

    if not rows:
        raise SystemExit("no rows assembled")
    random.Random(args.seed).shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    kinds = Counter(r["extra_info"]["kind"] for r in rows)
    srcs = Counter(r["data_source"] for r in rows)
    print(f"wrote {len(rows)} source rows -> {args.out}")
    print(f"by kind: {dict(kinds)}")
    print(f"by source: {dict(srcs.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
