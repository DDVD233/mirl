#!/usr/bin/env python3
"""Build verl train/val parquets for MedThinkVQA (bio-nlp-umass/MedThinkVQA).

MedThinkVQA is multi-image diagnostic radiology: a clinical history, up to 49
images of a case, and five candidate diagnoses. Unlike HealthBench-Pro the answer
is a single letter, so the reward is exact match — deterministic, free, and
impossible for the policy or a judge to game. That alone makes it a much cleaner
RL target than a rubric graded by an LLM.

WHAT IS DELIBERATELY WITHHELD FROM THE PROMPT, and why it matters more here than
usual. The dataset ships the supervision for its own intermediate steps, and
three fields would hand the model the answer:
  * ``IMAGING_FINDINGS`` — the radiologist's reading of the images. This is the
    Step-2 target. Include it and the task stops being visual: the model can
    answer from prose without looking at a single image.
  * ``image_NN_caption`` — per-image findings, the Step-1 target, and they are
    explicit ("...showing dorsal signal alteration at the capitate-metacarpal
    joint (arrow)"). Same leak, once per image.
  * ``discussion`` — the worked explanation, which names the diagnosis outright.
Only CLINICAL_HISTORY + the images + the options go in. The captions and findings
are kept in ``extra_info`` so they remain available for SFT or analysis, but
nothing in the prompt path reads them.

IMAGE BUDGET. Capped at MAX_IMAGES (default 4). This is a real cost, not a free
saving: 64% of train and 82% of test cases have more than 4 images, and the
benchmark's own README reports accuracy rising with image count. So the ceiling
here is below the paper's. It is a deliberate trade for context budget, and the
number of images dropped is recorded per row (``n_images_total`` vs
``n_images_used``) so the loss is measurable rather than assumed.

WHICH images survive the cap: evenly spaced across the case, always keeping the
first and last. Taking the first 4 would bias toward one series/view, and the
benchmark is explicitly about integrating evidence ACROSS views.

Images are re-encoded to a bounded pixel budget so their token cost is
predictable: a Qwen-VL image costs about (H/28)*(W/28)/4 tokens, so the default
768-px long side is ~190 tokens and four of them ~760 — small against an 8k
response budget, which is the point.

  python scripts/self_evolving/build_medthinkvqa.py --out_dir /scratch/sheng/medthinkvqa
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

REPO = "bio-nlp-umass/MedThinkVQA"

SYSTEM = (
    "You are an expert radiologist. You are given a patient's clinical history and "
    "the imaging study for the case. Examine every image, integrate the findings "
    "across views, and reason through the differential before committing.\n"
    "Think step by step inside <think></think>, then give ONLY the single letter of "
    "the best diagnosis in \\boxed{}. For example: \\boxed{C}"
)

USER_TEMPLATE = """{image_block}
# Clinical history
{history}

# Candidate diagnoses
{options}

Which diagnosis best fits this case? Weigh the imaging evidence against each option
and rule out the distractors. Answer with the single letter in \\boxed{{}}."""


def pick_indices(n_total: int, k: int) -> list[int]:
    """Evenly spaced indices over a case, always including first and last.

    Not the first k: a radiology case is several series, and the benchmark is
    about integrating across views, so a prefix would systematically show one
    view of the anatomy and hide the rest.
    """
    if n_total <= k:
        return list(range(n_total))
    if k == 1:
        return [0]
    step = (n_total - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/scratch/sheng/medthinkvqa")
    ap.add_argument("--max_images", type=int, default=4)
    ap.add_argument("--long_side", type=int, default=768,
                    help="Longest image side after resize; ~(L/28)^2/4 tokens each.")
    ap.add_argument("--jpeg_quality", type=int, default=88)
    ap.add_argument("--limit", type=int, default=0, help="Debug: cap rows per split.")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)
    img_out = os.path.join(args.out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    print(f"[1/3] downloading {REPO} (parquet + images) ...", flush=True)
    src = snapshot_download(
        REPO, repo_type="dataset",
        allow_patterns=["*.parquet", "images/**"],
        max_workers=16,
    )
    print("      cached at", src, flush=True)

    stats: dict = {}
    for split, fname in (("train", "train.parquet"), ("val", "test.parquet")):
        df = pd.read_parquet(os.path.join(src, fname))
        if args.limit:
            df = df.head(args.limit)
        rows, n_missing, dropped_imgs, kept_imgs = [], 0, 0, 0

        for idx, r in df.iterrows():
            n_total = int(r["image_count"])
            paths, caps, mods = [], [], []
            for i in range(1, 50):
                p = r.get("image_%02d_path" % i)
                if isinstance(p, str) and p.strip():
                    paths.append(p.strip())
                    caps.append(str(r.get("image_%02d_caption" % i) or ""))
                    mods.append(str(r.get("image_%02d_modality" % i) or ""))
            if not paths:
                n_missing += 1
                continue

            keep = pick_indices(len(paths), args.max_images)
            dropped_imgs += len(paths) - len(keep)
            kept_imgs += len(keep)

            out_paths = []
            for j in keep:
                srcp = os.path.join(src, paths[j])
                if not os.path.exists(srcp):
                    continue
                rel = paths[j].replace("/", "__")
                dstp = os.path.join(img_out, f"{args.long_side}__{rel}")
                if not os.path.exists(dstp):
                    try:
                        im = Image.open(srcp).convert("RGB")
                        w, h = im.size
                        scale = args.long_side / float(max(w, h))
                        if scale < 1.0:
                            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                           Image.LANCZOS)
                        im.save(dstp, "JPEG", quality=args.jpeg_quality)
                    except Exception as e:  # a corrupt source image must not kill the build
                        print(f"      skip {paths[j]}: {type(e).__name__}: {e}", flush=True)
                        continue
                out_paths.append(dstp)
            if not out_paths:
                n_missing += 1
                continue

            opts = r["options"]
            if isinstance(opts, str):
                opts = json.loads(opts)
            opts = {str(k): str(v) for k, v in dict(opts).items()}
            options_txt = "\n".join(f"{k}. {opts[k]}" for k in sorted(opts))

            # One <image> per image actually attached; RLHFDataset asserts the
            # placeholder count matches the images list, so these must agree.
            image_block = "\n".join("<image>" for _ in out_paths)
            user = USER_TEMPLATE.format(
                image_block=image_block,
                history=str(r["CLINICAL_HISTORY"]).strip(),
                options=options_txt,
            )

            rows.append({
                "data_source": "medthinkvqa",
                "prompt": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user}],
                "images": out_paths,
                "ability": "medical_vqa",
                "reward_model": {"style": "rule", "ground_truth": str(r["correct_answer"]).strip()},
                "extra_info": {
                    "index": int(idx),
                    "split": split,
                    "case": str(r["title"]),
                    "answer_text": str(r.get("correct_answer_text") or ""),
                    "n_images_total": n_total,
                    "n_images_used": len(out_paths),
                    "modalities": ",".join(sorted({m for m in mods if m})),
                    "icd_chapter": str(r.get("ICD Chapter") or ""),
                    "icd_block": str(r.get("ICD Block") or ""),
                    "is_longitudinal": bool(r.get("is_longitudinal", False)),
                    # Withheld from the prompt on purpose (they are the answer):
                    # kept only so SFT / error analysis can reach them.
                    "imaging_findings": str(r.get("IMAGING_FINDINGS") or ""),
                    "captions": json.dumps([caps[j] for j in keep]),
                    # Retrieval query seed: history only, never the findings.
                    "retrieval_query": str(r["CLINICAL_HISTORY"]).strip()[:600],
                    "question": user,
                },
            })

        out = os.path.join(args.out_dir, f"medthinkvqa_{split}.parquet")
        pd.DataFrame(rows).to_parquet(out, index=False)
        stats[split] = {
            "rows": len(rows), "skipped_no_images": n_missing,
            "images_kept": kept_imgs, "images_dropped": dropped_imgs,
            "frac_images_dropped": round(dropped_imgs / max(1, kept_imgs + dropped_imgs), 3),
            "path": out,
        }
        print(f"[2/3] {split}: {len(rows)} rows -> {out}", flush=True)

    print("[3/3] summary:")
    print(json.dumps(stats, indent=1))
    # The dropped-image fraction bounds what this run can achieve relative to the
    # published benchmark; print it loudly rather than burying it.
    for s, v in stats.items():
        if v["frac_images_dropped"] > 0.2:
            print(f"  NOTE {s}: {v['frac_images_dropped']:.0%} of available images were dropped by "
                  f"the max_images={args.max_images} cap. The benchmark reports accuracy rising "
                  f"with image count, so this run's ceiling is below the paper's.")


if __name__ == "__main__":
    main()
