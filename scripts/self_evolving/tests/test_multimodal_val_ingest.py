"""Does the SOLVER side actually ingest an image-bearing task?

Runs the real trainer ingestion path over the MedXpertQA MM val parquet and follows
the images all the way to processor tensors.

THE CONTRACT THIS FOLLOWS. These arms rollout through an AgentLoop, so
`RLHFDataset.__getitem__` does NOT tokenize: it returns `raw_prompt` (chat messages
whose content parts carry image references) and DROPS the `images` column, and the
AgentLoop later calls `RLHFDataset.process_vision_info` to fetch the pixels. A test
that asserts on `input_ids` / `multi_modal_inputs` at the dataset level is therefore
testing the wrong stage and fails on a perfectly healthy pipeline.

So this checks, in order:
  1. `<image>` placeholders bind to the `images` column and become image content
     parts in `raw_prompt`,
  2. `process_vision_info` fetches real pixels from those references,
  3. the images are not the BLACK PLACEHOLDERS that `_sanitize_image_messages`
     silently substitutes for unreadable files -- the failure that would otherwise
     look like a working pipeline that just answers badly,
  4. the processor expands them into image tokens in the final input_ids.

What it does NOT prove: that the rollout engine generates well from those inputs, or
that the PROPOSER can mint such a task (on this branch it cannot -- the multimodal
minting path was cut in 9da1522c).

    python scripts/self_evolving/tests/test_multimodal_val_ingest.py \
        --parquet /scratch/sheng/self_evolving/medxpertqa_mm_val.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from omegaconf import OmegaConf


def _image_parts(messages) -> list:
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts.extend([c for c in content if isinstance(c, dict) and c.get("type") == "image"])
    return parts


def _is_black_placeholder(img) -> bool:
    """`_sanitize_image_messages` substitutes a 224x224 black image for unreadable files."""
    if img.size != (224, 224):
        return False
    return not img.convert("RGB").getbbox()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/scratch/sheng/self_evolving/medxpertqa_mm_val.parquet")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n", type=int, default=4, help="rows to pull")
    args = ap.parse_args()

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    tokenizer = hf_tokenizer(args.model, trust_remote_code=True)
    processor = hf_processor(args.model, trust_remote_code=True, use_fast=True)
    print(f"processor: {type(processor).__name__}")
    if processor is None:
        print("FAIL: no processor for this model -- multimodal rows cannot be tokenized")
        return 1

    # Mirror the arm's data config (run_9b_hb_gen.sh).
    cfg = OmegaConf.create({
        "prompt_key": "prompt",
        "image_key": "images",
        "video_key": "videos",
        "max_prompt_length": 16384,
        "truncation": "left",
        "return_raw_chat": True,
        "filter_overlong_prompts": False,
        "apply_chat_template_kwargs": {"enable_thinking": True},
    })

    ds = RLHFDataset(data_files=args.parquet, tokenizer=tokenizer, config=cfg, processor=processor)
    print(f"dataset rows: {len(ds)}")

    # Cover a multi-image row too, not just single-image ones.
    idxs = list(range(min(args.n, len(ds))))
    for i in range(min(len(ds), 500)):
        if len(ds.dataframe[i].get("images") or []) > 1:
            if i not in idxs:
                idxs.append(i)
            print(f"multi-image row: index {i} with {len(ds.dataframe[i]['images'])} images")
            break

    failures = []
    for i in idxs:
        n_src = len(ds.dataframe[i].get("images") or [])
        item = ds[i]
        messages = item["raw_prompt"]
        parts = _image_parts(messages)
        if len(parts) != n_src:
            failures.append(f"row {i}: {n_src} images in the row but {len(parts)} image parts "
                            f"in raw_prompt")
            continue

        images, videos = asyncio.run(
            RLHFDataset.process_vision_info(messages, ds.image_patch_size, cfg))
        images = images or []
        if len(images) != n_src:
            failures.append(f"row {i}: process_vision_info returned {len(images)} of {n_src} images")
            continue
        placeholders = [k for k, im in enumerate(images) if _is_black_placeholder(im)]
        if placeholders:
            failures.append(f"row {i}: images {placeholders} came back as BLACK PLACEHOLDERS "
                            f"(unreadable file silently substituted)")
            continue

        raw_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[raw_text], images=images, return_tensors="pt")
        ids = inputs["input_ids"][0]
        img_tok_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        n_img_tok = int((ids == img_tok_id).sum()) if img_tok_id is not None else -1
        sizes = [im.size for im in images]
        print(f"row {i}: images={n_src} sizes={sizes} prompt_tokens={len(ids)} "
              f"image_tokens={n_img_tok} pixel_values={tuple(inputs['pixel_values'].shape)}")
        if n_img_tok == 0:
            failures.append(f"row {i}: image tokens never expanded into input_ids")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: image-bearing val rows reach the processor as real pixel tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
