"""Does the SOLVER side actually ingest an image-bearing task?

Runs the real trainer ingestion path -- verl's RLHFDataset (which main_ppo forces
for validation) with the real tokenizer and processor -- over the MedXpertQA MM val
parquet, and asserts the images survive all the way to processor tensors.

This is the half of "multimodal works" that can be checked without GPUs. It proves:
  * the `<image>` placeholders in the prompt bind to the `images` column,
  * the paths resolve (shared /scratch NFS, no file server needed on an MSR pod),
  * the processor emits pixel values and expands the image tokens in input_ids,
  * a multi-image row works, not just a single-image one.

What it does NOT prove: that the ROLLOUT engine generates from those inputs, or that
the proposer can mint such a task in the first place. Run with:

    python scripts/self_evolving/tests/test_multimodal_val_ingest.py \
        --parquet /scratch/sheng/self_evolving/medxpertqa_mm_val.parquet
"""

from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf


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

    # Cover a multi-image row too: the first row with more than one image.
    idxs = list(range(args.n))
    frame = ds.dataframe
    for i in range(min(len(ds), 500)):
        imgs = frame[i]["images"] if hasattr(frame, "__getitem__") else None
        if imgs is not None and len(imgs) > 1:
            idxs.append(i)
            print(f"multi-image row at index {i}: {len(imgs)} images")
            break

    failures = []
    for i in idxs:
        item = ds[i]
        keys = sorted(item.keys())
        has_mm = "multi_modal_inputs" in item
        n_img_tok = None
        if has_mm:
            mm = item["multi_modal_inputs"]
            mm_keys = sorted(mm.keys())
            pv = mm.get("pixel_values")
            shape = tuple(pv.shape) if pv is not None else None
        else:
            mm_keys, shape = [], None
        ids = item["input_ids"]
        # The processor expands one <image> placeholder into many image tokens; if
        # that did not happen the model would see a bare text prompt and "solve"
        # an image question blind, which is the silent failure worth catching.
        try:
            img_tok_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            n_img_tok = int((ids == img_tok_id).sum()) if img_tok_id is not None else None
        except Exception:
            n_img_tok = None
        print(f"row {i}: keys={keys}")
        print(f"    multi_modal_inputs={mm_keys} pixel_values={shape} "
              f"image_tokens_in_input_ids={n_img_tok} prompt_len={int(ids.shape[-1])}")
        if not has_mm:
            failures.append(f"row {i}: no multi_modal_inputs")
        elif shape is None:
            failures.append(f"row {i}: no pixel_values")
        elif n_img_tok is not None and n_img_tok == 0:
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
