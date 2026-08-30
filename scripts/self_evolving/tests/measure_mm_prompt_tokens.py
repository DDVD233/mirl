"""Measure the real prompt-token distribution of an image-bearing val parquet.

Image tokens ARE prompt tokens, and the agent loop refuses to truncate a multimodal
prompt (truncating corrupts vision feature alignment) -- so a single row over
`rollout.prompt_length` kills the whole validation, as one 10267-token MedXpertQA MM
row did on 2026-08-30 against a 10240 budget. An 8-row smoke will not find it; the
budget has to be set from the distribution, not guessed.

Prints the percentiles and the max, and tells you the smallest safe budget.

    python scripts/self_evolving/tests/measure_mm_prompt_tokens.py \
        --parquet /scratch/sheng/self_evolving/medxpertqa_mm_val.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report_top", type=int, default=5)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    tok = hf_tokenizer(args.model, trust_remote_code=True)
    proc = hf_processor(args.model, trust_remote_code=True, use_fast=True)
    cfg = OmegaConf.create({
        "prompt_key": "prompt", "image_key": "images", "video_key": "videos",
        "max_prompt_length": 1 << 20, "truncation": "left", "return_raw_chat": True,
        "filter_overlong_prompts": False,
        "apply_chat_template_kwargs": {"enable_thinking": True},
    })
    ds = RLHFDataset(data_files=args.parquet, tokenizer=tok, config=cfg, processor=proc)
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"measuring {n} rows from {args.parquet}", flush=True)

    lens = []
    for i in range(n):
        item = ds[i]
        messages = item["raw_prompt"]
        images, _ = asyncio.run(
            RLHFDataset.process_vision_info(messages, ds.image_patch_size, cfg))
        raw_text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=False)
        if images:
            ids = proc(text=[raw_text], images=images, return_tensors="pt")["input_ids"][0]
        else:
            ids = proc.tokenizer(raw_text, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]
        lens.append((int(len(ids)), i, [im.size for im in (images or [])]))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    lens.sort()
    vals = [x[0] for x in lens]

    def pct(p):
        return vals[min(int(len(vals) * p), len(vals) - 1)]

    print(f"\nprompt tokens: p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={vals[-1]}")
    print(f"over  8192: {sum(v > 8192 for v in vals)}")
    print(f"over 10240: {sum(v > 10240 for v in vals)}")
    print(f"over 12288: {sum(v > 12288 for v in vals)}")
    print(f"over 16384: {sum(v > 16384 for v in vals)}")
    print("\nlargest rows (tokens, index, image sizes):")
    for t, i, sizes in lens[-args.report_top:]:
        print(f"  {t:6d}  row {i:5d}  {sizes}")
    print(f"\nSmallest safe MAX_PROMPT_LEN for this set: {vals[-1]} "
          f"(nothing may be truncated; the agent loop refuses).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
