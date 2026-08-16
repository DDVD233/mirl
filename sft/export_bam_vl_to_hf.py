"""
Convert a BAM-VL training checkpoint into a Hugging Face model directory.

A training checkpoint written by accelerate is a single FSDP FULL_STATE_DICT
(`pytorch_model_fsdp.bin`) plus optimizer/RNG files. It has no config.json, no
tokenizer, and no safetensors shards, so `from_pretrained` cannot read it and it
cannot be uploaded to the Hub as-is. This script produces a directory that can.

What it emits into --out_dir:

  config.json / model-*.safetensors / ...   the LoRA-merged Qwen3.5 VL backbone,
                                            loadable with AutoModelForImageTextToText
  tokenizer* / preprocessor_config.json     copied from the base model repo
  bam_adapters.pt                           the BAM side-channel adapters + CLS heads,
                                            with the hyperparameters needed to rebuild them
  README.md                                 minimal model card (edit before publishing)

IMPORTANT — the merged backbone is NOT the full trained system. BAMVLQA.forward
pools the penultimate hidden state, computes a delta from the facial/pose/audio
adapters, and adds it to the last hidden state before lm_head. A plain
`generate()` call on the merged model does none of that. To reproduce training
behaviour you need bam_adapters.pt plus the BAMVLQA wrapper code.

Usage:
    python export_bam_vl_to_hf.py \
        --ckpt   /path/to/checkpoints/<run>/step_2023 \
        --out_dir /path/to/hf_export \
        --config configs/config_bam_vl_accelerate.yaml

Memory: peaks around one full copy of the model (~54 GB for 27B in bf16). The
state dict is memory-mapped rather than read into RAM, and tensors are moved into
the model with assign=True to avoid a second copy.
"""
import argparse
import json
import os

import torch
from omegaconf import OmegaConf

STREAMS = ("facial", "pose", "audio")
BACKBONE_PREFIX = "backbone."
ADAPTER_PREFIXES = tuple(f"{s}_adapter." for s in STREAMS)
HEAD_PREFIX = "heads."


def parse_args():
    p = argparse.ArgumentParser(description="Export a BAM-VL checkpoint to HF format")
    p.add_argument("--ckpt", required=True,
                   help="checkpoint step directory (the one containing pytorch_model_fsdp.bin)")
    p.add_argument("--out_dir", required=True, help="output directory for the HF model")
    p.add_argument("--config", default="configs/config_bam_vl_accelerate.yaml",
                   help="training config; LoRA settings must match the run that produced --ckpt")
    p.add_argument("--max_shard_size", default="5GB")
    p.add_argument("--no_merge", action="store_true",
                   help="keep LoRA separate (save the PEFT adapter) instead of merging into the base")
    return p.parse_args()


def load_flat_state_dict(ckpt_dir):
    """Memory-map the FSDP FULL_STATE_DICT so we never hold two copies of a 27B model."""
    path = os.path.join(ckpt_dir, "pytorch_model_fsdp.bin")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. --ckpt must point at the step_<N> directory itself."
        )
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        # Older saves, or a dict carrying non-tensor entries, cannot use weights_only/mmap.
        print("[export] mmap/weights_only load failed; falling back to a full read")
        return torch.load(path, map_location="cpu", weights_only=False)


def split_by_prefix(sd):
    """Partition the flat checkpoint into backbone / adapter / head / leftover groups."""
    backbone, adapters, heads, other = {}, {}, {}, {}
    for k, v in sd.items():
        if k.startswith(BACKBONE_PREFIX):
            # "backbone.base_model.model.*" -> "base_model.model.*", which is exactly
            # what the PEFT-wrapped model expects.
            backbone[k[len(BACKBONE_PREFIX):]] = v
        elif k.startswith(ADAPTER_PREFIXES):
            adapters[k] = v
        elif k.startswith(HEAD_PREFIX):
            heads[k] = v
        else:
            other[k] = v
    return backbone, adapters, heads, other


def build_backbone(cfg, merge):
    """Instantiate the base VL model and re-apply the training-time LoRA wrapper."""
    from transformers import AutoModelForImageTextToText

    name = cfg.model.backbone_name
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}.get(cfg.model.get("torch_dtype", "bfloat16"), torch.bfloat16)

    print(f"[export] loading base backbone {name} ({dtype})")
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=dtype, low_cpu_mem_usage=True, device_map=None,
        attn_implementation="eager",   # export is CPU-only; flash-attn needs CUDA
    )

    strategy = cfg.model.training_strategy
    if strategy != "lora":
        print(f"[export] training_strategy={strategy}; no LoRA wrapper applied")
        return model, False

    from peft import LoraConfig, get_peft_model
    lc = OmegaConf.to_container(cfg.model.lora_config, resolve=True)
    print(f"[export] applying LoRA r={lc['r']} alpha={lc['alpha']} targets={lc['target_modules']}")
    model = get_peft_model(model, LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc["dropout"]),
        target_modules=list(lc["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    ))
    return model, merge


def collect_bam_metadata(cfg):
    """Record the hyperparameters needed to rebuild the adapters from bam_adapters.pt."""
    bam = cfg.bam
    meta = {"task_type": bam.get("task_type", "qa"),
            "vl_use_native_video": bool(bam.get("vl_use_native_video", False)),
            "streams": {}}
    for s in STREAMS:
        if not bool(bam.get(f"use_bam_{s}", False)):
            continue
        meta["streams"][s] = {
            "feat_dim": bam.get(f"d_{s}_feat"),
            "hidden": bam.get(f"bam_hidden_{s}"),
            "temporal": bam.get(f"bam_{s}_temporal"),
            "p_moddrop": bam.get(f"bam_p_moddrop_{s}"),
            "use_ln": bool(bam.get(f"bam_{s}_use_ln", False)),
            "alpha_init": bam.get(f"bam_{s}_alpha_init", 1.0),
        }
    return meta


README = """---
license: other
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3_5_vl
- lora
---

# {name}

LoRA fine-tune of `{base}` for video question answering, exported from a BAM-VL
training checkpoint (`{ckpt}`, global_step {step}).

## Contents

- Merged backbone in the standard HF layout — loads with `AutoModelForImageTextToText`.
- `bam_adapters.pt` — the BAM side-channel adapters (facial / pose / audio) and CLS heads,
  with the hyperparameters needed to rebuild them.

## Important

The merged backbone alone does **not** reproduce training behaviour. During training the
model pools the penultimate hidden state, computes a delta from the side-channel adapters,
and adds it to the final hidden state before `lm_head`. A plain `generate()` call skips
this entirely. Faithful inference requires `bam_adapters.pt` together with the `BAMVLQA`
wrapper code and the corresponding facial/pose/audio feature extractors.

## Training

- Stage 1: `bam_only` — frozen backbone, side-channel adapters only.
- Stage 2: `bam_and_full_model` — LoRA (r={r}, alpha={alpha}) plus adapters.
"""


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    meta_path = os.path.join(args.ckpt, "meta.json")
    ckpt_meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            ckpt_meta = json.load(f)
        print(f"[export] checkpoint meta: {ckpt_meta}")
        strat = ckpt_meta.get("training_strategy")
        if strat and strat != cfg.model.training_strategy:
            raise SystemExit(
                f"[export] training_strategy mismatch: checkpoint was written with '{strat}' "
                f"but --config says '{cfg.model.training_strategy}'. The LoRA module tree "
                f"would not match; point --config at the config used for this run."
            )

    print(f"[export] reading {args.ckpt}")
    sd = load_flat_state_dict(args.ckpt)
    backbone_sd, adapter_sd, head_sd, other_sd = split_by_prefix(sd)
    print(f"[export] keys: backbone={len(backbone_sd)} adapters={len(adapter_sd)} "
          f"heads={len(head_sd)} other={len(other_sd)}")
    if not backbone_sd:
        raise SystemExit("[export] no 'backbone.*' keys found — is this a BAM-VL checkpoint?")
    if other_sd:
        print(f"[export] ignoring unrecognized keys: {sorted(other_sd)[:8]}")

    model, do_merge = build_backbone(cfg, merge=not args.no_merge)

    print("[export] loading trained weights into the model")
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False, assign=True)
    # PEFT keeps a second reference to each base weight, so some absences are structural.
    real_missing = [k for k in missing if "lora_" in k]
    if real_missing:
        raise SystemExit(
            f"[export] {len(real_missing)} LoRA weights missing from the checkpoint, e.g. "
            f"{real_missing[:3]}. The lora_config in --config does not match the run."
        )
    if unexpected:
        print(f"[export] {len(unexpected)} unexpected keys (first few): {unexpected[:5]}")
    print(f"[export] {len(missing)} missing keys (structural aliases expected)")

    if do_merge:
        print("[export] merging LoRA into the base weights")
        model = model.merge_and_unload()

    print(f"[export] writing model to {args.out_dir}")
    model.save_pretrained(args.out_dir, safe_serialization=True, max_shard_size=args.max_shard_size)

    print("[export] writing tokenizer + processor")
    from transformers import AutoProcessor, AutoTokenizer
    AutoTokenizer.from_pretrained(cfg.model.tokenizer_name).save_pretrained(args.out_dir)
    try:
        AutoProcessor.from_pretrained(cfg.model.processor_name).save_pretrained(args.out_dir)
    except Exception as e:
        print(f"[export] processor save skipped: {type(e).__name__}: {e}")

    bam_path = os.path.join(args.out_dir, "bam_adapters.pt")
    torch.save({
        "adapters": {k: v.clone() for k, v in adapter_sd.items()},
        "heads": {k: v.clone() for k, v in head_sd.items()},
        "bam_config": collect_bam_metadata(cfg),
        "checkpoint_meta": ckpt_meta,
        "backbone_name": cfg.model.backbone_name,
    }, bam_path)
    print(f"[export] wrote {bam_path}")

    lc = OmegaConf.to_container(cfg.model.lora_config, resolve=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write(README.format(
            name=os.path.basename(args.out_dir.rstrip("/")),
            base=cfg.model.backbone_name,
            ckpt=os.path.basename(args.ckpt.rstrip("/")),
            step=ckpt_meta.get("global_step", "?"),
            r=lc["r"], alpha=lc["alpha"],
        ))

    print(f"[export] done -> {args.out_dir}")
    print("[export] NOTE: the merged backbone does not apply the BAM delta; "
          "faithful inference needs bam_adapters.pt + the BAMVLQA wrapper.")


if __name__ == "__main__":
    main()
