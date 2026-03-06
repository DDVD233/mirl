#!/usr/bin/env python3
"""
Upload a trained SFT checkpoint (FSDP / accelerate format) to the HuggingFace Hub.

The checkpoint directory (step_XXXX/) must contain:
  pytorch_model_fsdp.bin   — consolidated model state dict
  meta.json                — training metadata

What gets pushed to your HF repo:
  - The backbone (Qwen2.5-Omni) saved with save_pretrained / push_to_hub
  - heads.bin              — classification head weights
  - label_scheme.json      — full label scheme (copied from your local path)
  - training_meta.json     — training metadata

LoRA handling:
  If the checkpoint contains LoRA weights (detected automatically), they are
  merged into the backbone using manual weight arithmetic:
    W_merged = W_base + lora_scaling * (lora_B @ lora_A)
  where lora_scaling = lora_alpha / r (default 2.0, matching all known training
  configs in this project). No PEFT installation required.

Usage:
    python upload_to_hf.py \\
        --ckpt_dir  /scratch/keane/.../step_4578 \\
        --repo_id   your-hf-username/your-model-name \\
        --backbone_name Qwen/Qwen2.5-Omni-7B \\
        --label_scheme /path/to/label_scheme.json \\
        [--lora_scaling 2.0]    # lora_alpha/r ratio; default 2.0
        [--private]             # make the HF repo private
        [--save_dir /tmp/upload_staging]
"""

import os
import json
import shutil
import argparse
import torch


# ---------------------------------------------------------------------------
# FSDP prefix stripping
# ---------------------------------------------------------------------------

def strip_fsdp_prefix(state_dict: dict) -> dict:
    """Strip accelerate FSDP wrapper prefixes (_fsdp_wrapped_module.*, module.*)."""
    prefixes = ("_fsdp_wrapped_module.", "_orig_mod.", "module.")
    cleaned = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        cleaned[k] = v
    return cleaned


def split_state_dict(state_dict: dict):
    """Split into backbone_sd (strip 'backbone.' prefix) and heads_sd."""
    backbone_sd = {k[len("backbone."):]: v
                   for k, v in state_dict.items() if k.startswith("backbone.")}
    heads_sd = {k: v for k, v in state_dict.items() if k.startswith("heads.")}
    return backbone_sd, heads_sd


def load_state_dict(ckpt_dir: str) -> dict:
    ckpt_file = os.path.join(ckpt_dir, "pytorch_model_fsdp.bin")
    if not os.path.isfile(ckpt_file):
        ckpt_file = os.path.join(ckpt_dir, "pytorch_model_fsdp_0.bin")
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(
                f"Cannot find pytorch_model_fsdp.bin in {ckpt_dir}. "
                "If sharded across multiple ranks, consolidate first with "
                "FSDP's full_state_dict utility."
            )
    print(f"Loading state dict from {ckpt_file} ...")
    sd = torch.load(ckpt_file, map_location="cpu")
    if isinstance(sd, dict) and "state" in sd and isinstance(sd["state"], dict):
        sd = sd["state"]
    return strip_fsdp_prefix(sd)


# ---------------------------------------------------------------------------
# LoRA manual merge (no PEFT required)
# ---------------------------------------------------------------------------

def manual_merge_lora(backbone_sd: dict, scaling: float) -> dict:
    """
    Merge LoRA adapter weights into base weights without PEFT.

    Input key format (backbone_sd after stripping 'backbone.' prefix):
      base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
      base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
      base_model.model.lm_head.weight        (non-LoRA, kept as-is)
      base_model.model.model.embed_tokens.weight  (non-LoRA, kept as-is)

    Output key format (standard HF Qwen2_5OmniThinkerForConditionalGeneration):
      model.layers.0.self_attn.q_proj.weight  (merged)
      lm_head.weight
      model.embed_tokens.weight
    """
    PEFT_PREFIX = "base_model.model."

    base_weights = {}  # module_path -> base tensor
    lora_a = {}        # module_path -> lora_A tensor
    lora_b = {}        # module_path -> lora_B tensor
    plain = {}         # clean_key -> tensor (non-LoRA keys)

    for k, v in backbone_sd.items():
        if ".base_layer.weight" in k:
            module = k.replace(".base_layer.weight", "")
            clean = module[len(PEFT_PREFIX):] if module.startswith(PEFT_PREFIX) else module
            base_weights[clean] = v
        elif ".base_layer.bias" in k:
            # Bias on a LoRA-targeted layer — keep as-is under the clean key
            module = k.replace(".base_layer.bias", "")
            clean = module[len(PEFT_PREFIX):] if module.startswith(PEFT_PREFIX) else module
            plain[clean + ".bias"] = v
        elif ".lora_A.default.weight" in k:
            module = k.replace(".lora_A.default.weight", "")
            clean = module[len(PEFT_PREFIX):] if module.startswith(PEFT_PREFIX) else module
            lora_a[clean] = v
        elif ".lora_B.default.weight" in k:
            module = k.replace(".lora_B.default.weight", "")
            clean = module[len(PEFT_PREFIX):] if module.startswith(PEFT_PREFIX) else module
            lora_b[clean] = v
        elif "lora_" not in k and "base_layer" not in k:
            # Regular weight (embed_tokens, lm_head, norm, etc.)
            clean = k[len(PEFT_PREFIX):] if k.startswith(PEFT_PREFIX) else k
            plain[clean] = v
        # Skip any other LoRA bookkeeping keys (e.g. lora_dropout, scaling attrs)

    merged = dict(plain)

    n_merged = 0
    for module_key, base_w in base_weights.items():
        weight_key = module_key + ".weight"
        if module_key in lora_a and module_key in lora_b:
            a = lora_a[module_key].float()
            b = lora_b[module_key].float()
            delta = scaling * (b @ a)
            merged[weight_key] = (base_w.float() + delta).to(base_w.dtype)
            n_merged += 1
        else:
            merged[weight_key] = base_w  # base_layer without LoRA (shouldn't happen)

    print(f"  Merged {n_merged} LoRA modules, kept {len(plain)} plain weights")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="Path to the step_XXXX checkpoint directory")
    p.add_argument("--repo_id", required=True,
                   help="HuggingFace repo id, e.g. your-username/model-name")
    p.add_argument("--backbone_name", default="Qwen/Qwen2.5-Omni-7B",
                   help="Original backbone name (used to load processor/tokenizer config)")
    p.add_argument("--label_scheme", default=None,
                   help="Path to label_scheme.json to include in the repo")
    p.add_argument("--lora_scaling", type=float, default=2.0,
                   help="LoRA scaling ratio lora_alpha/r used at training time "
                        "(default 2.0 = alpha 32/r 16 or alpha 64/r 32)")
    p.add_argument("--private", action="store_true",
                   help="Create the HF repo as private")
    p.add_argument("--save_dir", default="/tmp/hf_upload",
                   help="Local staging directory for the upload")
    p.add_argument("--dataset_repo", default=None,
                   help="HF dataset repo to cross-link in the model card "
                        "(e.g. keentomato/human_behaviour_atlas)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load + split state dict
    # ------------------------------------------------------------------
    sd = load_state_dict(args.ckpt_dir)
    backbone_sd, heads_sd = split_state_dict(sd)
    print(f"State dict: {len(backbone_sd)} backbone keys, {len(heads_sd)} head keys")

    if not backbone_sd:
        heads_sd = {k: v for k, v in sd.items() if k.startswith("heads.")}
        backbone_sd = {k: v for k, v in sd.items() if not k.startswith("heads.")}
        print(f"[fallback] treating {len(backbone_sd)} keys as backbone")

    # ------------------------------------------------------------------
    # 2. Load backbone (merging LoRA if detected)
    # ------------------------------------------------------------------
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor

    is_lora = any("lora_A" in k for k in backbone_sd)

    if is_lora:
        print(f"LoRA checkpoint detected. Merging with lora_scaling={args.lora_scaling} ...")
        model_sd = manual_merge_lora(backbone_sd, args.lora_scaling)
    else:
        print("No LoRA weights detected. Loading backbone weights directly.")
        model_sd = {k: v for k, v in backbone_sd.items()}

    print(f"Loading backbone architecture from {args.backbone_name} ...")
    backbone = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.backbone_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    missing, unexpected = backbone.load_state_dict(model_sd, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    # ------------------------------------------------------------------
    # 3. Save backbone + processor
    # ------------------------------------------------------------------
    backbone_save = os.path.join(args.save_dir, "backbone")
    print(f"Saving backbone to {backbone_save} ...")
    backbone.save_pretrained(backbone_save, safe_serialization=True)

    print("Saving processor ...")
    processor = AutoProcessor.from_pretrained(args.backbone_name)
    processor.save_pretrained(backbone_save)

    # ------------------------------------------------------------------
    # 4. Save classification heads
    # ------------------------------------------------------------------
    if heads_sd:
        heads_file = os.path.join(backbone_save, "heads.bin")
        print(f"Saving {len(heads_sd)} head tensors to {heads_file} ...")
        torch.save(heads_sd, heads_file)
    else:
        print("[info] No classification head weights found in checkpoint.")

    # ------------------------------------------------------------------
    # 5. Copy meta.json and label_scheme.json
    # ------------------------------------------------------------------
    meta = {}
    meta_src = os.path.join(args.ckpt_dir, "meta.json")
    if os.path.isfile(meta_src):
        shutil.copy(meta_src, os.path.join(backbone_save, "training_meta.json"))
        with open(meta_src) as f:
            meta = json.load(f)
        print(f"Training meta: step={meta.get('global_step')}, "
              f"epoch={meta.get('epoch')}, strategy={meta.get('training_strategy')}")

    if args.label_scheme and os.path.isfile(args.label_scheme):
        shutil.copy(args.label_scheme, os.path.join(backbone_save, "label_scheme.json"))
        print(f"Copied label_scheme.json from {args.label_scheme}")

    # ------------------------------------------------------------------
    # 6. Write model card (README.md)
    # ------------------------------------------------------------------
    readme_lines = [
        "---",
        "language: en",
        "license: apache-2.0",
        "tags:",
        "  - human-behavior",
        "  - multimodal",
        "  - qwen2.5-omni",
    ]
    if args.dataset_repo:
        readme_lines += ["datasets:", f"  - {args.dataset_repo}"]
    readme_lines += [
        "---",
        "",
        "# OmniSapiens SFT",
        "",
        f"Fine-tuned [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) "
        "for human behavior understanding.",
    ]
    if args.dataset_repo:
        readme_lines += [
            "",
            "## Benchmark",
            f"Evaluated on [{args.dataset_repo}](https://huggingface.co/datasets/{args.dataset_repo}).",
        ]
    readme_path = os.path.join(backbone_save, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(readme_lines) + "\n")
    print(f"Wrote model card to {readme_path}")

    # ------------------------------------------------------------------
    # 7. Push to HuggingFace Hub
    # ------------------------------------------------------------------
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\nPushing to HuggingFace Hub: {args.repo_id} (private={args.private}) ...")
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=backbone_save,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload SFT checkpoint step={meta.get('global_step', '?')}",
    )
    print(f"\nDone. Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
