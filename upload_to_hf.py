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
    """Split into backbone_sd (strip 'backbone.' prefix), heads_sd, and adapters_sd.

    BAM checkpoints additionally contain video_adapter.* and audio_adapter.* keys
    which are saved separately as adapters.bin.
    """
    backbone_sd = {k[len("backbone."):]: v
                   for k, v in state_dict.items() if k.startswith("backbone.")}
    heads_sd = {k: v for k, v in state_dict.items() if k.startswith("heads.")}
    adapters_sd = {k: v for k, v in state_dict.items()
                   if k.startswith("video_adapter.") or k.startswith("audio_adapter.")}
    return backbone_sd, heads_sd, adapters_sd


def load_state_dict(ckpt_dir: str) -> dict:
    """Load the full model state dict, including BAM adapter shards if present.

    BAM checkpoints saved by accelerate with multiple prepared modules use the
    model_order field in meta.json to map shard indices to component names:
      pytorch_model_fsdp.bin   → base model (backbone + heads)
      pytorch_model_fsdp_1.bin → model_order[1]  (e.g. video adapter)
      pytorch_model_fsdp_2.bin → model_order[2]  (e.g. audio adapter)

    Adapter weights are prefixed as {name}_adapter.* before merging.
    """
    ckpt_file = os.path.join(ckpt_dir, "pytorch_model_fsdp.bin")
    if not os.path.isfile(ckpt_file):
        ckpt_file = os.path.join(ckpt_dir, "pytorch_model_fsdp_0.bin")
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(
                f"Cannot find pytorch_model_fsdp.bin in {ckpt_dir}. "
                "If sharded across multiple ranks, consolidate first with "
                "FSDP's full_state_dict utility."
            )
    print(f"Loading base state dict from {ckpt_file} ...")
    sd = torch.load(ckpt_file, map_location="cpu")
    if isinstance(sd, dict) and "state" in sd and isinstance(sd["state"], dict):
        sd = sd["state"]
    combined = strip_fsdp_prefix(sd)

    # Load adapter shards based on model_order in meta.json
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model_order = meta.get("model_order", ["base"])
        for i, name in enumerate(model_order[1:], start=1):
            shard_file = os.path.join(ckpt_dir, f"pytorch_model_fsdp_{i}.bin")
            if not os.path.isfile(shard_file):
                print(f"[warn] Expected adapter shard {shard_file} for '{name}' but not found")
                continue
            print(f"Loading {name} adapter shard from {shard_file} ...")
            adapter_sd = torch.load(shard_file, map_location="cpu")
            prefix = f"{name}_adapter"
            for k, v in adapter_sd.items():
                combined[f"{prefix}.{k}"] = v
            print(f"  Added {len(adapter_sd)} keys as '{prefix}.*'")

    return combined


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
    p.add_argument("--readme_only", action="store_true",
                   help="Skip checkpoint loading; only regenerate and push README.md")
    p.add_argument("--task", default="sarcasm",
                   choices=["sarcasm", "emotion", "sentiment", "humour", "mental_health", "generic"],
                   help="Task type — controls README title, tags, and domain example")
    return p.parse_args()


# Task metadata used by _write_readme
_TASK_META = {
    "sarcasm": {
        "title":       "OmniSapiens BAM — Sarcasm Detection",
        "description": "multimodal sarcasm detection on the MUStARD/MMSD benchmark",
        "tags":        ["sarcasm-detection", "sarcasm"],
        "domain":      "sarcasm",
    },
    "emotion": {
        "title":       "OmniSapiens BAM — Emotion Recognition",
        "description": "multimodal emotion recognition",
        "tags":        ["emotion-recognition", "emotion"],
        "domain":      "emotion",
    },
    "sentiment": {
        "title":       "OmniSapiens BAM — Sentiment Polarity",
        "description": "multimodal sentiment polarity classification",
        "tags":        ["sentiment-analysis", "sentiment"],
        "domain":      "sentiment_intensity",
    },
    "humour": {
        "title":       "OmniSapiens BAM — Humour Detection",
        "description": "multimodal humour detection",
        "tags":        ["humor-detection", "humor"],
        "domain":      "humour",
    },
    "mental_health": {
        "title":       "OmniSapiens BAM — Mental Health Detection",
        "description": "multimodal mental health indicator detection",
        "tags":        ["mental-health"],
        "domain":      "mental_health_ptsd",
    },
    "generic": {
        "title":       "OmniSapiens BAM",
        "description": "multimodal human behavior understanding",
        "tags":        [],
        "domain":      "sarcasm",
    },
}


def _write_readme(args, backbone_save: str, has_adapters: bool = False) -> None:
    """Write README.md into backbone_save."""
    meta = _TASK_META.get(args.task, _TASK_META["generic"])

    base_tags = ["human-behavior", "multimodal", "qwen2.5-omni"] + meta["tags"]
    readme_lines = [
        "---",
        "language: en",
        "license: apache-2.0",
        "tags:",
    ] + [f"  - {t}" for t in base_tags]

    if args.dataset_repo:
        readme_lines += ["datasets:", f"  - {args.dataset_repo}"]
    readme_lines += [
        "---",
        "",
        f"# {meta['title']}",
        "",
        f"Fine-tuned [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) "
        f"for {meta['description']}. "
        "Uses LoRA adapters merged into the backbone and a lightweight classification head.",
    ]
    if args.dataset_repo:
        readme_lines += [
            "",
            "## Benchmark",
            f"Evaluated on [{args.dataset_repo}](https://huggingface.co/datasets/{args.dataset_repo}).",
        ]
    readme_lines += [
        "",
        "## Usage",
        "",
        "### Installation",
        "```bash",
        "pip install transformers torch huggingface_hub",
        "```",
        "",
        "### Classification",
        "",
        "```python",
        "import json, torch",
        "from huggingface_hub import hf_hub_download",
        "from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor",
        "",
        f'MODEL_ID = "{args.repo_id}"',
        "",
        "# 1. Load backbone and processor",
        "model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(",
        "    MODEL_ID, torch_dtype=torch.float16, device_map=\"auto\"",
        ")",
        "processor = AutoProcessor.from_pretrained(MODEL_ID)",
        "",
        "# 2. Load classification heads and label scheme",
        "heads_path = hf_hub_download(MODEL_ID, \"heads.bin\")",
        "label_path = hf_hub_download(MODEL_ID, \"label_scheme.json\")",
        "heads_sd = torch.load(heads_path, map_location=\"cpu\")",
        "with open(label_path) as f:",
        "    label_scheme = json.load(f)",
        "",
        "# 3. Reconstruct domain heads",
        "global_classes = label_scheme[\"meta\"][\"global_classes\"]  # {domain: [{index, label}, ...]}",
        "hidden_size = model.config.hidden_size",
        "domain_names = list(global_classes.keys())",
        "domain_heads = torch.nn.ModuleList([",
        "    torch.nn.Linear(hidden_size, len(global_classes[d])) for d in domain_names",
        "])",
        "domain_heads.load_state_dict({k.replace(\"heads.\", \"\"): v for k, v in heads_sd.items()})",
        "domain_heads.eval().to(model.device).to(torch.float16)",
        "domain_to_id = {d: i for i, d in enumerate(domain_names)}",
        "",
        "# 4. Prepare multimodal inputs",
        "# video_tensor: [T, C, H, W] tensor or list of PIL images",
        "# audio_waveform: 1-D numpy array / tensor at 16 kHz",
        f'domain = "{meta["domain"]}"',
        "messages = [{\"role\": \"user\", \"content\": [",
        "    {\"type\": \"video\"},",
        "    {\"type\": \"audio\"},",
        "    {\"type\": \"text\", \"text\": \"Classify the human behavior expressed.\"},",
        "]}]",
        "text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)",
        "inputs = processor(text=[text], videos=[video_tensor], audio=[audio_waveform], return_tensors=\"pt\")",
        "inputs = {k: v.to(model.device) for k, v in inputs.items()}",
        "",
        "# 5. Forward pass — pool penultimate hidden layer, route through domain head",
        "with torch.no_grad():",
        "    out = model(**inputs, output_hidden_states=True, use_cache=False)",
        "    h = out.hidden_states[-2]                            # [B, T, H]",
        "    mask = inputs[\"attention_mask\"].unsqueeze(-1).float()",
        "    pooled = (h * mask).sum(1) / mask.sum(1)             # [B, H]",
        "    logits = domain_heads[domain_to_id[domain]](pooled.float())  # [B, K_d]",
        "    pred_idx = logits.argmax(dim=-1).item()",
        "",
        "label_name = global_classes[domain][pred_idx][\"label\"]",
        "print(f\"Predicted {domain}: {label_name}\")",
        "```",
    ]
    if has_adapters:
        readme_lines += [
        "",
        "### Behavioral Descriptors (BAM Adapters)",
        "",
        "As `adapters.bin` is present in the repo, the model supports side-channel",
        "behavioral descriptors extracted from OpenPose (video) and OpenSmile (audio).",
        "These replace the raw video/audio inputs to the backbone with pre-computed",
        "behavioral feature vectors that are injected via lightweight MLP adapters.",
        "",
        "**Video — OpenPose keypoints**",
        "",
        "OpenPose produces a dict per clip with keys `pose`, `face`, `left_hand`, `right_hand`,",
        "each a `[T, K, 2or3]` tensor (T frames, K keypoints, x/y/conf).",
        "",
        "```python",
        "def prepare_video_feats(openpose_dict, temporal_mode=\"meanstd\"):",
        "    \"\"\"OpenPose dict → pooled feature vector [D_v_pooled].\"\"\"",
        "    parts = []",
        "    for key in (\"pose\", \"face\", \"left_hand\", \"right_hand\"):",
        "        t = openpose_dict.get(key)  # [T, K, 2or3]",
        "        if t is None: continue",
        "        t = torch.as_tensor(t).float()[..., :2]  # drop confidence, keep x/y",
        "        parts.append(t.reshape(t.shape[0], -1))  # [T, K*2]",
        "    seq = torch.cat(parts, dim=-1).float()       # [T, D_v]",
        "    if temporal_mode == \"meanstd\":",
        "        return torch.cat([seq.mean(0), seq.std(0)])  # [D_v*2]",
        "    return seq.mean(0)                               # [D_v]",
        "",
        "video_feats = prepare_video_feats(openpose_dict).unsqueeze(0)  # [1, D_v_pooled]",
        "```",
        "",
        "**Audio — OpenSmile features**",
        "",
        "OpenSmile produces a dict with key `features` → `[T, D_a]` or `[D_a]`.",
        "",
        "```python",
        "def prepare_audio_feats(opensmile_dict):",
        "    \"\"\"OpenSmile dict → L2-normalised feature vector [D_a].\"\"\"",
        "    x = torch.as_tensor(opensmile_dict[\"features\"]).float()",
        "    if x.ndim == 2: x = x.squeeze(0)  # [D_a] (single frame assumed)",
        "    return x / x.norm(p=2).clamp_min(1e-6)",
        "",
        "audio_feats = prepare_audio_feats(opensmile_dict).unsqueeze(0)  # [1, D_a]",
        "```",
        "",
        "**Loading and applying the adapters**",
        "",
        "```python",
        "import torch, torch.nn as nn",
        "from huggingface_hub import hf_hub_download",
        "",
        "adapters_sd = torch.load(hf_hub_download(MODEL_ID, \"adapters.bin\"), map_location=\"cpu\")",
        "",
        "# Infer architecture from saved weight shapes — no config needed",
        "def _make_adapter(prefix, sd):",
        "    w0 = sd[f\"{prefix}.mlp.0.weight\"]          # [hidden, feat_dim]",
        "    w2 = sd[f\"{prefix}.mlp.2.weight\"]          # [out_dim, hidden]",
        "    feat_dim, hidden, out_dim = w0.shape[1], w0.shape[0], w2.shape[0]",
        "    mlp = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))",
        "    alpha = nn.Parameter(sd[f\"{prefix}.alpha\"])",
        "    class _Adapter(nn.Module):",
        "        def __init__(self): super().__init__(); self.mlp = mlp; self.alpha = alpha",
        "        def forward(self, x): return self.mlp(x) * self.alpha",
        "    m = _Adapter()",
        "    m.load_state_dict({k[len(prefix)+1:]: v for k, v in sd.items() if k.startswith(prefix)}, strict=False)",
        "    return m.eval()",
        "",
        "video_adapter = _make_adapter(\"video_adapter\", adapters_sd).to(model.device).half()",
        "audio_adapter = _make_adapter(\"audio_adapter\", adapters_sd).to(model.device).half()",
        "",
        "# Augment pooled repr with BAM deltas before the classification head",
        "with torch.no_grad():",
        "    out = model(**inputs, output_hidden_states=True, use_cache=False)",
        "    h = out.hidden_states[-2]",
        "    mask = inputs[\"attention_mask\"].unsqueeze(-1).float()",
        "    pooled = (h * mask).sum(1) / mask.sum(1)                    # [B, H]",
        "    pooled = pooled + video_adapter(video_feats.to(model.device).half())",
        "    pooled = pooled + audio_adapter(audio_feats.to(model.device).half())",
        "    logits = domain_heads[domain_to_id[domain]](pooled.float())",
        "    pred_idx = logits.argmax(dim=-1).item()",
        "",
        "label_name = global_classes[domain][pred_idx][\"label\"]",
        "print(f\"Predicted {domain}: {label_name}\")",
        "```",
        ]  # end has_adapters block
    readme_path = os.path.join(backbone_save, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(readme_lines) + "\n")
    print(f"Wrote model card to {readme_path}")


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # README-only mode: regenerate and push just README.md
    # ------------------------------------------------------------------
    if args.readme_only:
        from huggingface_hub import HfApi
        readme_dir = os.path.join(args.save_dir, "readme_only")
        os.makedirs(readme_dir, exist_ok=True)
        _write_readme(args, readme_dir)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=os.path.join(readme_dir, "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update README.md",
        )
        print(f"\nREADME updated at: https://huggingface.co/{args.repo_id}")
        return

    # ------------------------------------------------------------------
    # 1. Load + split state dict
    # ------------------------------------------------------------------
    sd = load_state_dict(args.ckpt_dir)
    backbone_sd, heads_sd, adapters_sd = split_state_dict(sd)
    print(f"State dict: {len(backbone_sd)} backbone keys, {len(heads_sd)} head keys, "
          f"{len(adapters_sd)} adapter keys")

    if not backbone_sd:
        heads_sd = {k: v for k, v in sd.items() if k.startswith("heads.")}
        adapters_sd = {k: v for k, v in sd.items()
                       if k.startswith("video_adapter.") or k.startswith("audio_adapter.")}
        backbone_sd = {k: v for k, v in sd.items()
                       if not k.startswith("heads.") and not k.startswith("video_adapter.")
                       and not k.startswith("audio_adapter.")}
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
    # 4b. Save BAM adapters (video_adapter / audio_adapter)
    # ------------------------------------------------------------------
    if adapters_sd:
        adapters_file = os.path.join(backbone_save, "adapters.bin")
        print(f"Saving {len(adapters_sd)} adapter tensors to {adapters_file} ...")
        torch.save(adapters_sd, adapters_file)
    else:
        print("[info] No BAM adapter weights found in checkpoint.")

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
    _write_readme(args, backbone_save, has_adapters=bool(adapters_sd))

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
