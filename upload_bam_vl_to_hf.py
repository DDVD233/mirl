#!/usr/bin/env python3
"""
Convert a BAM-on-Qwen3-VL SFT checkpoint (accelerate FSDP format) into a
HuggingFace repo layout, and optionally upload it to the Hub.

This is the VL / three-adapter counterpart of upload_to_hf.py (which handles the
Qwen2.5-Omni two-adapter variant and is left untouched). The checkpoint directory
(step_XXXX/) must contain:
  pytorch_model_fsdp.bin   — consolidated wrapper state dict
  meta.json                — training metadata (model_order, global_step, ...)

Expected checkpoint key layout (verified against stage-2 step_2023):
  backbone.base_model.model.<hf_path>[.base_layer].weight     — PEFT-wrapped backbone
  backbone.base_model.model.<hf_path>.lora_{A,B}.default.weight
  heads.N.{weight,bias}                                        — classifier heads (NOT exported)
  {facial,pose,audio}_adapter.{alpha,mlp.0.*,mlp.2.*[,ln_feats.*]}

What gets staged into --save_dir (and pushed with --upload):
  model-XXXXX-of-XXXXX.safetensors (+ index)  — backbone with LoRA merged
                                                (W_merged = W_base + scaling * B @ A)
  config.json / generation_config.json / tokenizer / processor / chat template
                                              — copied from --backbone_name
  adapters.bin                                — the three BAM adapter modules, keys as-is
  bam_config.json                             — adapter structure (dims derived from the
                                                weights themselves) + pooling modes
  training_meta.json                          — original meta.json + conversion record

Classifier heads and the label scheme are intentionally NOT exported: the QA
generation path never routes through them.

Usage (conversion only — cheap key/shape validation first via --dry_run):
    python upload_bam_vl_to_hf.py --ckpt_dir .../step_2023 --dry_run
    python upload_bam_vl_to_hf.py --ckpt_dir .../step_2023 --save_dir /path/on/scratch

Upload (after inspecting the staged folder):
    python upload_bam_vl_to_hf.py --ckpt_dir .../step_2023 --save_dir /path/on/scratch \
        --repo_id your-user/bam-vl-qwen35-27b --upload [--private]
"""

import os
import json
import time
import argparse

import torch

STREAMS = ("facial", "pose", "audio")

# Pooling modes are a property of how features are fed to the adapters, not of the
# weights, so they cannot be derived from the checkpoint. These are the locked
# training choices (config_bam_vl_accelerate.yaml / VL_STREAM_SPECS).
DEFAULT_POOL_MODES = {"facial": "meanstd", "pose": "meanstd", "audio": "none"}

DEFAULT_BACKBONE = "Qwen/Qwen3.5-27B"
DEFAULT_TRAIN_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sft", "configs", "config_bam_vl_accelerate.yaml"
)

FSDP_PREFIXES = ("_fsdp_wrapped_module.", "_orig_mod.", "module.")
PEFT_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# Loading + key normalization
# ---------------------------------------------------------------------------

def strip_fsdp_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in FSDP_PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def load_checkpoint(ckpt_dir: str) -> dict:
    """mmap-load the consolidated state dict (metadata only until tensors are touched).

    Multi-shard saves (pytorch_model_fsdp_1.bin, ...) map shard i to
    meta.json model_order[i] and prefix its keys as {name}_adapter.* — same
    convention as the Omni script. The VL trainer attaches adapters to the model
    before prepare(), so current checkpoints are single-shard.
    """
    main = os.path.join(ckpt_dir, "pytorch_model_fsdp.bin")
    if not os.path.isfile(main):
        raise FileNotFoundError(f"missing {main}")
    sd = {strip_fsdp_prefixes(k): v
          for k, v in torch.load(main, map_location="cpu", mmap=True, weights_only=True).items()}

    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)
    model_order = meta.get("model_order", ["base"])

    for i, name in enumerate(model_order):
        if i == 0:
            continue
        shard = os.path.join(ckpt_dir, f"pytorch_model_fsdp_{i}.bin")
        if not os.path.isfile(shard):
            continue
        extra = torch.load(shard, map_location="cpu", mmap=True, weights_only=True)
        for k, v in extra.items():
            sd[f"{name}_adapter.{strip_fsdp_prefixes(k)}"] = v
        print(f"[load] merged shard {shard} as {name}_adapter.*  ({len(extra)} keys)")

    return sd, meta


def split_state_dict(sd: dict):
    """Split into backbone (backbone. stripped), adapters (as-is), heads (dropped)."""
    backbone, adapters, heads, unknown = {}, {}, {}, []
    for k, v in sd.items():
        if k.startswith("backbone."):
            backbone[k[len("backbone."):]] = v
        elif any(k.startswith(f"{s}_adapter.") for s in STREAMS):
            adapters[k] = v
        elif k.startswith("heads."):
            heads[k] = v
        else:
            unknown.append(k)
    if unknown:
        raise RuntimeError(f"unrecognized top-level keys (refusing to guess): {unknown[:10]}")
    return backbone, adapters, heads


# ---------------------------------------------------------------------------
# LoRA merge
# ---------------------------------------------------------------------------

def plan_lora_merge(backbone_sd: dict):
    """Compute the post-merge key mapping WITHOUT touching tensor data.

    Returns (plain: {out_key: in_key}, merges: {out_key: (base_key, A_key, B_key)}).
    """
    keys = set(backbone_sd.keys())
    plain, merges = {}, {}
    for k in keys:
        if ".lora_A." in k or ".lora_B." in k:
            continue  # consumed via their base_layer partner
        out = k[len(PEFT_PREFIX):] if k.startswith(PEFT_PREFIX) else k
        if ".base_layer." in k:
            module = k.split(".base_layer.")[0]           # PEFT-prefixed module path
            leaf = k.split(".base_layer.")[1]             # "weight" or "bias"
            out = out.replace(".base_layer.", ".")
            a = f"{module}.lora_A.default.{leaf}"
            b = f"{module}.lora_B.default.{leaf}"
            if leaf == "weight" and a in keys and b in keys:
                merges[out] = (k, a, b)
                continue
        plain[out] = k

    n_lora = sum(1 for k in keys if ".lora_A." in k)
    if len(merges) != n_lora:
        raise RuntimeError(f"LoRA accounting mismatch: {n_lora} lora_A keys but "
                           f"{len(merges)} planned merges — orphaned adapters?")
    return plain, merges


def materialize_merged(backbone_sd: dict, plain: dict, merges: dict, scaling: float) -> dict:
    """Build the merged state dict. Merges run in fp32, cast back to the base dtype."""
    out = {}
    for out_key, in_key in plain.items():
        out[out_key] = backbone_sd[in_key]
    for i, (out_key, (base_k, a_k, b_k)) in enumerate(sorted(merges.items())):
        w = backbone_sd[base_k]
        a = backbone_sd[a_k].float()
        b = backbone_sd[b_k].float()
        out[out_key] = (w.float() + scaling * (b @ a)).to(w.dtype)
        if (i + 1) % 64 == 0:
            print(f"[merge] {i + 1}/{len(merges)} modules merged")
    print(f"[merge] done: {len(merges)} LoRA modules merged (scaling={scaling}), "
          f"{len(plain)} passthrough tensors")
    return out


# ---------------------------------------------------------------------------
# Validation against the pretrained backbone
# ---------------------------------------------------------------------------

def find_reference_snapshot(backbone_name: str):
    """Locate the cached HF snapshot dir for the backbone (None if not cached)."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(backbone_name, local_files_only=True)
    except Exception as e:
        print(f"[validate] backbone snapshot not in local cache ({e}); "
              f"key/shape validation will be skipped")
        return None


def validate_against_reference(merged_keys, merged_shapes, snapshot_dir, config):
    """Diff merged key set (and shapes where available) against the reference index.

    Returns the list of reference keys to copy verbatim from the base snapshot:
    keys whose entire module family is absent from the training checkpoint (the
    training wrapper never instantiated that submodule — e.g. Qwen3.5's optional
    `mtp.*` multi-token-prediction head — so the base values are still correct
    and downstream loaders like vLLM get a key-complete repo). A missing key
    whose module family IS present in ours remains a hard error.
    """
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print("[validate] reference has no safetensors index; skipping")
        return []
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    ref_keys = set(weight_map.keys())
    ours = set(merged_keys)

    extra = ours - ref_keys
    # Tied-embedding special case: a tied lm_head is not stored in the reference.
    if "lm_head.weight" in extra and getattr(config, "tie_word_embeddings", False):
        print("[validate] note: lm_head.weight present here but tied (absent) in reference")
        extra.discard("lm_head.weight")

    missing = ref_keys - ours
    our_tops = {k.split(".")[0] for k in ours}
    copy_from_base = sorted(k for k in missing if k.split(".")[0] not in our_tops)
    missing -= set(copy_from_base)

    if extra or missing:
        raise RuntimeError(
            f"key-set mismatch vs {index_path}:\n"
            f"  extra ({len(extra)}): {sorted(extra)[:8]}\n"
            f"  missing ({len(missing)}): {sorted(missing)[:8]}"
        )
    if copy_from_base:
        fams = sorted({k.split(".")[0] + ".*" for k in copy_from_base})
        print(f"[validate] {len(copy_from_base)} tensors from module(s) {fams} are absent from "
              f"the training checkpoint (module never instantiated in training); they will be "
              f"copied verbatim from the base snapshot")
    print(f"[validate] key set matches reference ({len(ref_keys)} tensors, "
          f"{len(copy_from_base)} copied from base)")

    # Shape check via safetensors headers (no tensor data is read).
    try:
        from safetensors import safe_open
    except ImportError:
        print("[validate] safetensors not importable; skipping shape check")
        return copy_from_base
    by_file = {}
    for k, fname in weight_map.items():
        by_file.setdefault(fname, []).append(k)
    bad = []
    for fname, ks in by_file.items():
        with safe_open(os.path.join(snapshot_dir, fname), framework="pt") as f:
            for k in ks:
                ref_shape = tuple(f.get_slice(k).get_shape())
                if k in merged_shapes and tuple(merged_shapes[k]) != ref_shape:
                    bad.append((k, merged_shapes[k], ref_shape))
    if bad:
        raise RuntimeError(f"shape mismatches vs reference: {bad[:8]}")
    print(f"[validate] all tensor shapes match reference")
    return copy_from_base


def load_base_tensors(snapshot_dir: str, keys: list) -> dict:
    """Read the given tensors verbatim from the base snapshot's safetensors shards."""
    if not keys:
        return {}
    from safetensors import safe_open
    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_file = {}
    for k in keys:
        by_file.setdefault(weight_map[k], []).append(k)
    out = {}
    for fname, ks in by_file.items():
        with safe_open(os.path.join(snapshot_dir, fname), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    print(f"[stage] copied {len(out)} untouched tensors from the base snapshot")
    return out


# ---------------------------------------------------------------------------
# BAM adapter config (dims derived from the weights — the source of truth)
# ---------------------------------------------------------------------------

def build_bam_config(adapters_sd: dict, meta: dict,
                     backbone_name: str, scaling: float, train_config_path: str):
    pool_modes = dict(DEFAULT_POOL_MODES)
    yaml_dims = {}
    if train_config_path and os.path.isfile(train_config_path):
        try:
            import yaml
            with open(train_config_path) as f:
                bam = (yaml.safe_load(f) or {}).get("bam", {})
            for s in STREAMS:
                pool_modes[s] = bam.get(f"bam_{s}_temporal", pool_modes[s])
                yaml_dims[s] = bam.get(f"d_{s}_feat")
        except Exception as e:
            print(f"[bam_config] could not read training YAML ({e}); using locked defaults")

    streams, out_dims = {}, set()
    for s in STREAMS:
        w0 = adapters_sd.get(f"{s}_adapter.mlp.0.weight")
        w2 = adapters_sd.get(f"{s}_adapter.mlp.2.weight")
        if w0 is None or w2 is None:
            print(f"[bam_config] stream '{s}' absent from checkpoint; skipping")
            continue
        feat_dim, hidden, out_dim = w0.shape[1], w0.shape[0], w2.shape[0]
        out_dims.add(int(out_dim))
        if yaml_dims.get(s) is not None and int(yaml_dims[s]) != int(feat_dim):
            raise RuntimeError(f"{s}: training YAML says d_feat={yaml_dims[s]} but adapter "
                               f"weights have feat_dim={feat_dim}")
        streams[s] = {
            "feat_dim": int(feat_dim),
            "hidden": int(hidden),
            "use_ln": f"{s}_adapter.ln_feats.weight" in adapters_sd,
            "pool_mode": pool_modes[s],
            "alpha": float(adapters_sd[f"{s}_adapter.alpha"].float().item()),
        }
    if len(out_dims) > 1:
        raise RuntimeError(f"adapters disagree on out_dim: {out_dims}")

    return {
        "format_version": 1,
        "streams": streams,
        "out_dim_hidden": out_dims.pop() if out_dims else None,
        "delta_rule": ("delta = sum_i mask_i * adapter_i(pooled_feats_i); constant per sample; "
                       "inject by adding delta to the input of the backbone's final norm "
                       "(lm_head(norm(h_last + delta)))"),
        "backbone_name": backbone_name,
        "lora_merged": True,
        "lora_scaling": scaling,
        # Deliberately no filesystem paths here: these files ship in the public HF repo.
        "source_global_step": meta.get("global_step"),
        "converted_at_unix": time.time(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", required=True, help="step_XXXX checkpoint directory")
    ap.add_argument("--backbone_name", default=DEFAULT_BACKBONE)
    ap.add_argument("--lora_scaling", type=float, default=None,
                    help="alpha/r; default: read from --train_config, fallback 2.0")
    ap.add_argument("--train_config", default=DEFAULT_TRAIN_CONFIG,
                    help="training YAML (pool modes, dim cross-check, lora scaling)")
    ap.add_argument("--save_dir", default=None,
                    help="staging dir for the HF layout (default: <ckpt_dir>_hf_export)")
    ap.add_argument("--max_shard_size", default="4GB")
    ap.add_argument("--dry_run", action="store_true",
                    help="key accounting + validation only; writes nothing, minimal RAM")
    ap.add_argument("--repo_id", default=None, help="HF repo id (required with --upload)")
    ap.add_argument("--upload", action="store_true", help="push save_dir to the Hub")
    ap.add_argument("--private", action="store_true", help="create the HF repo as private")
    args = ap.parse_args()

    # --- scaling ---
    scaling = args.lora_scaling
    if scaling is None and os.path.isfile(args.train_config):
        try:
            import yaml
            with open(args.train_config) as f:
                lc = (yaml.safe_load(f) or {}).get("model", {}).get("lora_config", {})
            if lc.get("alpha") and lc.get("r"):
                scaling = float(lc["alpha"]) / float(lc["r"])
                print(f"[scaling] from {args.train_config}: alpha={lc['alpha']} r={lc['r']} "
                      f"-> scaling={scaling}")
        except Exception as e:
            print(f"[scaling] could not read training YAML ({e})")
    if scaling is None:
        scaling = 2.0
        print("[scaling] WARNING: using fallback scaling=2.0")

    # --- load + split ---
    sd, meta = load_checkpoint(args.ckpt_dir)
    backbone_sd, adapters_sd, heads_sd = split_state_dict(sd)
    print(f"[split] backbone={len(backbone_sd)} adapters={len(adapters_sd)} "
          f"heads={len(heads_sd)} (heads are not exported)")

    # --- plan merge (no tensor math yet) ---
    plain, merges = plan_lora_merge(backbone_sd)
    merged_keys = list(plain.keys()) + list(merges.keys())
    merged_shapes = {k: tuple(backbone_sd[v].shape) for k, v in plain.items()}
    merged_shapes.update({k: tuple(backbone_sd[base].shape) for k, (base, _, _) in merges.items()})

    # --- bam config (also validates adapter shapes vs training YAML) ---
    bam_config = build_bam_config(adapters_sd, meta,
                                  args.backbone_name, scaling, args.train_config)
    print("[bam_config] " + json.dumps(bam_config["streams"], indent=2))

    # --- validate against reference snapshot ---
    from transformers import AutoConfig
    snapshot_dir = find_reference_snapshot(args.backbone_name)
    config = AutoConfig.from_pretrained(snapshot_dir or args.backbone_name)
    copy_from_base = []
    if snapshot_dir:
        copy_from_base = validate_against_reference(merged_keys, merged_shapes, snapshot_dir, config) or []

    if args.dry_run:
        print("[dry_run] all checks passed; nothing written. "
              f"Would stage {len(merged_keys)} merged + {len(copy_from_base)} base-copied "
              f"backbone tensors + {len(adapters_sd)} adapter tensors into "
              f"{args.save_dir or args.ckpt_dir.rstrip('/') + '_hf_export'}")
        return

    # --- materialize + stage ---
    save_dir = args.save_dir or args.ckpt_dir.rstrip("/") + "_hf_export"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[stage] staging into {save_dir}")

    merged = materialize_merged(backbone_sd, plain, merges, scaling)
    del sd, backbone_sd
    if copy_from_base:
        merged.update(load_base_tensors(snapshot_dir, copy_from_base))

    from huggingface_hub import save_torch_state_dict
    save_torch_state_dict(merged, save_dir, max_shard_size=args.max_shard_size)
    print(f"[stage] backbone shards written")
    del merged

    config.save_pretrained(save_dir)
    from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
    AutoProcessor.from_pretrained(snapshot_dir or args.backbone_name).save_pretrained(save_dir)
    AutoTokenizer.from_pretrained(snapshot_dir or args.backbone_name).save_pretrained(save_dir)
    try:
        GenerationConfig.from_pretrained(snapshot_dir or args.backbone_name).save_pretrained(save_dir)
    except Exception:
        print("[stage] no generation_config on the base model; skipping")

    torch.save(adapters_sd, os.path.join(save_dir, "adapters.bin"))
    with open(os.path.join(save_dir, "bam_config.json"), "w") as f:
        json.dump(bam_config, f, indent=2)
    with open(os.path.join(save_dir, "training_meta.json"), "w") as f:
        json.dump({"training_meta": meta,
                   "conversion": {k: v for k, v in bam_config.items() if k != "streams"}}, f, indent=2)
    print("[stage] adapters.bin, bam_config.json, training_meta.json written")

    # --- upload ---
    if args.upload:
        if not args.repo_id:
            raise SystemExit("--upload requires --repo_id")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        print(f"[upload] pushing {save_dir} -> {args.repo_id} (resumable; re-run to resume)")
        api.upload_large_folder(repo_id=args.repo_id, folder_path=save_dir, repo_type="model")
        print(f"[upload] done: https://huggingface.co/{args.repo_id}")
    else:
        print(f"[done] staged only. Inspect {save_dir}, then upload with:\n"
              f"  python {os.path.basename(__file__)} --ckpt_dir {args.ckpt_dir} "
              f"--save_dir {save_dir} --repo_id <user/repo> --upload [--private]")


if __name__ == "__main__":
    main()
