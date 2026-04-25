"""
Zero-shot inference for Qwen2.5-Omni-based models over multimodal JSONL datasets.

Speed features:
  - Flash Attention 2 (auto-detected, falls back to SDPA then eager)
  - device_map="auto": model-parallel across all visible GPUs
  - Batch inference: --batch_size N groups N samples per forward pass
  - Dataset sharding: --num_shards K --shard_idx I splits work across K processes
    (run_inference.sh launches one process per GPU automatically)
  - torch.compile: --torch_compile (Torch >= 2.0, ~20-30% speedup after warm-up)
  - Periodic checkpointing: output is flushed every --save_every N samples

Usage (single GPU):
    python inference.py \
        --model PhilipC/HumanOmniV2 \
        --input_jsonl data/zero_shot_data/test_eatd_prompts.jsonl \
        --output_jsonl /tmp/out/HumanOmniV2_eatd.jsonl

Usage (multi-GPU, shard 0 of 4 on GPU 0):
    CUDA_VISIBLE_DEVICES=0 python inference.py \
        --model PhilipC/HumanOmniV2 \
        --input_jsonl data/zero_shot_data/test_mvsa_prompts.jsonl \
        --output_jsonl /tmp/out/HumanOmniV2_mvsa_shard0.jsonl \
        --num_shards 4 --shard_idx 0 --batch_size 4

Merge shards afterward with:
    python inference.py --merge_shards /tmp/out/HumanOmniV2_mvsa_shard*.jsonl \
        --output_jsonl /tmp/out/HumanOmniV2_mvsa_merged.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys
import traceback

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# ── Thinking prompt  (mirrors verl/examples/format_prompt/default.jinja) ─────
THINKING_INSTRUCTION = (
    " You FIRST think about the reasoning process as an internal monologue and then "
    "provide the final answer. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. The final answer MUST BE put in \\boxed{}."
)
NO_THINKING_INSTRUCTION = (
    " You MUST provide the final answer directly without any extra information. "
    "Enclose the final answer in \\boxed{}."
)
# Gemma 4 uses <|think|> in the system prompt to trigger thinking;
# the model manages its own reasoning tags, so only ask for \boxed{}.
GEMMA4_THINKING_INSTRUCTION = (
    " Provide the final answer in \\boxed{}."
)


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Try \\boxed{...}, then <answer>...</answer>, then last non-empty line."""
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>([^<]+)</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ── Media loaders ─────────────────────────────────────────────────────────────

def _resolve(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


# ── Default loaders (decord + soundfile; compatible with HumanOmniV2 etc.) ───

def _default_load_audio_list(audio_paths: list[str], base_dir: str, target_sr: int = 16000):
    """Load each audio file, resample to target_sr, return as a list of 1-D arrays."""
    if not audio_paths:
        return None
    try:
        import soundfile as sf
        import librosa
    except ImportError:
        raise ImportError("pip install soundfile librosa")

    arrays = []
    for p in audio_paths:
        arr, sr = sf.read(_resolve(p, base_dir), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        arrays.append(arr)
    return arrays if arrays else None


def load_images(image_paths: list[str], base_dir: str) -> list[Image.Image]:
    return [Image.open(_resolve(p, base_dir)).convert("RGB") for p in image_paths]


def _video_num_frames_override(vframes_list: list, processor) -> int | None:
    """Return a num_frames override when any video is shorter than processor requires.

    Gemma4's video processor has num_frames=32 in its config and raises when the
    actual frame count is lower. Passing num_frames=<actual> as a processor kwarg
    overrides that config for the call. No-op for processors that don't expose
    num_frames (e.g. HumanOmniV2, Qwen). For batches, uses the minimum across all
    videos so every video in the batch can be sampled uniformly.
    """
    vp = getattr(processor, "video_processor", None)
    required = getattr(vp, "num_frames", None)
    if not required or not vframes_list:
        return None
    min_frames = min(len(v) for v in vframes_list)
    return min_frames if min_frames < required else None


def _default_load_video_frames(video_paths: list[str], base_dir: str,
                                fps: float = 1.0, max_frames: int = 32) -> list:
    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError("pip install decord")
    frames = []
    for p in video_paths:
        vr = VideoReader(_resolve(p, base_dir), ctx=cpu(0))
        stride = max(1, int(vr.get_avg_fps() / fps))
        indices = list(range(0, len(vr), stride))[:max_frames]
        frames.extend(vr[i].asnumpy() for i in indices)
    return frames


# ── Verl-style loaders (qwen_vl_utils + torchaudio; matches harpo/omnisapiens training) ──

def _verl_load_video(path: str, base_dir: str, nframes: int = 4,
                     min_pixels: int = 147456, max_pixels: int = 147456) -> torch.Tensor:
    """Returns [T,3,H,W] uint8 tensor via qwen_vl_utils.fetch_video — matches training."""
    _verl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if _verl_dir not in sys.path:
        sys.path.insert(0, _verl_dir)
    from verl.utils.dataset.vision_utils import process_video
    video_dict = {
        "type": "video",
        "video": _resolve(path, base_dir),
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "nframes": nframes,
    }
    return process_video(video_dict)  # handles errors internally, returns dummy on failure


def _verl_load_audio(path: str, base_dir: str, max_seconds: float = 10.0):
    """Returns (numpy_float32, sr) via torchaudio — matches training (clips to 10 s)."""
    _verl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if _verl_dir not in sys.path:
        sys.path.insert(0, _verl_dir)
    from verl.utils.dataset.audio_utils import process_audio
    tensor, sr = process_audio(_resolve(path, base_dir), processor=None, max_seconds=max_seconds)
    return tensor.numpy().astype("float32"), sr


# ── Entry → content list + flat media collectors ─────────────────────────────

def build_entry_inputs(entry: dict, base_dir: str, thinking: bool,
                       data_loading: str = "default", model_name: str = "",
                       gemma_legacy_thinking: bool = False):
    """
    Returns (content_list, audio_list_or_None, pil_images, video_frames).

    data_loading="verl_style":
        Parses <image>/<video>/<audio> tags in text order and places media at
        those positions — mirrors _build_messages() in rl_dataset.py.
        Uses qwen_vl_utils + torchaudio preprocessing (matches harpo/omnisapiens
        training pipeline).

    data_loading="default":
        Original behaviour (images prepended, video appended, audio external).
        Compatible with HumanOmniV2 and other models not trained via verl.
        Includes the <video> tag bug-fix (was incorrectly checking for [video]).
    """
    is_gemma = "gemma" in model_name.lower()
    if is_gemma and thinking and not gemma_legacy_thinking:
        instruction = GEMMA4_THINKING_INSTRUCTION
    elif thinking:
        instruction = THINKING_INSTRUCTION
    else:
        instruction = NO_THINKING_INSTRUCTION
    problem = entry.get("problem", "")

    if data_loading == "verl_style":
        images = [load_images([p], base_dir)[0] for p in entry.get("images", [])]
        videos = [_verl_load_video(p, base_dir) for p in entry.get("videos", [])]
        audios_raw = [_verl_load_audio(p, base_dir) for p in entry.get("audios", [])]

        img_idx = vid_idx = aud_idx = 0
        content = []
        for seg in re.split(r"(<image>|<video>|<audio>)", problem):
            if seg == "<image>" and img_idx < len(images):
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
            elif seg == "<video>" and vid_idx < len(videos):
                content.append({"type": "video", "video": videos[vid_idx].numpy()})
                vid_idx += 1
            elif seg == "<audio>" and aud_idx < len(audios_raw):
                arr, _sr = audios_raw[aud_idx]
                content.append({"type": "audio", "audio": arr})
                aud_idx += 1
            elif seg:
                content.append({"type": "text", "text": seg})
        content.append({"type": "text", "text": instruction})

        all_audios = [c["audio"] for c in content if c["type"] == "audio"]
        all_images = [c["image"] for c in content if c["type"] == "image"]
        all_videos = [c["video"] for c in content if c["type"] == "video"]
        return content, (all_audios if all_audios else None), all_images, all_videos

    else:  # default
        content = []
        pil_images = []
        video_frames = []

        is_gemma = "gemma" in model_name.lower()
        if entry.get("audios"):
            audio_list = _default_load_audio_list(entry["audios"], base_dir)
        else:
            audio_list = None

        if entry.get("images"):
            pil_images = load_images(entry["images"], base_dir)
            for img in pil_images:
                content.append({"type": "image", "image": img})

        # BUG FIX: was "[video]" — data uses <video> tags
        if entry.get("videos") and "<video>" in problem:
            video_frames = _default_load_video_frames(entry["videos"], base_dir)
            if video_frames:
                content.append({"type": "video", "video": video_frames})

        # Gemma's processor checks that audio soft tokens in input_ids match the
        # number of extracted audio features. Embedding audio dicts in the content
        # list causes apply_chat_template to insert those tokens; passing audio
        # as a bare kwarg (with no tokens in the text) causes a count mismatch.
        if is_gemma and audio_list:
            if "<audio>" in problem:
                aud_idx = 0
                for seg in re.split(r"(<audio>)", problem):
                    if seg == "<audio>" and aud_idx < len(audio_list):
                        content.append({"type": "audio", "audio": audio_list[aud_idx]})
                        aud_idx += 1
                    elif seg:
                        content.append({"type": "text", "text": seg})
                content.append({"type": "text", "text": instruction})
            else:
                for arr in audio_list:
                    content.append({"type": "audio", "audio": arr})
                content.append({"type": "text", "text": problem + instruction})
        else:
            content.append({"type": "text", "text": problem + instruction})

        return content, audio_list, pil_images, video_frames


# ── Model loading ─────────────────────────────────────────────────────────────

def _fix_meta_params(model):
    """Materialise any parameters/buffers left on the meta device after device_map dispatch.

    device_map="auto" bootstraps weights as meta tensors then loads from checkpoint.
    Parameters absent from the checkpoint (e.g. Gemma4's pad_embedding) are never
    dispatched and stay on meta, causing a device mismatch at forward time.
    Zero-init is safe: pad_embedding is a learnable placeholder initialised to zeros.
    """
    target_device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            print(f"  [fix] materialising meta param {name!r} on {target_device}")
            param.data = torch.zeros(param.shape, dtype=torch.bfloat16, device=target_device)
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            print(f"  [fix] materialising meta buffer {name!r} on {target_device}")
            buf.data = torch.zeros(buf.shape, dtype=torch.bfloat16, device=target_device)


def load_model(model_name: str, torch_compile: bool = False):
    print(f"Loading model: {model_name}")

    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    def _load(model_cls, attn_impl):
        kw = {**load_kwargs, "attn_implementation": attn_impl}
        return model_cls.from_pretrained(model_name, **kw)

    def _try_attn_impls(model_cls):
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                m = _load(model_cls, attn_impl)
                print(f"  Attention implementation: {attn_impl}")
                return m
            except Exception as exc:
                if attn_impl == "eager":
                    raise
                print(f"  {attn_impl} unavailable ({exc.__class__.__name__}), trying next...")

    # Try AutoModelForCausalLM first.
    # Gemma 4 doesn't accept attn_implementation kwargs; retry without them.
    # Other unrecognised architectures fall back to Qwen2_5OmniThinkerForConditionalGeneration.
    is_gemma = "gemma" in model_name.lower()
    try:
        model = _try_attn_impls(AutoModelForCausalLM)
    except ValueError as exc:
        exc_lower = str(exc).lower()
        if is_gemma:
            print("  Retrying Gemma without attn_implementation...")
            gemma_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, **gemma_kwargs)
            _fix_meta_params(model)
        elif "does not recognize this architecture" in exc_lower or "model type" in exc_lower:
            try:
                from transformers import Qwen2_5OmniThinkerForConditionalGeneration as OmniCls
            except ImportError:
                raise RuntimeError(
                    "transformers does not recognise 'qwen2_5_omni_thinker'. "
                    "Run: pip install --upgrade transformers"
                ) from exc
            print("  Falling back to Qwen2_5OmniThinkerForConditionalGeneration")
            model = _try_attn_impls(OmniCls)
        else:
            raise

    # Gemma4 bug: _update_causal_mask returns None for sdpa/flash_attention_2, which
    # then crashes in _convert_4d_mask_to_blocked_5d(None). Force eager so the method
    # always returns a real 4D causal mask. Patch both the outer config and the nested
    # text_config (for VLM wrappers like Gemma4ForConditionalGeneration).
    if is_gemma:
        for cfg in (model.config, getattr(model.config, 'text_config', None)):
            if cfg is not None and hasattr(cfg, '_attn_implementation'):
                old = cfg._attn_implementation
                if old != 'eager':
                    cfg._attn_implementation = 'eager'
                    print(f"  Patched {type(cfg).__name__}._attn_implementation: {old!r} → 'eager'")

    model.eval()

    if torch_compile:
        print("  Applying torch.compile (first batch will be slow — this is expected)...")
        model = torch.compile(model, mode="reduce-overhead")

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except ValueError as exc:
        if "unrecognized processing class" not in str(exc).lower():
            raise
        print("  AutoProcessor unavailable, falling back to AutoTokenizer...")
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    visible = set(p.device for p in model.parameters())
    print(f"  Devices: {visible}")
    return model, processor


# ── Batch inference ───────────────────────────────────────────────────────────

def _get_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_kwargs(processor) -> dict:
    """Return eos/pad token kwargs so generate() stops at the right token."""
    tok = getattr(processor, "tokenizer", processor)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    return {"eos_token_id": eos_id, "pad_token_id": pad_id}


def run_batch(model, processor, entries: list[dict], base_dir: str,
              thinking: bool, max_new_tokens: int,
              data_loading: str = "default",
              sampling_kwargs: dict | None = None,
              model_name: str = "",
              num_frames: int | None = None,
              gemma_legacy_thinking: bool = False) -> list[str]:
    """
    Run inference on a list of entries as a single batched forward pass.
    All entries should share the same modality_signature for reliable batching.
    Falls back to one-at-a-time on any processor error.
    """
    _skw = sampling_kwargs if sampling_kwargs is not None else {"do_sample": False}
    if len(entries) == 1:
        return [_run_one(model, processor, entries[0], base_dir, thinking,
                         max_new_tokens, data_loading, _skw, model_name, num_frames,
                         gemma_legacy_thinking)]

    try:
        texts, batch_audios, batch_images, batch_videos = [], [], [], []

        _is_gemma = "gemma" in model_name.lower()
        for entry in entries:
            content, audio_list, imgs, vframes = build_entry_inputs(
                entry, base_dir, thinking, data_loading, model_name, gemma_legacy_thinking)
            if _is_gemma and thinking and not gemma_legacy_thinking:
                msgs = [{"role": "system", "content": "<|think|>"}, {"role": "user", "content": content}]
            else:
                msgs = [{"role": "user", "content": content}]
            texts.append(
                processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
            if audio_list is not None:
                batch_audios.extend(audio_list)
            if _is_gemma:
                if imgs:
                    batch_images.append(imgs)
            else:
                batch_images.extend(imgs)
            if vframes:
                batch_videos.append(vframes)

        proc_kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if batch_audios:
            proc_kwargs["audio"] = batch_audios
        if batch_images:
            proc_kwargs["images"] = batch_images
        if batch_videos:
            proc_kwargs["videos"] = batch_videos
            nf = num_frames if num_frames is not None else _video_num_frames_override(batch_videos, processor)
            if nf is not None:
                proc_kwargs["num_frames"] = nf

        device = _get_device(model)
        inputs = processor(**proc_kwargs)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if inputs.get("attention_mask") is None and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with torch.inference_mode():
            raw = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **_skw,
                **_generate_kwargs(processor),
            )

        # Qwen2_5OmniThinkerForConditionalGeneration returns (text_ids, audio); unwrap if needed
        output_ids = raw[0] if isinstance(raw, (tuple, list)) else raw

        # Strip prompt tokens — with left-padding all prompts end at the same column
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_len:]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)

    except Exception as exc:
        # Batching failed (e.g. mixed modalities or processor limitation); fall back
        print(f"\n[WARN] Batch of {len(entries)} failed ({exc.__class__.__name__}: {exc}); "
              "retrying one-by-one.")
        return [_run_one(model, processor, e, base_dir, thinking, max_new_tokens, data_loading, _skw, model_name, num_frames, gemma_legacy_thinking)
                for e in entries]


def _run_one(model, processor, entry: dict, base_dir: str,
             thinking: bool, max_new_tokens: int,
             data_loading: str = "default",
             sampling_kwargs: dict | None = None,
             model_name: str = "",
             num_frames: int | None = None,
             gemma_legacy_thinking: bool = False) -> str:
    content, audio_list, imgs, vframes = build_entry_inputs(
        entry, base_dir, thinking, data_loading, model_name, gemma_legacy_thinking)
    _is_gemma = "gemma" in model_name.lower()
    if _is_gemma and thinking and not gemma_legacy_thinking:
        msgs = [{"role": "system", "content": "<|think|>"}, {"role": "user", "content": content}]
    else:
        msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    proc_kwargs = dict(text=text, return_tensors="pt", padding=True)
    if audio_list is not None:
        proc_kwargs["audio"] = audio_list
    if imgs:
        proc_kwargs["images"] = imgs
    if vframes:
        proc_kwargs["videos"] = [vframes]
        nf = num_frames if num_frames is not None else _video_num_frames_override([vframes], processor)
        if nf is not None:
            proc_kwargs["num_frames"] = nf

    device = _get_device(model)
    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    if inputs.get("attention_mask") is None and "input_ids" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    _skw = sampling_kwargs if sampling_kwargs is not None else {"do_sample": False}
    with torch.inference_mode():
        raw = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **_skw,
            **_generate_kwargs(processor),
        )

    # Qwen2_5OmniThinkerForConditionalGeneration returns (text_ids, audio); unwrap if needed
    out_ids = raw[0] if isinstance(raw, (tuple, list)) else raw

    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(out_ids[0][prompt_len:], skip_special_tokens=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _log_wandb(args, dataset_name: str, metrics: dict):
    """Log metrics to W&B if --wandb_project is set. One run per model (resume='allow')."""
    if not getattr(args, "wandb_project", None):
        return
    try:
        import wandb
        run_id   = getattr(args, "wandb_run_id",   None) or None
        run_name = getattr(args, "wandb_run_name", None) or run_id
        entity   = getattr(args, "wandb_entity",   None) or None
        model_name = getattr(args, "model", "unknown")
        wandb.init(
            project=args.wandb_project,
            entity=entity,
            id=run_id,
            resume="allow",
            name=run_name,
            config={"model": model_name},
        )
        wandb.log({f"{dataset_name}/{k}": v for k, v in metrics.items()})
        wandb.finish()
    except Exception as exc:
        print(f"[WARN] W&B logging failed: {exc}")


def compute_and_print_metrics(results: list[dict], model_key: str, model_name: str,
                              output_jsonl: str, args=None):
    preds = [e[model_key] for e in results if e.get(model_key)]
    gts   = [e["answer"]  for e in results if e.get(model_key)]
    if not preds:
        print("No valid predictions — skipping metrics.")
        return

    preds = [p.lower() for p in preds]
    gts   = [g.lower() for g in gts]
    acc = accuracy_score(gts, preds)
    wf1 = f1_score(gts, preds, average="weighted", zero_division=0)
    dataset_name = results[0].get("dataset", "unknown") if results else "unknown"

    print(f"\n{'='*54}")
    print(f"  Dataset   : {dataset_name}")
    print(f"  Model     : {model_name}")
    print(f"  N samples : {len(preds)}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  WF1       : {wf1:.4f}")
    print(f"{'='*54}\n")

    metrics = {"accuracy": acc, "weighted_f1": wf1, "n_samples": len(preds)}
    metrics_path = re.sub(r"\.jsonl$", "_metrics.json", output_jsonl)
    with open(metrics_path, "w") as f:
        json.dump({"model": model_name, "dataset": dataset_name, **metrics}, f, indent=2)
    print(f"Metrics → {metrics_path}")
    _log_wandb(args, dataset_name, metrics)


def _parse_multilabel(text: str) -> list[str]:
    return sorted([lbl.strip().lower() for lbl in text.split(",") if lbl.strip()])


def compute_and_print_metrics_multilabel(results: list[dict], model_key: str, model_name: str,
                                         output_jsonl: str, args=None):
    """Multilabel metrics (subset/exact-match accuracy + weighted F1) for comma-separated labels."""
    valid = [e for e in results if e.get(model_key)]
    if not valid:
        print("No valid predictions — skipping metrics.")
        return

    gt_parsed   = [_parse_multilabel(e["answer"])    for e in valid]
    pred_parsed = [_parse_multilabel(e[model_key])   for e in valid]

    mlb = MultiLabelBinarizer()
    mlb.fit(gt_parsed + pred_parsed)
    gt_bin   = mlb.transform(gt_parsed)
    pred_bin = mlb.transform(pred_parsed)

    exact_match = float(accuracy_score(gt_bin, pred_bin))
    wf1         = float(f1_score(gt_bin, pred_bin, average="weighted", zero_division=0))
    dataset_name = valid[0].get("dataset", "unknown")

    print(f"\n{'='*54}")
    print(f"  Dataset         : {dataset_name}  [multilabel]")
    print(f"  Model           : {model_name}")
    print(f"  N samples       : {len(valid)}")
    print(f"  Exact-match Acc : {exact_match:.4f}")
    print(f"  WF1 (weighted)  : {wf1:.4f}")
    print(f"{'='*54}\n")

    metrics = {"exact_match_accuracy": exact_match, "weighted_f1": wf1, "n_samples": len(valid)}
    metrics_path = re.sub(r"\.jsonl$", "_metrics.json", output_jsonl)
    with open(metrics_path, "w") as f:
        json.dump({"model": model_name, "dataset": dataset_name, **metrics}, f, indent=2)
    print(f"Metrics → {metrics_path}")
    _log_wandb(args, dataset_name, metrics)


# ── Merge-shards mode ─────────────────────────────────────────────────────────

def merge_shards(shard_pattern: str, output_jsonl: str, model_name: str, args=None):
    paths = sorted(glob.glob(shard_pattern))
    if not paths:
        raise FileNotFoundError(f"No files match: {shard_pattern}")

    all_entries = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            all_entries.extend(json.loads(ln) for ln in f if ln.strip())

    all_entries.sort(key=lambda e: e.get("_orig_idx", 0))

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Merged {len(all_entries)} entries from {len(paths)} shards → {output_jsonl}")
    model_key = f"predicted_answer_{model_name.replace('/', '_')}"
    multilabel = getattr(args, "multilabel", False)
    if multilabel:
        compute_and_print_metrics_multilabel(all_entries, model_key, model_name, output_jsonl, args)
    else:
        compute_and_print_metrics(all_entries, model_key, model_name, output_jsonl, args)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Merge-only mode ───────────────────────────────────────────────────────
    if args.merge_shards:
        merge_shards(args.merge_shards, args.output_jsonl, args.model or "unknown", args)
        return

    base_dir = os.path.abspath(
        args.data_base_dir if args.data_base_dir else os.path.dirname(args.input_jsonl)
    )
    thinking      = not args.no_thinking
    data_loading  = args.data_loading

    if args.temperature:
        sampling_kwargs: dict = {"do_sample": True, "temperature": args.temperature}
        if args.top_p  is not None: sampling_kwargs["top_p"]  = args.top_p
        if args.top_k  is not None: sampling_kwargs["top_k"]  = args.top_k
        if args.min_p  is not None and args.min_p > 0: sampling_kwargs["min_p"] = args.min_p
    else:
        sampling_kwargs = {"do_sample": False}

    # ── Load JSONL ────────────────────────────────────────────────────────────
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        all_entries = [json.loads(ln) for ln in f if ln.strip()]

    if args.max_samples:
        all_entries = all_entries[: args.max_samples]

    # Stamp original index for correct merge ordering
    for i, e in enumerate(all_entries):
        e.setdefault("_orig_idx", i)

    # ── Sharding ──────────────────────────────────────────────────────────────
    if args.num_shards > 1:
        shard_size = (len(all_entries) + args.num_shards - 1) // args.num_shards
        lo = args.shard_idx * shard_size
        hi = min(lo + shard_size, len(all_entries))
        entries = all_entries[lo:hi]
        print(f"Shard {args.shard_idx}/{args.num_shards}: "
              f"entries {lo}–{hi-1} ({len(entries)} samples)")
    else:
        entries = all_entries

    # ── Resume support ────────────────────────────────────────────────────────
    model_key    = f"predicted_answer_{args.model.replace('/', '_')}"
    response_key = f"model_response_{args.model.replace('/', '_')}"

    pending_idx = [i for i, e in enumerate(entries) if model_key not in e]
    if len(pending_idx) < len(entries):
        print(f"Resuming: {len(entries) - len(pending_idx)} done, "
              f"{len(pending_idx)} remaining.")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(args.model, torch_compile=args.torch_compile)

    print(f"Data loading mode: {data_loading}")

    # ── Batch inference loop ──────────────────────────────────────────────────
    save_every = args.save_every
    results    = list(entries)

    def flush():
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as wf:
            for e in results:
                wf.write(json.dumps(e, ensure_ascii=False) + "\n")

    batch_size = args.batch_size
    pending    = [i for i in pending_idx]  # indices into results[]

    with tqdm(total=len(pending), desc=f"Inference (bs={batch_size})") as pbar:
        for batch_start in range(0, len(pending), batch_size):
            batch_indices = pending[batch_start: batch_start + batch_size]
            batch_entries = [results[i] for i in batch_indices]

            try:
                responses = run_batch(
                    model, processor, batch_entries, base_dir, thinking,
                    args.max_new_tokens, data_loading, sampling_kwargs,
                    args.model, args.num_frames,
                    args.gemma_legacy_thinking,
                )
            except Exception as exc:
                print(f"\n[ERROR] Batch {batch_start}–{batch_start+len(batch_indices)-1}: {exc}")
                traceback.print_exc()
                responses = [""] * len(batch_entries)

            for idx, resp in zip(batch_indices, responses):
                results[idx][response_key] = resp
                results[idx][model_key]    = extract_answer(resp)

            pbar.update(len(batch_indices))

            # Periodic checkpoint
            if save_every > 0 and (batch_start // batch_size + 1) % save_every == 0:
                flush()

    # ── Final save + metrics ──────────────────────────────────────────────────
    flush()
    print(f"\nSaved → {args.output_jsonl}")
    if args.multilabel:
        compute_and_print_metrics_multilabel(results, model_key, args.model, args.output_jsonl, args)
    else:
        compute_and_print_metrics(results, model_key, args.model, args.output_jsonl, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot multimodal inference")

    # Core
    parser.add_argument("--model", default=None,
                        help="HF model name or local path (required unless --merge_shards)")
    parser.add_argument("--input_jsonl",  default=None, help="Input JSONL (prepare_*.py output)")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    parser.add_argument(
        "--data_base_dir",
        default=None,
        help="Base dir for resolving relative media paths (default: directory of --input_jsonl)",
    )

    # Speed
    parser.add_argument("--batch_size",    type=int,  default=1,
                        help="Samples per forward pass (increase for image-only datasets)")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Apply torch.compile (Torch >=2.0; slow first batch)")

    # Sharding (multi-GPU data parallelism)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel shards (= number of GPUs)")
    parser.add_argument("--shard_idx",  type=int, default=0,
                        help="Index of this shard (0-indexed)")

    # Merge mode
    parser.add_argument("--merge_shards", default=None,
                        help="Glob pattern of shard JSONLs to merge (e.g. 'out/*_shard*.jsonl'). "
                             "Skips inference; just merges and computes metrics.")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature; omit or 0 for greedy decoding")
    parser.add_argument("--top_p",       type=float, default=None)
    parser.add_argument("--top_k",       type=int,   default=None)
    parser.add_argument("--min_p",       type=float, default=None)
    parser.add_argument("--no_thinking",    action="store_true",
                        help="Use no-thinking prompt (direct answer, no <think> tags)")
    parser.add_argument("--gemma_legacy_thinking", action="store_true",
                        help="For Gemma 4: use the legacy <think></think> instruction instead of "
                             "the native <|think|> system-prompt mechanism")

    # Data loading mode
    parser.add_argument(
        "--data_loading",
        default="default",
        choices=["default", "verl_style"],
        help=(
            "'verl_style': uses qwen_vl_utils + torchaudio preprocessing — matches "
            "harpo_hier / omnisapiens training pipeline (nframes=4, pixel budget, 10s audio clip). "
            "'default': uses decord + soundfile — compatible with HumanOmniV2 and others."
        ),
    )

    # Video
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Force a fixed num_frames for video processing (overrides processor "
                             "default and the auto-reduce logic). Gemma4 defaults to 32; "
                             "reduce to 16 or 8 to cut memory on longer clips.")

    # Misc
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of entries processed (useful for smoke-testing)")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Flush output JSONL every N batches (0 = only at end)")
    parser.add_argument("--multilabel", action="store_true",
                        help="Use multilabel metrics (comma-separated labels in answer field)")

    # W&B (optional — only used in merge/metrics step)
    parser.add_argument("--wandb_project",  default=None, help="W&B project name")
    parser.add_argument("--wandb_run_id",   default=None, help="W&B run ID (for resume)")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run display name")
    parser.add_argument("--wandb_entity",   default=None, help="W&B entity (org/team)")

    args = parser.parse_args()

    if not args.merge_shards and not args.model:
        parser.error("--model is required unless --merge_shards is set")
    if not args.merge_shards and not args.input_jsonl:
        parser.error("--input_jsonl is required unless --merge_shards is set")

    main(args)
