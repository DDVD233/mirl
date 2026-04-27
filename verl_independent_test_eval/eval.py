"""
Test-set evaluation for multimodal HB datasets using a single combined JSONL.

Runs inference with the same GPU sharding and model-loading logic as
zero_shot_inference/inference.py, then computes dataset-specific unified
metrics (sentiment ACC2/3/5/7, emotion weighted-accuracy, etc.) using the
verl/sft/evaluate evaluation framework.

Usage (single GPU):
    python eval.py \\
        --model_name google/gemma-4-e4b-it \\
        --input_jsonl /path/to/test_all.jsonl \\
        --output_jsonl /tmp/out/gemma4_all.jsonl

Usage (multi-GPU, shard 0 of 8 on GPU 0):
    CUDA_VISIBLE_DEVICES=0 python eval.py \\
        --model_name google/gemma-4-e4b-it \\
        --input_jsonl /path/to/test_all.jsonl \\
        --output_jsonl /tmp/out/gemma4_shard0.jsonl \\
        --num_shards 8 --shard_idx 0 --batch_size 4

Merge shards and compute unified metrics:
    python eval.py \\
        --merge_shards /tmp/out/gemma4_shard*.jsonl \\
        --output_jsonl /tmp/out/gemma4_all_merged.jsonl \\
        --model_name google/gemma-4-e4b-it
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
import traceback

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# ── Thinking prompts ──────────────────────────────────────────────────────────

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
        try:
            arr, sr = sf.read(_resolve(p, base_dir), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
            arrays.append(arr)
        except Exception as e:
            import numpy as np
            print(f"[WARN] Failed to load audio {p}: {e}; substituting 0.5 s silence.")
            arrays.append(np.zeros(int(target_sr * 0.5), dtype="float32"))
    return arrays if arrays else None


def load_images(image_paths: list[str], base_dir: str) -> list[Image.Image]:
    return [Image.open(_resolve(p, base_dir)).convert("RGB") for p in image_paths]


def _video_num_frames_override(vframes_list: list, processor, requested: int | None = None) -> int | None:
    """Return num_frames to pass to the processor, capped at actual available frames.

    When requested is given (e.g. from --num_frames), caps it at the minimum actual
    frame count so a short video doesn't raise ValueError. When not given, falls back
    to the processor's configured num_frames (Gemma4 defaults to 32).
    """
    if not vframes_list:
        return None
    min_frames = min(len(v) for v in vframes_list)
    if requested is not None:
        return min(requested, min_frames)
    vp = getattr(processor, "video_processor", None)
    required = getattr(vp, "num_frames", None)
    if not required:
        return None
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
    return process_video(video_dict)


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
        audio_list = _default_load_audio_list(entry["audios"], base_dir) if entry.get("audios") else None

        if entry.get("images"):
            pil_images = load_images(entry["images"], base_dir)
            for img in pil_images:
                content.append({"type": "image", "image": img})

        # BUG FIX (from inference.py): was "[video]" — data uses <video> tags
        if entry.get("videos") and "<video>" in problem:
            video_frames = _default_load_video_frames(entry["videos"], base_dir)
            if video_frames:
                if is_gemma:
                    video_frames = np.stack(video_frames, axis=0)  # (T, H, W, C)
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

    Gemma4's pad_embedding is absent from the checkpoint and stays on meta.
    Zero-init is safe: it's a learnable placeholder initialised to zeros.
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
    # crashes in _convert_4d_mask_to_blocked_5d(None). Force eager so the method
    # always returns a real 4D causal mask. Patch both the outer config and the nested
    # text_config (for VLM wrappers like Gemma4ForConditionalGeneration).
    if is_gemma:
        for cfg in (model.config, getattr(model.config, "text_config", None)):
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                old = cfg._attn_implementation
                if old != "eager":
                    cfg._attn_implementation = "eager"
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
            nf = _video_num_frames_override(batch_videos, processor, num_frames)
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
        print(f"\n[WARN] Batch of {len(entries)} failed ({exc.__class__.__name__}: {exc}); "
              "retrying one-by-one.")
        return [_run_one(model, processor, e, base_dir, thinking, max_new_tokens,
                         data_loading, _skw, model_name, num_frames, gemma_legacy_thinking)
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
        nf = _video_num_frames_override([vframes], processor, num_frames)
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


# ── Evaluation framework integration ─────────────────────────────────────────

def _load_evaluate_module():
    """Dynamically load detailed_multi_task_evaluation, resolving its relative imports."""
    eval_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sft/evaluate")
    )
    pkg = "_hb_evaluate"

    def _load(module_name: str, filename: str):
        path = os.path.join(eval_dir, filename)
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load leaf helpers first so relative imports in the main module resolve correctly.
    # detailed_multi_task_evaluation.py has `from .helper_emo import ...` — Python resolves
    # `.helper_emo` against __package__ = "_hb_evaluate", finding "_hb_evaluate.helper_emo"
    # which we've already registered in sys.modules.
    _load(f"{pkg}.helper_emo",         "helper_emo.py")
    _load(f"{pkg}.helper_senti",       "helper_senti.py")
    _load(f"{pkg}.helper_senti_extra", "helper_senti_extra.py")
    return _load(f"{pkg}.detailed_multi_task_evaluation", "detailed_multi_task_evaluation.py")


def build_label_index_maps(label_map_path: str):
    """
    Returns (label_mapping, index_to_label).
      label_mapping:  {"cremad_anger": 7, ...}   dataset+label → global index (lowercase keys)
      index_to_label: {7: "anger", ...}           global index → label string
    """
    with open(label_map_path, "r") as f:
        cfg = json.load(f)
    label_mapping = {k.lower(): v for k, v in cfg["label_mapping"].items()}
    index_to_label = {}
    for domain_entries in cfg["meta"]["global_classes"].values():
        for e in domain_entries:
            index_to_label[e["index"]] = e["label"]
    return label_mapping, index_to_label


def pred_to_index(pred_str: str, dataset: str, label_mapping: dict) -> int:
    """Convert a predicted label string + dataset name to its global index.

    Returns -1 if the combination is not found in the label map (counted as
    incorrect by the evaluation framework).
    """
    if not pred_str:
        return -1
    key = f"{dataset.lower()}_{pred_str.lower().strip()}"
    if key in label_mapping:
        return label_mapping[key]
    # Try normalising underscores to spaces (some models output "no_depression")
    key_norm = f"{dataset.lower()}_{pred_str.lower().strip().replace('_', ' ')}"
    if key_norm in label_mapping:
        return label_mapping[key_norm]
    return -1


def _to_jsonable(o):
    """Recursively convert numpy scalars/arrays to plain Python types for json.dump."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    return o


def compute_and_save_metrics(
    merged_jsonl: str,
    model_name: str,
    label_map_path: str,
    metrics_out: str,
    judge_out: str,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_entity: str | None = None,
):
    """Load merged JSONL, compute unified per-dataset metrics, and save outputs."""
    with open(merged_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(ln) for ln in f if ln.strip()]

    if not entries:
        print("[WARN] No entries in merged JSONL — skipping metric computation.")
        return

    model_key    = f"predicted_answer_{model_name.replace('/', '_')}"
    response_key = f"model_response_{model_name.replace('/', '_')}"

    # Auto-detect key if the model slug doesn't exactly match (e.g. after rename)
    if model_key not in entries[0]:
        candidates = [k for k in entries[0] if k.startswith("predicted_answer_")]
        if not candidates:
            print("[WARN] No predicted_answer_* key found in JSONL — skipping metric computation.")
            return
        model_key    = candidates[0]
        response_key = model_key.replace("predicted_answer_", "model_response_", 1)
        print(f"  Auto-detected model key: {model_key}")

    label_mapping, _ = build_label_index_maps(label_map_path)

    # Read QA datasets that should not be evaluated by label matching (they
    # have no label_mapping entries, so both pred and GT map to -1, which
    # would produce artificially perfect 1.00 metrics).
    with open(label_map_path, "r") as _f:
        _cfg = json.load(_f)
    excluded_qa    = set(_cfg["meta"].get("excluded_qa_datasets", []))
    meta_config    = _cfg["meta"]

    predictions_str:   list[str] = []
    ground_truths_str: list[str] = []
    predictions_int:   list[int] = []
    ground_truths_int: list[int] = []
    datasets_list:     list[str] = []   # non-QA only — passed to evaluate_predictions
    datasets_all_list: list[str] = []   # all entries — used for the judge payload
    responses_list:    list[str] = []
    sample_ids_list:   list[str] = []

    # Per-dataset grouped predictions (needed for extra sentiment metrics).
    from collections import defaultdict as _dd
    per_dataset_preds: dict = _dd(lambda: ([], []))

    qa_datasets_seen: set = set()
    n_pred_unmapped = 0
    n_gt_unmapped   = 0

    for e in entries:
        pred_str = e.get(model_key, "")
        gt_str   = e.get("answer", "")
        dataset  = e.get("dataset", "unknown")

        predictions_str.append(pred_str)
        ground_truths_str.append(gt_str)
        responses_list.append(e.get(response_key, ""))
        sample_ids_list.append(str(e.get("sample_id", e.get("_orig_idx", ""))))
        datasets_all_list.append(dataset)

        # Skip label-based evaluation for LLM-judge QA datasets.
        if dataset in excluded_qa:
            qa_datasets_seen.add(dataset)
            continue

        p_idx = pred_to_index(pred_str, dataset, label_mapping)
        g_idx = pred_to_index(gt_str,   dataset, label_mapping)
        if p_idx == -1:
            n_pred_unmapped += 1
        if g_idx == -1:
            n_gt_unmapped += 1

        predictions_int.append(p_idx)
        ground_truths_int.append(g_idx)
        datasets_list.append(dataset)

        per_dataset_preds[dataset][0].append(p_idx)
        per_dataset_preds[dataset][1].append(g_idx)

    total = len(entries)
    print(f"  Predictions  : {total - n_pred_unmapped}/{total} mapped to label indices "
          f"({n_pred_unmapped} unmapped → -1, counted as wrong)")
    if n_gt_unmapped:
        print(f"  [WARN] Ground truths: {n_gt_unmapped} unmapped — check dataset/label names")
    if qa_datasets_seen:
        print(f"  QA datasets (excluded from label eval, placeholder=0.0): "
              f"{sorted(qa_datasets_seen)}")

    eval_mod = _load_evaluate_module()
    results = eval_mod.evaluate_predictions(
        predictions=predictions_int,
        ground_truths=ground_truths_int,
        datasets=datasets_list,
        label_map_path=label_map_path,
    )

    # ── Add 0.0 placeholders for LLM-judge QA datasets ───────────────────────
    for qa_ds in sorted(qa_datasets_seen):
        results["per_dataset_metrics"][f"{qa_ds}/llm_judge_accuracy"] = 0.0

    # ── Add extra sentiment metrics (macro + micro F1 per collapse level) ────
    senti_extra_mod = sys.modules.get("_hb_evaluate.helper_senti_extra")
    if senti_extra_mod:
        dataset_domain = meta_config.get("dataset_domain", {})
        for ds, (preds, gts) in per_dataset_preds.items():
            if dataset_domain.get(ds) == "sentiment_intensity" and preds:
                extra = senti_extra_mod.compute_sentiment_extra_metrics(
                    preds, gts, meta_config, eval_mod.compute_set_metrics
                )
                for k, v in extra.items():
                    results["per_dataset_metrics"][f"{ds}/{k}"] = v

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model   : {model_name}")
    print(f"  Samples : {total}")
    print(f"\n  Aggregate metrics:")
    for k, v in sorted(results["aggregate_metrics"].items()):
        print(f"    {k:35s}: {v:.4f}")

    per_ds = results.get("per_dataset_metrics", {})
    if per_ds:
        # Group by dataset, print top-level metrics only (no sub-class breakdown)
        from collections import defaultdict
        ds_keys: dict[str, list] = defaultdict(list)
        for k in sorted(per_ds):
            ds = k.split("/")[0]
            ds_keys[ds].append(k)
        print(f"\n  Per-dataset metrics:")
        for ds, keys in ds_keys.items():
            print(f"  [{ds}]")
            for k in keys:
                if k.count("/") == 1:  # skip per-class sub-metrics
                    print(f"    {k:55s}: {per_ds[k]:.4f}")
    print(f"{'='*60}\n")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_dir = os.path.dirname(os.path.abspath(metrics_out))
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_payload = _to_jsonable({
        "model": model_name,
        "n_samples": total,
        "aggregate_metrics": results["aggregate_metrics"],
        "per_dataset_metrics": results.get("per_dataset_metrics", {}),
    })
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    print(f"Metrics      → {metrics_out}")

    # ── Save judge-format JSON ────────────────────────────────────────────────
    # predictions = full model responses (matching hb_evaluation.py step*.json
    # format) so that extract_boxed() in hb_evaluation.py / llm_judge_eval.py
    # works correctly. extracted_predictions holds the pre-extracted answers.
    judge_dir = os.path.dirname(os.path.abspath(judge_out))
    os.makedirs(judge_dir, exist_ok=True)
    judge_payload = {
        "model": model_name,
        "predictions": responses_list,
        "ground_truths": ground_truths_str,
        "datasets": datasets_all_list,
        "extracted_predictions": predictions_str,
        "sample_ids": sample_ids_list,
    }
    with open(judge_out, "w", encoding="utf-8") as f:
        json.dump(judge_payload, f, indent=2, ensure_ascii=False)
    print(f"Judge format → {judge_out}")

    # ── Wandb logging ─────────────────────────────────────────────────────────
    if wandb_project:
        import wandb as _wandb
        _wandb.init(
            project=wandb_project,
            name=wandb_run_name or None,
            entity=wandb_entity or None,
            config={"model": model_name, "n_samples": total},
        )
        wandb_metrics: dict = {"n_samples": total}
        wandb_metrics.update(results["aggregate_metrics"])
        wandb_metrics.update(results.get("per_dataset_metrics", {}))
        _wandb.log(wandb_metrics)
        _wandb.finish()
        print(f"Wandb       → project={wandb_project} run={wandb_run_name or 'auto'}")


# ── Merge-shards mode ─────────────────────────────────────────────────────────

def merge_shards(
    shard_pattern: str,
    output_jsonl: str,
    model_name: str,
    label_map_path: str | None = None,
    metrics_out: str | None = None,
    judge_out: str | None = None,
    no_metrics: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_entity: str | None = None,
):
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

    if no_metrics:
        return
    if not label_map_path:
        print("[INFO] No --label_map_path provided — skipping unified metric computation.")
        return
    if not os.path.isfile(label_map_path):
        print(f"[WARN] Label map not found at {label_map_path} — skipping metric computation.")
        return

    stem = re.sub(r"\.jsonl$", "", output_jsonl)
    _metrics_out = metrics_out or f"{stem}_metrics.json"
    _judge_out   = judge_out   or f"{stem}_judge.json"

    compute_and_save_metrics(
        merged_jsonl=output_jsonl,
        model_name=model_name,
        label_map_path=label_map_path,
        metrics_out=_metrics_out,
        judge_out=_judge_out,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
    )


# ── Main (inference loop) ─────────────────────────────────────────────────────

def main(args):
    # ── Merge-only mode ───────────────────────────────────────────────────────
    if args.merge_shards:
        merge_shards(
            shard_pattern=args.merge_shards,
            output_jsonl=args.output_jsonl,
            model_name=args.model_name or "unknown",
            label_map_path=args.label_map_path,
            metrics_out=args.metrics_output,
            judge_out=args.judge_output,
            no_metrics=args.no_metrics,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_entity=args.wandb_entity,
        )
        return

    base_dir = os.path.abspath(
        args.data_base_dir if args.data_base_dir else os.path.dirname(args.input_jsonl)
    )
    thinking     = not args.no_thinking
    data_loading = args.data_loading

    if args.temperature:
        sampling_kwargs: dict = {"do_sample": True, "temperature": args.temperature}
        if args.top_p is not None: sampling_kwargs["top_p"] = args.top_p
        if args.top_k is not None: sampling_kwargs["top_k"] = args.top_k
        if args.min_p is not None and args.min_p > 0: sampling_kwargs["min_p"] = args.min_p
    else:
        sampling_kwargs = {"do_sample": False}

    # ── Load JSONL ────────────────────────────────────────────────────────────
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        all_entries = [json.loads(ln) for ln in f if ln.strip()]

    if args.smoke_n_per_dataset:
        import random
        from collections import defaultdict as _defaultdict
        ds_map: dict[str, list] = _defaultdict(list)
        for e in all_entries:
            ds_map[e.get("dataset", "unknown")].append(e)
        selected = []
        for ds in sorted(ds_map):
            pool = ds_map[ds]
            selected.extend(random.sample(pool, min(args.smoke_n_per_dataset, len(pool))))
        all_entries = selected
        print(f"Smoke test  : {args.smoke_n_per_dataset} sample(s)/dataset × "
              f"{len(ds_map)} datasets = {len(all_entries)} total entries")
    elif args.max_samples:
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
    model_key    = f"predicted_answer_{args.model_name.replace('/', '_')}"
    response_key = f"model_response_{args.model_name.replace('/', '_')}"

    _done: dict[int, dict] = {}

    resume_dir = getattr(args, "resume_dir", None)
    if resume_dir and os.path.isdir(resume_dir):
        # Scan every .jsonl in the directory; an entry counts as done only if
        # the raw model response is non-empty (empty = error-handler fallback).
        shard_files = sorted(glob.glob(os.path.join(resume_dir, "*.jsonl")))
        for _p in shard_files:
            with open(_p, "r", encoding="utf-8") as _rf:
                for _ln in _rf:
                    if not _ln.strip():
                        continue
                    _e = json.loads(_ln)
                    if "_orig_idx" in _e and _e.get(response_key, ""):
                        _done[_e["_orig_idx"]] = _e
        print(f"Resume dir  : {len(_done)} valid prediction(s) across "
              f"{len(shard_files)} file(s) in {resume_dir}")
    elif os.path.isfile(args.output_jsonl):
        # No resume_dir given — auto-resume within this run's own output file.
        with open(args.output_jsonl, "r", encoding="utf-8") as _rf:
            for _ln in _rf:
                if not _ln.strip():
                    continue
                _e = json.loads(_ln)
                if "_orig_idx" in _e and _e.get(response_key, ""):
                    _done[_e["_orig_idx"]] = _e
        if _done:
            print(f"Auto-resume : {len(_done)} valid prediction(s) from {args.output_jsonl}")

    for e in entries:
        if e.get("_orig_idx") in _done:
            e[model_key]    = _done[e["_orig_idx"]][model_key]
            e[response_key] = _done[e["_orig_idx"]].get(response_key, "")

    pending_idx = [i for i, e in enumerate(entries) if model_key not in e]
    if len(pending_idx) < len(entries):
        print(f"Resuming    : {len(entries) - len(pending_idx)} done, "
              f"{len(pending_idx)} remaining.")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(args.model_name, torch_compile=args.torch_compile)
    print(f"Data loading mode: {data_loading}")

    # ── Batch inference loop ──────────────────────────────────────────────────
    save_every = args.save_every
    results    = list(entries)

    def flush():
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as wf:
            for e in results:
                wf.write(json.dumps(e, ensure_ascii=False) + "\n")

    batch_size   = args.batch_size
    pending      = list(pending_idx)
    shard_offset = lo if args.num_shards > 1 else 0
    already_done = len(entries) - len(pending)

    with tqdm(
        total=len(all_entries),
        initial=shard_offset + already_done,
        desc=f"Inference shard={args.shard_idx}/{args.num_shards} (bs={batch_size})",
    ) as pbar:
        for batch_start in range(0, len(pending), batch_size):
            batch_indices = pending[batch_start: batch_start + batch_size]
            batch_entries = [results[i] for i in batch_indices]

            try:
                responses = run_batch(
                    model, processor, batch_entries, base_dir, thinking,
                    args.max_new_tokens, data_loading, sampling_kwargs,
                    args.model_name, args.num_frames,
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

            # Sidecar file for the shell-level combined progress bar across all shards.
            # Format: "{done}\n{total}\n" — shell monitor aggregates all shards.
            try:
                with open(args.output_jsonl + ".progress", "w") as _pf:
                    _pf.write(f"{batch_start + len(batch_indices)}\n{len(pending)}\n")
            except OSError:
                pass

            if save_every > 0 and (batch_start // batch_size + 1) % save_every == 0:
                flush()

    flush()
    print(f"\nSaved → {args.output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test-set evaluation for HB multimodal datasets (unified metrics)"
    )

    # Core
    parser.add_argument("--model_name", default=None,
                        help="HF model name or local path (required unless --merge_shards)")
    parser.add_argument("--input_jsonl", default=None,
                        help="Single combined test JSONL with entries from all datasets")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--data_base_dir", default=None,
                        help="Base dir for resolving relative media paths "
                             "(default: directory of --input_jsonl)")

    # Label map & evaluation
    _default_map = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../sft/label_maps/unified_label_map.json")
    )
    parser.add_argument("--label_map_path", default=_default_map,
                        help="Path to unified_label_map.json "
                             "(default: ../sft/label_maps/unified_label_map.json)")
    parser.add_argument("--metrics_output", default=None,
                        help="Where to write metrics JSON "
                             "(default: <output_jsonl stem>_metrics.json)")
    parser.add_argument("--judge_output", default=None,
                        help="Where to write judge-format JSON "
                             "(default: <output_jsonl stem>_judge.json)")
    parser.add_argument("--no_metrics", action="store_true",
                        help="Skip unified metric computation after merge")

    # Speed
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Samples per forward pass")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Apply torch.compile (Torch >=2.0; slow first batch)")

    # Sharding (multi-GPU data parallelism)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel shards")
    parser.add_argument("--shard_idx",  type=int, default=0,
                        help="Index of this shard (0-indexed)")

    # Merge mode
    parser.add_argument("--merge_shards", default=None,
                        help="Glob pattern of shard JSONLs to merge. Skips inference.")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature; omit or 0 for greedy decoding")
    parser.add_argument("--top_p",  type=float, default=None)
    parser.add_argument("--top_k",  type=int,   default=None)
    parser.add_argument("--min_p",  type=float, default=None)
    parser.add_argument("--no_thinking", action="store_true",
                        help="Use no-thinking prompt (direct answer, no <think> tags)")
    parser.add_argument("--gemma_legacy_thinking", action="store_true",
                        help="For Gemma 4: use the legacy <think></think> instruction instead of "
                             "the native <|think|> system-prompt mechanism")

    # Data loading mode
    parser.add_argument("--data_loading", default="default",
                        choices=["default", "verl_style"],
                        help=(
                            "'verl_style': qwen_vl_utils + torchaudio — matches harpo/omnisapiens. "
                            "'default': decord + soundfile — compatible with HumanOmniV2 and Gemma."
                        ))

    # Video
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Force a fixed num_frames for video processing. "
                             "Gemma4 defaults to 32; reduce to cut memory on longer clips.")

    # Resume
    parser.add_argument("--resume_dir", default=None,
                        help="Directory containing prior shard output JSONLs. "
                             "The script scans all *.jsonl files there, collects every entry "
                             "that has a valid (non-empty) model response, and skips those on "
                             "this run. Use this to resume after a partial or crashed multi-shard "
                             "run without having to specify shard indices.")

    # Misc
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of entries processed (smoke-testing)")
    parser.add_argument("--smoke_n_per_dataset", type=int, default=None,
                        help="Take N random entries per dataset (covers all datasets; "
                             "takes priority over --max_samples)")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Flush output JSONL every N batches (0 = only at end)")

    # Wandb
    parser.add_argument("--wandb_project",  default=None,
                        help="W&B project name (enables wandb logging when set)")
    parser.add_argument("--wandb_run_name", default=None,
                        help="W&B run name (optional; auto-generated if omitted)")
    parser.add_argument("--wandb_entity",   default=None,
                        help="W&B entity / team (optional)")

    args = parser.parse_args()

    if not args.merge_shards and not args.model_name:
        parser.error("--model_name is required unless --merge_shards is set")
    if not args.merge_shards and not args.input_jsonl:
        parser.error("--input_jsonl is required unless --merge_shards is set")

    main(args)
