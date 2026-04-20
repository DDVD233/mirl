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
import traceback

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
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


def load_audio_concat(audio_paths: list[str], base_dir: str, target_sr: int = 16000):
    """Load each audio file, resample to target_sr, concatenate with 0.5 s silence."""
    if not audio_paths:
        return None
    try:
        import soundfile as sf
        import librosa
    except ImportError:
        raise ImportError("pip install soundfile librosa")

    clips = []
    for p in audio_paths:
        arr, sr = sf.read(_resolve(p, base_dir), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        clips.append(arr)

    if len(clips) == 1:
        return clips[0], target_sr

    silence = np.zeros(int(0.5 * target_sr), dtype=np.float32)
    merged = clips[0]
    for clip in clips[1:]:
        merged = np.concatenate([merged, silence, clip])
    return merged, target_sr


def load_images(image_paths: list[str], base_dir: str) -> list[Image.Image]:
    return [Image.open(_resolve(p, base_dir)).convert("RGB") for p in image_paths]


def load_video_frames(video_paths: list[str], base_dir: str,
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


# ── Entry → content list + flat media collectors ─────────────────────────────

def build_entry_inputs(entry: dict, base_dir: str, thinking: bool):
    """
    Returns (content_list, audio_tuple_or_None, pil_images, video_frames).
    content_list is the list of dicts for the chat message.
    """
    content = []
    audio_tuple = None
    pil_images = []
    video_frames = []

    if entry.get("audios"):
        result = load_audio_concat(entry["audios"], base_dir)
        if result is not None:
            audio_tuple = result
            # One <audio> tag per source clip (the text template has that many placeholders)
            for _ in entry["audios"]:
                content.append({"type": "audio", "audio": result[0],
                                 "sampling_rate": result[1]})

    if entry.get("images"):
        pil_images = load_images(entry["images"], base_dir)
        for img in pil_images:
            content.append({"type": "image", "image": img})

    if entry.get("videos"):
        video_frames = load_video_frames(entry["videos"], base_dir)
        if video_frames:
            content.append({"type": "video", "video": video_frames})

    instruction = THINKING_INSTRUCTION if thinking else NO_THINKING_INSTRUCTION
    content.append({"type": "text", "text": entry["problem"] + instruction})

    return content, audio_tuple, pil_images, video_frames


# ── Model loading ─────────────────────────────────────────────────────────────

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

    # Try AutoModelForCausalLM first; fall back to Qwen2_5OmniThinkerForConditionalGeneration
    # if the model type is not registered in the installed transformers version.
    try:
        model = _try_attn_impls(AutoModelForCausalLM)
    except ValueError as exc:
        if "does not recognize this architecture" not in str(exc) and "model type" not in str(exc):
            raise
        try:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration as OmniCls
        except ImportError:
            raise RuntimeError(
                "transformers does not recognise 'qwen2_5_omni_thinker'. "
                "Run: pip install --upgrade transformers"
            ) from exc
        print("  Falling back to Qwen2_5OmniThinkerForConditionalGeneration")
        model = _try_attn_impls(OmniCls)

    model.eval()

    if torch_compile:
        print("  Applying torch.compile (first batch will be slow — this is expected)...")
        model = torch.compile(model, mode="reduce-overhead")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    visible = set(p.device for p in model.parameters())
    print(f"  Devices: {visible}")
    return model, processor


# ── Batch inference ───────────────────────────────────────────────────────────

def _get_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_batch(model, processor, entries: list[dict], base_dir: str,
              thinking: bool, max_new_tokens: int) -> list[str]:
    """
    Run inference on a list of entries as a single batched forward pass.
    All entries should share the same modality_signature for reliable batching.
    Falls back to one-at-a-time on any processor error.
    """
    if len(entries) == 1:
        return [_run_one(model, processor, entries[0], base_dir, thinking, max_new_tokens)]

    try:
        texts, batch_audios, batch_images, batch_videos = [], [], [], []

        for entry in entries:
            content, audio_t, imgs, vframes = build_entry_inputs(entry, base_dir, thinking)
            msgs = [{"role": "user", "content": content}]
            texts.append(
                processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
            if audio_t is not None:
                batch_audios.append(audio_t)
            batch_images.extend(imgs)
            if vframes:
                batch_videos.append(vframes)

        proc_kwargs = dict(text=texts, return_tensors="pt", padding=True)
        if batch_audios:
            proc_kwargs["audios"] = batch_audios
        if batch_images:
            proc_kwargs["images"] = batch_images
        if batch_videos:
            proc_kwargs["videos"] = batch_videos

        device = _get_device(model)
        inputs = processor(**proc_kwargs)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Strip prompt tokens — with left-padding all prompts end at the same column
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_len:]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)

    except Exception as exc:
        # Batching failed (e.g. mixed modalities or processor limitation); fall back
        print(f"\n[WARN] Batch of {len(entries)} failed ({exc.__class__.__name__}: {exc}); "
              "retrying one-by-one.")
        return [_run_one(model, processor, e, base_dir, thinking, max_new_tokens)
                for e in entries]


def _run_one(model, processor, entry: dict, base_dir: str,
             thinking: bool, max_new_tokens: int) -> str:
    content, audio_t, imgs, vframes = build_entry_inputs(entry, base_dir, thinking)
    msgs = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    proc_kwargs = dict(text=text, return_tensors="pt", padding=True)
    if audio_t is not None:
        proc_kwargs["audios"] = [audio_t]
    if imgs:
        proc_kwargs["images"] = imgs
    if vframes:
        proc_kwargs["videos"] = [vframes]

    device = _get_device(model)
    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(out_ids[0][prompt_len:], skip_special_tokens=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_and_print_metrics(results: list[dict], model_key: str, model_name: str,
                              output_jsonl: str):
    preds = [e[model_key] for e in results if e.get(model_key)]
    gts   = [e["answer"]  for e in results if e.get(model_key)]
    if not preds:
        print("No valid predictions — skipping metrics.")
        return

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

    metrics_path = re.sub(r"\.jsonl$", "_metrics.json", output_jsonl)
    with open(metrics_path, "w") as f:
        json.dump({"model": model_name, "dataset": dataset_name,
                   "n_samples": len(preds), "accuracy": acc, "weighted_f1": wf1}, f, indent=2)
    print(f"Metrics → {metrics_path}")


# ── Merge-shards mode ─────────────────────────────────────────────────────────

def merge_shards(shard_pattern: str, output_jsonl: str, model_name: str):
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
    compute_and_print_metrics(all_entries, model_key, model_name, output_jsonl)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── Merge-only mode ───────────────────────────────────────────────────────
    if args.merge_shards:
        merge_shards(args.merge_shards, args.output_jsonl, args.model or "unknown")
        return

    base_dir = os.path.abspath(args.data_base_dir)
    thinking  = not args.no_thinking

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
                    model, processor, batch_entries, base_dir, thinking, args.max_new_tokens
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
    compute_and_print_metrics(results, model_key, args.model, args.output_jsonl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot multimodal inference")

    # Core
    parser.add_argument("--model", default=None,
                        help="HF model name or local path (required unless --merge_shards)")
    parser.add_argument("--input_jsonl",  default=None, help="Input JSONL (prepare_*.py output)")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    parser.add_argument(
        "--data_base_dir",
        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        help="Base dir for resolving relative media paths (default: repo root)",
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
    parser.add_argument("--no_thinking",    action="store_true",
                        help="Use no-thinking prompt (direct answer, no <think> tags)")

    # Misc
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of entries processed (useful for smoke-testing)")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Flush output JSONL every N batches (0 = only at end)")

    args = parser.parse_args()

    if not args.merge_shards and not args.model:
        parser.error("--model is required unless --merge_shards is set")
    if not args.merge_shards and not args.input_jsonl:
        parser.error("--input_jsonl is required unless --merge_shards is set")

    main(args)
