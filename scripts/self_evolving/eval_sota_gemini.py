"""Evaluate a SoTA model (Gemini) on the MIMIC-IV rare-disease test set.

Uses the EXACT same scoring pipeline as training/val
(verl.utils.reward_score.self_evolving.compute_score) so metrics are
directly comparable to wandb's val-core / val-aux numbers from training.

Pipeline per test entry:
    1. Send the system + user prompt (with chest X-ray + ECG images) to
       Gemini.
    2. Pass the raw response through `compute_score` which produces:
         - acc                 exact / normalized match (0/1)
         - answer_quality      Qwen-judge 1-5
         - reasoning_quality   Qwen-judge 1-5
         - biobert_sim         cosine [0,1] via biobert server
         - char_bleu           NLTK char-level BLEU-4
         - format_ok           \\boxed{} present
         - score               composite weighted reward
    3. Aggregate (mean, max, min, p50) per metric.

The same Qwen judge endpoint (api_base) and BioBERT server
(biobert_api_base) used during training MUST be reachable so the LLM-
judge and embedding components match exactly.

Usage:
    GEMINI_API_KEY=... \
    API_BASE=http://node2500:8002/v1 \
    MODEL_NAME=Qwen/Qwen3.6-27B \
    BIOBERT_API_BASE=http://localhost:8003 \
    python scripts/self_evolving/eval_sota_gemini.py \
        --val_file $HOME/scratch/dvdai/self_evolving_datasets/mimiciv_rare/test.jsonl \
        --model_name gemini-3.1-pro-preview \
        --concurrency 8 \
        --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import aiohttp


def _load_compute_score():
    """Load compute_score directly from the source file, bypassing
    verl/__init__.py (which drags in ray, tensordict, omegaconf, etc.).
    The reward function only needs aiohttp + nltk and we don't want eval
    machines to install the entire training stack just to score outputs.
    """
    import importlib.util
    here = Path(__file__).resolve()
    target = None
    for parent in here.parents:
        candidate = parent / "verl" / "utils" / "reward_score" / "self_evolving.py"
        if candidate.exists():
            target = candidate
            break
    if target is None:
        raise RuntimeError("could not locate verl/utils/reward_score/self_evolving.py")
    spec = importlib.util.spec_from_file_location("self_evolving_reward", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_score


compute_score = _load_compute_score()


GEMINI_DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _read_image_b64(
    path: str,
    max_pixels: int = 256 * 256,
    max_bytes: int = 6 * 1024 * 1024,
) -> tuple[str, str] | None:
    """Return (mime_type, base64_data) for the given path, or None if unreadable.

    Resizes the image so total pixels <= max_pixels (aspect ratio preserved)
    to mirror what the training pipeline gives the trained Qwen actor via
    qwen_vl_utils.fetch_image with max_pixels=65536 (256x256). This keeps
    Gemini and the trained actor on parity for visual input bandwidth.
    """
    try:
        from io import BytesIO
        from PIL import Image as _PILImage

        with _PILImage.open(path) as im:
            im.load()
            im = im.convert("RGB")
            w, h = im.size
            total = w * h
            if total > max_pixels:
                import math
                scale = math.sqrt(max_pixels / total)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                im = im.resize((new_w, new_h), _PILImage.BILINEAR)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return "image/jpeg", base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _truncate_middle(text: str, max_chars: int) -> str:
    """If text exceeds max_chars, drop the middle and insert a marker.

    Keeps the head + tail (which usually carry demographics + final
    question) and ellides the labs/charts in the middle if necessary.
    The training preprocessor already caps the user content to
    max_text_chars=9000, so this is a defensive safety net for any
    entries that slipped past or any new evals run on uncapped data.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[... middle truncated to fit context ...]\n\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars]
    head_keep = keep // 2
    tail_keep = keep - head_keep
    return text[:head_keep] + marker + text[-tail_keep:]


def _build_gemini_request(
    entry: dict,
    model_name: str,
    max_pixels: int = 256 * 256,
    max_text_chars: int = 9000,
) -> dict:
    """Convert a verl-format prompt entry into a Gemini API request body.

    Gemini's REST API expects a `contents` list of `parts`, where text and
    image parts are interleaved. We follow the order of `<image>` placeholders
    in the user content so the model sees images in the same positions as the
    trained Qwen actor would.

    To match training input parity:
    - images are resized so total pixels <= max_pixels (default 65536 ≈
      256x256 — same cap as preprocess_mimiciv_rare.py:max_pixels_per_image);
    - the user text is middle-truncated to <= max_text_chars (default 9000
      — same as preprocessing). test.jsonl is already capped, so this is
      a defensive safety net.
    """
    prompt = entry.get("prompt", [])
    images = entry.get("images", []) or []
    image_paths: list[str] = []
    for img in images:
        if isinstance(img, dict):
            p = img.get("image") or img.get("url")
        else:
            p = img
        if p:
            image_paths.append(p)

    sys_text = ""
    user_text = ""
    for msg in prompt:
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        if role == "system":
            sys_text = content
        elif role == "user":
            user_text = content

    user_text = _truncate_middle(user_text, max_text_chars)

    parts: list[dict] = []
    img_iter = iter(image_paths)
    pieces = re.split(r"<image>", user_text)
    for i, piece in enumerate(pieces):
        if piece:
            parts.append({"text": piece})
        if i < len(pieces) - 1:
            try:
                ipath = next(img_iter)
            except StopIteration:
                continue
            blob = _read_image_b64(ipath, max_pixels=max_pixels)
            if blob is not None:
                mime, b64 = blob
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    # Any remaining unmatched image paths get appended after the text.
    for ipath in img_iter:
        blob = _read_image_b64(ipath, max_pixels=max_pixels)
        if blob is not None:
            mime, b64 = blob
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    body: dict = {
        "contents": [{"role": "user", "parts": parts}],
        # Gemini 2.5+ / 3.x preview models are "thinking" models: internal
        # reasoning tokens count against maxOutputTokens. We need a generous
        # budget so the visible response (the part we score) isn't truncated
        # while the model is still thinking. includeThoughts=False keeps the
        # raw chain-of-thought out of the returned text — we score only the
        # final answer, matching how the actor's <think>...</think> is treated
        # by extract_boxed_answer in compute_score.
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 16384,
            "thinkingConfig": {"includeThoughts": False},
        },
    }
    if sys_text:
        body["systemInstruction"] = {"parts": [{"text": sys_text}]}
    return body


async def _call_gemini(session: aiohttp.ClientSession, model_name: str, api_key: str, body: dict) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    timeout = aiohttp.ClientTimeout(total=180)
    async with session.post(url, json=body, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"Gemini HTTP {resp.status}: {text[:500]}")
        data = json.loads(text)
    cands = data.get("candidates") or []
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", []) or []
    return "".join(p.get("text", "") for p in parts)


async def _score_one(
    entry: dict,
    api_base: str,
    api_key: str,
    judge_model: str,
    biobert_api_base: str,
) -> dict:
    """Run compute_score on the entry's response, mirroring the reward path."""
    extra_info = entry.get("extra_info", {}) or {}
    extra_info = dict(extra_info)
    extra_info.setdefault("question", entry.get("prompt", [{}, {}])[1].get("content", "")[:2000])

    return await compute_score(
        data_source=entry.get("data_source", "mimiciv_rare_dx"),
        solution_str=entry["__response__"],
        ground_truth=entry["reward_model"]["ground_truth"],
        extra_info=extra_info,
        api_base=api_base,
        api_key=api_key,
        model_name=judge_model,
        biobert_api_base=biobert_api_base,
    )


async def _eval_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    entry: dict,
    args,
) -> dict:
    """Generate with Gemini, then score with the verl pipeline."""
    async with sem:
        body = _build_gemini_request(
            entry,
            args.model_name,
            max_pixels=args.max_pixels,
            max_text_chars=args.max_text_chars,
        )
        try:
            response = await _call_gemini(session, args.model_name, args.gemini_api_key, body)
        except Exception as e:
            response = ""
            err = str(e)
        else:
            err = None

    entry["__response__"] = response
    entry["__error__"] = err

    score = await _score_one(
        entry,
        api_base=args.api_base,
        api_key=args.judge_api_key,
        judge_model=args.judge_model_name,
        biobert_api_base=args.biobert_api_base,
    )
    score = dict(score)
    score["error"] = err is not None
    score["response_chars"] = len(response)
    score["response"] = response
    score["error_msg"] = err or ""
    score["ground_truth"] = entry["reward_model"]["ground_truth"]
    score["hadm_id"] = entry.get("extra_info", {}).get("hadm_id")
    return score


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "n": 0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "max": float(max(values)),
        "min": float(min(values)),
        "n": len(values),
    }


async def _main_async(args) -> int:
    with open(args.val_file) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]
    print(f"Loaded {len(entries)} entries from {args.val_file}")
    print(f"Gemini model: {args.model_name}, judge: {args.judge_model_name} @ {args.api_base}")
    print(f"BioBERT: {args.biobert_api_base or '(disabled)'}")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: read previously completed entries (keyed by hadm_id) and seed the
    # aggregator with them so the final summary covers the full set, then skip
    # those hadm_ids when issuing new Gemini calls. The last line of the JSONL
    # may be a partial write from a SIGINT mid-flush — caught by the JSONDecodeError
    # handler and silently skipped.
    aggregated: dict[str, list[float]] = defaultdict(list)
    done_ids: set = set()
    resumed = 0
    if output_path.exists():
        with output_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                hid = rec.get("hadm_id")
                if hid is None:
                    continue
                done_ids.add(hid)
                resumed += 1
                if rec.get("error"):
                    pass  # still counted in done_ids; skip on retry
                for k in ("acc", "judge_acc_lenient", "judge_acc_strict",
                          "answer_quality", "reasoning_quality",
                          "biobert_sim", "char_bleu", "format_ok", "score"):
                    if k in rec and isinstance(rec[k], (int, float)):
                        aggregated[k].append(float(rec[k]))
        if resumed:
            print(f"Resuming from {output_path}: {resumed} entries already done")

    pending_entries = [
        e for e in entries
        if e.get("extra_info", {}).get("hadm_id") not in done_ids
    ]
    skipped = len(entries) - len(pending_entries)
    if skipped:
        print(f"Skipping {skipped} entries already in {output_path.name}")

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)

    error_count = 0  # NEW errors this run; existing errors already in the file
    started = time.time()
    completed = 0
    total = len(pending_entries)

    out_fp = output_path.open("a")
    flush_every = max(1, int(getattr(args, "flush_every", 10)))
    log_every = flush_every

    if total == 0:
        print("Nothing to do — all entries already completed.")
        out_fp.close()
    else:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [asyncio.create_task(_eval_one(sem, session, e, args)) for e in pending_entries]
            for fut in asyncio.as_completed(tasks):
                score = await fut
                completed += 1
                if score.get("error"):
                    error_count += 1
                for k in ("acc", "judge_acc_lenient", "judge_acc_strict",
                          "answer_quality", "reasoning_quality",
                          "biobert_sim", "char_bleu", "format_ok", "score"):
                    if k in score and isinstance(score[k], (int, float)):
                        aggregated[k].append(float(score[k]))
                out_fp.write(json.dumps(score) + "\n")
                if completed % flush_every == 0:
                    out_fp.flush()
                    os.fsync(out_fp.fileno())
                if completed % log_every == 0 or completed == total:
                    elapsed = time.time() - started
                    rate = completed / max(elapsed, 1e-9)
                    eta_min = (total - completed) / max(rate, 1e-9) / 60

                    def _mean(k: str) -> float:
                        v = aggregated.get(k, [])
                        return (sum(v) / len(v)) if v else 0.0

                    print(
                        f"  [{completed}/{total}]  new_err={error_count}  "
                        f"acc={_mean('acc'):.4f}  "
                        f"jL={_mean('judge_acc_lenient'):.4f}  "
                        f"jS={_mean('judge_acc_strict'):.4f}  "
                        f"qual={_mean('answer_quality'):.3f}  "
                        f"bleu={_mean('char_bleu'):.3f}  "
                        f"bio={_mean('biobert_sim'):.3f}  "
                        f"rate={rate:.2f}/s  eta={eta_min:.1f} min"
                    )

        out_fp.flush()
        os.fsync(out_fp.fileno())
        out_fp.close()

    print()
    print("=" * 70)
    print(f"Gemini evaluation summary  (model={args.model_name})")
    print(f"  total: {completed}, errors: {error_count}")
    print("=" * 70)
    for key in ("score", "acc", "judge_acc_lenient", "judge_acc_strict",
                "answer_quality", "reasoning_quality",
                "biobert_sim", "char_bleu", "format_ok"):
        a = _agg(aggregated.get(key, []))
        print(
            f"  {key:>20s}  mean={a['mean']:.4f}  median={a['median']:.4f}  "
            f"min={a['min']:.4f}  max={a['max']:.4f}  n={a['n']}"
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val_file",
        default="/home/dvdai/scratch/dvdai/self_evolving_datasets/mimiciv_rare/test.jsonl",
    )
    parser.add_argument("--model_name", default=GEMINI_DEFAULT_MODEL)
    parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--judge_model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-27B"))
    parser.add_argument("--api_base", default=os.environ.get("API_BASE", "http://node2500:8002/v1"))
    parser.add_argument("--judge_api_key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--biobert_api_base", default=os.environ.get("BIOBERT_API_BASE", "http://localhost:8003"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all entries")
    parser.add_argument(
        "--output_jsonl",
        default="",
        help="Path to per-sample JSONL. Defaults to ./eval_gemini_<model>.jsonl. "
             "If the file exists, completed entries (matched by hadm_id) are "
             "skipped and we append new ones — so the run is resumable across "
             "interruptions. Pick a stable filename per run; do NOT timestamp "
             "it if you want resume to work.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=10,
        help="fsync the output JSONL every N completed entries (default 10). "
             "Smaller = more durable on crash, more I/O.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=256 * 256,
        help="Per-image max pixel count (aspect preserved). Defaults to 65536, "
             "matching preprocess_mimiciv_rare.py's max_pixels_per_image so "
             "Gemini and the trained actor see the same visual input bandwidth.",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=9000,
        help="Char cap on user content. Test entries are already truncated to "
             "this in preprocessing; the runtime middle-truncate here is a "
             "defensive safety net.",
    )
    args = parser.parse_args()

    if not args.gemini_api_key:
        sys.exit("GEMINI_API_KEY (or --gemini_api_key) is required")
    if not args.api_base:
        print("WARNING: api_base is empty — Qwen judge metrics will fall back to defaults")
    if not args.biobert_api_base:
        print("WARNING: biobert_api_base is empty — biobert_sim will be 0.0 for every sample")

    if not args.output_jsonl:
        # Stable default — no timestamp — so re-running resumes the prior run
        # for the same model. Override with --output_jsonl if you want a fresh
        # file or to keep multiple runs separate.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_name).strip("_")
        args.output_jsonl = f"eval_gemini_{safe}.jsonl"

    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
