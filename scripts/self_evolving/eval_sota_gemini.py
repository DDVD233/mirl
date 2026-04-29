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

The same Qwen3-VL judge endpoint (api_base) and BioBERT server
(biobert_api_base) used during training MUST be reachable so the LLM-
judge and embedding components match exactly.

Usage:
    GEMINI_API_KEY=... \
    API_BASE=http://localhost:8002/v1 \
    BIOBERT_API_BASE=http://localhost:8003 \
    python scripts/self_evolving/eval_sota_gemini.py \
        --val_file /home/dvdai/scratch/dvdai/self_evolving_datasets/mimiciv_rare/test.jsonl \
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


def _add_repo_to_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "verl").is_dir() and (parent / "verl" / "__init__.py").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_to_path()

from verl.utils.reward_score.self_evolving import compute_score  # noqa: E402


GEMINI_DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _read_image_b64(path: str, max_bytes: int = 6 * 1024 * 1024) -> tuple[str, str] | None:
    """Return (mime_type, base64_data) for the given path, or None if unreadable.

    Caps at max_bytes to keep request bodies manageable.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            data = data[:max_bytes]
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        return mime, base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _build_gemini_request(entry: dict, model_name: str) -> dict:
    """Convert a verl-format prompt entry into a Gemini API request body.

    Gemini's REST API expects a `contents` list of `parts`, where text and
    image parts are interleaved. We follow the order of `<image>` placeholders
    in the user content so the model sees images in the same positions as the
    Qwen3-VL actor would.
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
            blob = _read_image_b64(ipath)
            if blob is not None:
                mime, b64 = blob
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    # Any remaining unmatched image paths get appended after the text.
    for ipath in img_iter:
        blob = _read_image_b64(ipath)
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
        body = _build_gemini_request(entry, args.model_name)
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

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)

    aggregated: dict[str, list[float]] = defaultdict(list)
    error_count = 0
    started = time.time()
    completed = 0

    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_fp = output_path.open("w")
    else:
        out_fp = None

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [asyncio.create_task(_eval_one(sem, session, e, args)) for e in entries]
        for fut in asyncio.as_completed(tasks):
            score = await fut
            completed += 1
            if score.get("error"):
                error_count += 1
            for k in ("acc", "answer_quality", "reasoning_quality",
                      "biobert_sim", "char_bleu", "format_ok", "score"):
                if k in score and isinstance(score[k], (int, float)):
                    aggregated[k].append(float(score[k]))
            if out_fp is not None:
                out_fp.write(json.dumps(score) + "\n")
                out_fp.flush()
            if completed % max(1, len(entries) // 20) == 0:
                elapsed = time.time() - started
                rate = completed / max(elapsed, 1e-9)
                eta_min = (len(entries) - completed) / max(rate, 1e-9) / 60
                acc_so_far = sum(aggregated["acc"]) / max(len(aggregated["acc"]), 1)
                print(
                    f"  [{completed}/{len(entries)}]  err={error_count}  "
                    f"acc={acc_so_far:.4f}  rate={rate:.2f}/s  eta={eta_min:.1f} min"
                )

    if out_fp is not None:
        out_fp.close()

    print()
    print("=" * 70)
    print(f"Gemini evaluation summary  (model={args.model_name})")
    print(f"  total: {completed}, errors: {error_count}")
    print("=" * 70)
    for key in ("score", "acc", "answer_quality", "reasoning_quality",
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
    parser.add_argument("--judge_model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--api_base", default=os.environ.get("API_BASE", "http://localhost:8002/v1"))
    parser.add_argument("--judge_api_key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--biobert_api_base", default=os.environ.get("BIOBERT_API_BASE", "http://localhost:8003"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all entries")
    parser.add_argument("--output_jsonl", default="")
    args = parser.parse_args()

    if not args.gemini_api_key:
        sys.exit("GEMINI_API_KEY (or --gemini_api_key) is required")
    if not args.api_base:
        print("WARNING: api_base is empty — Qwen judge metrics will fall back to defaults")
    if not args.biobert_api_base:
        print("WARNING: biobert_api_base is empty — biobert_sim will be 0.0 for every sample")

    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
