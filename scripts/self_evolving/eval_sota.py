"""Evaluate a SoTA model on the MIMIC-IV rare-disease test set.

Supports Gemini (``--provider gemini``) and OpenAI-compatible chat
endpoints (``--provider openai``, e.g. GPT-5.5). Uses the EXACT same
scoring pipeline as training/val
(verl.utils.reward_score.self_evolving.compute_score) so metrics are
directly comparable to wandb's val-core / val-aux numbers from training.

Pipeline per test entry:
    1. Send the system + user prompt (with chest X-ray + ECG images) to
       the provider.
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

Usage (Gemini):
    GEMINI_API_KEY=... \
    API_BASE=http://node2500:8002/v1 \
    MODEL_NAME=Qwen/Qwen3.6-27B \
    BIOBERT_API_BASE=http://localhost:8003 \
    python scripts/self_evolving/eval_sota.py \
        --provider gemini \
        --val_file $HOME/scratch/dvdai/self_evolving_datasets/mimiciv_rare/test.jsonl \
        --model_name gemini-3.1-pro-preview \
        --concurrency 8 \
        --limit 200

Usage (GPT-5.5):
    OPENAI_API_KEY=... \
    API_BASE=http://node2500:8002/v1 \
    MODEL_NAME=Qwen/Qwen3.6-27B \
    BIOBERT_API_BASE=http://localhost:8003 \
    python scripts/self_evolving/eval_sota.py \
        --provider openai \
        --model_name gpt-5.5 \
        --val_file $HOME/scratch/dvdai/self_evolving_datasets/mimiciv_rare/test.jsonl \
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
OPENAI_DEFAULT_MODEL = "gpt-5.5"
OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
KIMI_DEFAULT_MODEL = "kimi-k2.6"
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/v1"
PROVIDER_DEFAULT_MODEL = {
    "gemini": GEMINI_DEFAULT_MODEL,
    "openai": OPENAI_DEFAULT_MODEL,
    "kimi": KIMI_DEFAULT_MODEL,
}


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


def _extract_prompt_pieces(
    entry: dict, max_pixels: int, max_text_chars: int,
) -> tuple[str, list[tuple[str, str]], list[str | None]]:
    """Parse a verl-format prompt entry.

    Returns ``(sys_text, image_blobs, text_pieces)`` where ``text_pieces`` is
    the user text split on ``<image>``, and ``image_blobs`` is a list of
    ``(mime, b64)`` tuples paired with each placeholder (or ``None`` if the
    image was unreadable / unavailable). Trailing unmatched images are
    appended after the last text piece.

    Same image-resize / text-truncate parity rules as the training preprocess.
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
    text_pieces = re.split(r"<image>", user_text)

    # Align images with placeholders, then append the rest after.
    image_blobs: list[tuple[str, str] | None] = []
    img_iter = iter(image_paths)
    for _ in range(max(0, len(text_pieces) - 1)):
        try:
            ipath = next(img_iter)
        except StopIteration:
            image_blobs.append(None)
            continue
        image_blobs.append(_read_image_b64(ipath, max_pixels=max_pixels))
    trailing: list[tuple[str, str] | None] = []
    for ipath in img_iter:
        trailing.append(_read_image_b64(ipath, max_pixels=max_pixels))
    return sys_text, image_blobs, text_pieces, trailing  # type: ignore[return-value]


def _build_gemini_request(
    entry: dict,
    model_name: str,
    max_pixels: int = 256 * 256,
    max_text_chars: int = 9000,
) -> dict:
    """Convert a verl-format prompt entry into a Gemini API request body."""
    sys_text, image_blobs, text_pieces, trailing = _extract_prompt_pieces(
        entry, max_pixels=max_pixels, max_text_chars=max_text_chars,
    )

    parts: list[dict] = []
    for i, piece in enumerate(text_pieces):
        if piece:
            parts.append({"text": piece})
        if i < len(text_pieces) - 1:
            blob = image_blobs[i]
            if blob is not None:
                mime, b64 = blob
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    for blob in trailing:
        if blob is not None:
            mime, b64 = blob
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})

    body: dict = {
        "contents": [{"role": "user", "parts": parts}],
        # Gemini 2.5+ / 3.x preview models are "thinking" models: internal
        # reasoning tokens count against maxOutputTokens. includeThoughts=False
        # keeps the raw chain-of-thought out of the returned text — we score
        # only the final boxed answer via compute_score.
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 16384,
            "thinkingConfig": {"includeThoughts": False},
        },
    }
    if sys_text:
        body["systemInstruction"] = {"parts": [{"text": sys_text}]}
    return body


def _build_openai_request(
    entry: dict,
    model_name: str,
    max_pixels: int = 256 * 256,
    max_text_chars: int = 9000,
) -> dict:
    """Convert a verl-format prompt entry into an OpenAI chat-completions body.

    Images are inlined as ``data:<mime>;base64,...`` URLs in ``image_url``
    content parts. The system message goes into a separate ``role:system``
    entry to match how training-time SYSTEM_PROMPT is applied.
    """
    sys_text, image_blobs, text_pieces, trailing = _extract_prompt_pieces(
        entry, max_pixels=max_pixels, max_text_chars=max_text_chars,
    )

    parts: list[dict] = []
    for i, piece in enumerate(text_pieces):
        if piece:
            parts.append({"type": "text", "text": piece})
        if i < len(text_pieces) - 1:
            blob = image_blobs[i]
            if blob is not None:
                mime, b64 = blob
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
    for blob in trailing:
        if blob is not None:
            mime, b64 = blob
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

    messages: list[dict] = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": parts})

    # NOTE: GPT-5+ reasoning models only accept the default temperature (1).
    # max_completion_tokens still works but reasoning tokens count against it,
    # so we leave it generous and let the model decide.
    return {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": 16384,
    }


def _build_kimi_request(
    entry: dict,
    model_name: str,
    max_pixels: int = 256 * 256,
    max_text_chars: int = 9000,
    enable_thinking: bool = True,
) -> dict:
    """Kimi k2.6 chat-completions body.

    Same OpenAI HTTP shape as _build_openai_request, but:
      - Kimi uses ``max_tokens``, not ``max_completion_tokens``.
      - Reasoning is controlled via ``thinking={"type": "enabled"|"disabled"}``
        (not by temperature). Default enabled — for diagnosis tasks the
        long CoT is worth the latency.
      - Kimi hard-fixes temperature / top_p / etc. and 400s if you pass
        them explicitly, so we omit those entirely.
    """
    # Reuse the openai builder for the messages array (system + multimodal
    # user content), then rewrite the top-level params for Kimi.
    body = _build_openai_request(
        entry, model_name,
        max_pixels=max_pixels, max_text_chars=max_text_chars,
    )
    body.pop("max_completion_tokens", None)
    body["max_tokens"] = 16384
    body["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
    return body


# Provider HTTP timeout. Bumped well above the original 180s because Kimi
# k2.6 with thinking enabled routinely takes 3-5 minutes per call (long
# medical CoT). Override with EVAL_HTTP_TIMEOUT env if you want a tighter
# bound. EVAL_HTTP_RETRIES caps attempts on transient failures.
_HTTP_TIMEOUT_TOTAL = float(os.environ.get("EVAL_HTTP_TIMEOUT", "600"))
_HTTP_RETRIES = int(os.environ.get("EVAL_HTTP_RETRIES", "4"))


# Errors worth retrying:
#   asyncio.TimeoutError       — the whole-request timeout fired
#   aiohttp.ClientError        — connection reset, DNS hiccup, 5xx convert,
#                                ServerDisconnectedError, etc.
#   RuntimeError "HTTP 5xx"    — provider 5xx wrapped by our callers below
# We do NOT retry on 4xx (caller's fault — bad model name, bad auth).
_TRANSIENT_EXCEPTIONS = (asyncio.TimeoutError, aiohttp.ClientError)


def _is_retryable_runtime_error(e: BaseException) -> bool:
    msg = str(e)
    if "HTTP 5" in msg:
        return True
    if "HTTP 429" in msg:  # rate limit
        return True
    return False


async def _retry_http(label: str, coro_factory):
    """Run ``await coro_factory()`` with retry/backoff on transient failures.

    ``coro_factory`` is a no-arg callable returning a fresh coroutine each
    attempt (since coroutines aren't reusable). Backoff: 5s, 15s, 45s,
    135s (exponential x3 with a 5s base). Total worst-case wait between
    a successful 4th attempt and the first failure: ~200s.
    """
    last_err: BaseException | None = None
    for attempt in range(_HTTP_RETRIES):
        try:
            return await coro_factory()
        except _TRANSIENT_EXCEPTIONS as e:
            last_err = e
        except RuntimeError as e:
            if not _is_retryable_runtime_error(e):
                raise
            last_err = e
        if attempt < _HTTP_RETRIES - 1:
            delay = 5 * (3 ** attempt)
            print(
                f"[retry {label}] attempt {attempt+1}/{_HTTP_RETRIES} failed: "
                f"{type(last_err).__name__}: {str(last_err)[:200]} — "
                f"sleeping {delay}s",
                file=sys.stderr, flush=True,
            )
            await asyncio.sleep(delay)
    raise last_err  # type: ignore[misc]


async def _call_gemini(session: aiohttp.ClientSession, model_name: str, api_key: str, body: dict) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    async def _one_attempt() -> str:
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_TOTAL)
        async with session.post(url, json=body, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"Gemini HTTP {resp.status}: {text[:2000]}")
            data = json.loads(text)
        cands = data.get("candidates") or []
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", []) or []
        return "".join(p.get("text", "") for p in parts)

    return await _retry_http(f"gemini/{model_name}", _one_attempt)


async def _call_openai(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    body: dict,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async def _one_attempt() -> str:
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_TOTAL)
        async with session.post(url, json=body, headers=headers, timeout=timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"OpenAI HTTP {resp.status}: {text[:2000]}")
            data = json.loads(text)
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message", {}) or {}
        content = msg.get("content")
        if isinstance(content, list):
            # Reasoning-style models can return content as a list of parts.
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    label = body.get("model") or os.path.basename(base_url.rstrip("/"))
    return await _retry_http(f"openai/{label}", _one_attempt)


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
    """Generate with the chosen provider, then score with the verl pipeline."""
    async with sem:
        try:
            if args.provider == "gemini":
                body = _build_gemini_request(
                    entry,
                    args.model_name,
                    max_pixels=args.max_pixels,
                    max_text_chars=args.max_text_chars,
                )
                response = await _call_gemini(
                    session, args.model_name, args.gemini_api_key, body,
                )
            elif args.provider == "openai":
                body = _build_openai_request(
                    entry,
                    args.model_name,
                    max_pixels=args.max_pixels,
                    max_text_chars=args.max_text_chars,
                )
                response = await _call_openai(
                    session, args.openai_base_url, args.openai_api_key, body,
                )
            elif args.provider == "kimi":
                body = _build_kimi_request(
                    entry,
                    args.model_name,
                    max_pixels=args.max_pixels,
                    max_text_chars=args.max_text_chars,
                    enable_thinking=args.kimi_thinking,
                )
                response = await _call_openai(
                    session, args.openai_base_url, args.openai_api_key, body,
                )
            else:
                raise RuntimeError(f"unknown provider: {args.provider}")
        except Exception as e:
            response = ""
            err = f"{type(e).__name__}: {e}"
            print(f"[ERROR provider={args.provider} model={args.model_name}] {err}",
                  file=sys.stderr, flush=True)
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
    print(
        f"Provider: {args.provider}, model: {args.model_name}, "
        f"judge: {args.judge_model_name} @ {args.api_base}"
    )
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
    print(f"SoTA evaluation summary  (provider={args.provider}, model={args.model_name})")
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
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai", "kimi"),
        default=os.environ.get("EVAL_PROVIDER", "gemini"),
        help="Which API to call for the candidate model.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Model ID. Defaults to gemini-3.1-pro-preview / gpt-5.5 / "
             "kimi-k2.6 for gemini / openai / kimi respectively.",
    )
    parser.add_argument(
        "--kimi_thinking",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "on"),
        default=True,
        help="When --provider kimi, send thinking={type:enabled}; pass False "
             "to use thinking-disabled mode (faster, fixed temp 0.6).",
    )
    parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument(
        "--openai_api_key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (only required when --provider openai).",
    )
    parser.add_argument(
        "--openai_base_url",
        default=os.environ.get("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE),
        help="OpenAI-compatible chat-completions base URL (no trailing /chat/completions).",
    )
    parser.add_argument("--judge_model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-27B"))
    parser.add_argument("--api_base", default=os.environ.get("API_BASE", "http://node2500:8002/v1"))
    parser.add_argument("--judge_api_key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--biobert_api_base", default=os.environ.get("BIOBERT_API_BASE", "http://localhost:8003"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all entries")
    parser.add_argument(
        "--output_jsonl",
        default="",
        help="Path to per-sample JSONL. Defaults to ./eval_<provider>_<model>.jsonl. "
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

    if args.model_name is None:
        args.model_name = PROVIDER_DEFAULT_MODEL[args.provider]
    # Auto-flip openai_base_url to the Kimi endpoint when the user picked
    # --provider kimi but didn't explicitly override the base URL. We keep
    # the same --openai_api_key arg / OPENAI_API_KEY env (or you can pass
    # --openai_api_key=$MOONSHOT_API_KEY) — saves adding kimi-* twins.
    if args.provider == "kimi" and args.openai_base_url == OPENAI_DEFAULT_BASE:
        args.openai_base_url = KIMI_DEFAULT_BASE
    if args.provider == "gemini" and not args.gemini_api_key:
        sys.exit("GEMINI_API_KEY (or --gemini_api_key) is required for --provider gemini")
    if args.provider == "openai" and not args.openai_api_key:
        sys.exit("OPENAI_API_KEY (or --openai_api_key) is required for --provider openai")
    if args.provider == "kimi" and not args.openai_api_key:
        sys.exit("Set MOONSHOT_API_KEY (or pass --openai_api_key) for --provider kimi")
    if not args.api_base:
        print("WARNING: api_base is empty — Qwen judge metrics will fall back to defaults")
    if not args.biobert_api_base:
        print("WARNING: biobert_api_base is empty — biobert_sim will be 0.0 for every sample")

    if not args.output_jsonl:
        # Stable default — no timestamp — so re-running resumes the prior run
        # for the same model. Override with --output_jsonl if you want a fresh
        # file or to keep multiple runs separate.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_name).strip("_")
        args.output_jsonl = f"eval_{args.provider}_{safe}.jsonl"

    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
