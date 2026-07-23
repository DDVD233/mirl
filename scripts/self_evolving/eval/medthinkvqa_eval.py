#!/usr/bin/env python3
"""Evaluate a (multimodal) chat model on the MedThinkVQA test set.

MedThinkVQA (bio-nlp-umass, ICLR 2026) is a multi-image radiology benchmark. The
public metric we replicate here is *final-answer accuracy* on the Step-3
Differential-Diagnosis task: each case gives a short clinical history, a set of
case images, and five candidate diagnoses (A-E); the model must pick the single
best diagnosis letter.

IMPORTANT — faithfulness note
-----------------------------
The public MedThinkVQA repo (https://github.com/benluwang/MedThinkVQA) ships only
preprocessing utilities plus an OpenAI Responses-API wrapper (`model/gpt.py`); the
actual final-answer eval harness (inference prompt + answer-letter parser) is NOT
released. This script is therefore a faithful *reconstruction*:

  * Request construction mirrors `model/gpt.py::APIModel` — OpenAI Responses API,
    system+user roles, images passed as `input_image` base64 data URLs, an
    explicit `reasoning.effort`, and no sampling params under reasoning effort
    (gpt-5.1/5.2 reject temperature/top_p unless effort == none).
  * Model input = CLINICAL_HISTORY + all case images + the 5 options. The
    IMAGING_FINDINGS free text is deliberately withheld (it is answer-adjacent;
    the benchmark filters text-solvable cases so the task is image-grounded).

For GPT models we drive an Azure/TRAPI-style OpenAI-compatible proxy via
--base-url / --api-key (the openai SDK also honours OPENAI_BASE_URL / OPENAI_API_KEY).

Outputs (written under --output-dir):
  <tag>.jsonl      one line per case: id, gt, pred letter, correct, raw text, usage
  <tag>.summary.json   accuracy + config + breakdowns
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert radiologist. You are given a patient's clinical history and "
    "a set of medical images from a single case, followed by five candidate diagnoses "
    "labelled A through E. Carefully read the history, examine every image, integrate "
    "the cross-image evidence, and choose the single best final diagnosis.\n"
    "Respond with your reasoning, then end your reply with a line in exactly this format:\n"
    "Answer: <LETTER>\n"
    "where <LETTER> is one of A, B, C, D, or E."
)

LETTERS = ["A", "B", "C", "D", "E"]


def build_user_text(row: Dict[str, Any]) -> str:
    history = str(row.get("CLINICAL_HISTORY") or "").strip()
    opts = row.get("options") or {}
    lines = [f"Clinical history: {history}", "", "Candidate diagnoses:"]
    for letter in LETTERS:
        if letter in opts:
            lines.append(f"{letter}. {str(opts[letter]).strip()}")
    lines.append("")
    lines.append(
        "Based on the clinical history and the images, which single option is the "
        "most likely diagnosis? End with 'Answer: <LETTER>'."
    )
    return "\n".join(lines)


def image_paths(row: Dict[str, Any], data_dir: Path, max_images: Optional[int]) -> List[Path]:
    paths: List[Path] = []
    for i in range(1, 50):
        rel = row.get(f"image_{i:02d}_path")
        if not rel:
            continue
        paths.append(data_dir / rel)
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def b64_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def build_input(row: Dict[str, Any], data_dir: Path, max_images: Optional[int], with_images: bool):
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": build_user_text(row)}]
    n_img = 0
    if with_images:
        for p in image_paths(row, data_dir, max_images):
            content.append({"type": "input_image", "image_url": b64_data_url(p)})
            n_img += 1
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        n_img,
    )


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #
_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\**\s*([A-E])\b", re.IGNORECASE)
_STANDALONE_RE = re.compile(r"\b([A-E])\b")


def parse_letter(text: str, options: Dict[str, str]) -> Optional[str]:
    if not text:
        return None
    # 1) explicit "Answer: X" — take the LAST occurrence.
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    # 2) last standalone A-E token.
    tokens = _STANDALONE_RE.findall(text)
    if tokens:
        return tokens[-1].upper()
    # 3) match option text verbatim.
    low = text.lower()
    for letter in LETTERS:
        val = str(options.get(letter, "")).strip().lower()
        if val and val in low:
            return letter
    return None


def extract_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    parts: List[str] = []
    for o in getattr(resp, "output", []) or []:
        if getattr(o, "type", "") == "message":
            for c in getattr(o, "content", []) or []:
                if getattr(c, "type", "") in ("output_text", "input_text"):
                    parts.append(getattr(c, "text", "") or "")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Inference (mirrors model/gpt.py request construction)
# --------------------------------------------------------------------------- #
async def run_one(
    client: AsyncOpenAI,
    row: Dict[str, Any],
    args,
    data_dir: Path,
) -> Dict[str, Any]:
    inputs, n_img = build_input(row, data_dir, args.max_images, not args.no_images)
    req: Dict[str, Any] = {
        "model": args.model_name,
        "input": inputs,
        "max_output_tokens": args.max_output_tokens,
    }
    if args.reasoning_effort and args.reasoning_effort.lower() != "none":
        req["reasoning"] = {"effort": args.reasoning_effort.lower()}
    # gpt-5.1/5.2 reject sampling params unless effort == none; omit temperature.

    last_err = None
    for attempt in range(args.max_retries):
        try:
            resp = await client.responses.create(**req)
            text = extract_text(resp)
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            pred = parse_letter(text, row.get("options") or {})
            gt = str(row.get("correct_answer") or "").strip().upper()
            return {
                "title": row.get("title"),
                "gt": gt,
                "pred": pred,
                "correct": (pred == gt) if pred else False,
                "n_images": n_img,
                "status": getattr(resp, "status", None),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "raw": text,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 20) + 0.5)
    return {
        "title": row.get("title"),
        "gt": str(row.get("correct_answer") or "").strip().upper(),
        "pred": None,
        "correct": False,
        "n_images": n_img,
        "status": "error",
        "error": f"{type(last_err).__name__}: {last_err}",
        "raw": "",
    }


async def run_all(rows: List[Dict[str, Any]], args, data_dir: Path, out_path: Path) -> List[Dict[str, Any]]:
    client = AsyncOpenAI(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "x"),
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        timeout=args.request_timeout,
        max_retries=0,
    )
    sem = asyncio.Semaphore(args.concurrency)
    results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    done = 0
    lock = asyncio.Lock()
    t0 = time.time()

    async def worker(idx: int, row: Dict[str, Any]):
        nonlocal done
        async with sem:
            res = await run_one(client, row, args, data_dir)
        results[idx] = res
        async with lock:
            done += 1
            if done % 10 == 0 or done == len(rows):
                acc = sum(1 for r in results if r and r["correct"]) / done
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  [{done}/{len(rows)}] running_acc={acc:.4f} ({rate:.2f}/s)", flush=True)

    await asyncio.gather(*(worker(i, r) for i, r in enumerate(rows)))
    final = [r for r in results if r is not None]
    with out_path.open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return final


# --------------------------------------------------------------------------- #
def summarize(results: List[Dict[str, Any]], args) -> Dict[str, Any]:
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_parsed = sum(1 for r in results if r["pred"])
    n_err = sum(1 for r in results if r.get("status") == "error")
    pred_dist = Counter(r["pred"] for r in results)
    gt_dist = Counter(r["gt"] for r in results)
    return {
        "model": args.model_name,
        "reasoning_effort": args.reasoning_effort,
        "with_images": not args.no_images,
        "max_images": args.max_images,
        "max_output_tokens": args.max_output_tokens,
        "n": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "n_parsed": n_parsed,
        "n_unparsed": n - n_parsed,
        "n_errors": n_err,
        "pred_letter_dist": dict(sorted(pred_dist.items(), key=lambda x: str(x[0]))),
        "gt_letter_dist": dict(sorted(gt_dist.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/scratch/dvd/medthinkvqa", help="Dir with test.jsonl + images/")
    ap.add_argument("--model-name", default="gpt-5.1_2025-11-13")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base url (TRAPI proxy)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--reasoning-effort", default="medium", help="none|low|medium|high|xhigh")
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all case images")
    ap.add_argument("--no-images", action="store_true", help="text-only ablation")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    ap.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/results_medthinkvqa")
    ap.add_argument("--tag", default=None, help="output filename stem")
    args = ap.parse_args()
    if args.max_images == 0:
        args.max_images = None

    data_dir = Path(args.data_dir)
    rows = [json.loads(l) for l in (data_dir / "test.jsonl").read_text().splitlines() if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    tag = args.tag or (
        f"{re.sub(r'[^A-Za-z0-9.]+', '-', args.model_name)}"
        f"_{'noimg' if args.no_images else 'img'}_{args.reasoning_effort}"
        f"{('_n' + str(len(rows))) if args.limit else ''}"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.jsonl"
    sum_path = out_dir / f"{tag}.summary.json"

    print(f"MedThinkVQA eval | model={args.model_name} effort={args.reasoning_effort} "
          f"images={not args.no_images} n={len(rows)} conc={args.concurrency}", flush=True)
    t0 = time.time()
    results = asyncio.run(run_all(rows, args, data_dir, out_path))
    summary = summarize(results, args)
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["output_jsonl"] = str(out_path)
    sum_path.write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}\nwrote {sum_path}")


if __name__ == "__main__":
    main()
