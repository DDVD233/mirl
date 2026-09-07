"""Standalone MedXpertQA runner: generate against an OpenAI-compatible endpoint, score.

MedXpertQA (TsinghuaC3I/MedXpertQA, arXiv 2501.18362) is expert-level medical
multiple choice: Text = 2450 questions x 10 options (A-J), MM = 2000 questions x
5 options (A-E) with one or more images. Official scoring is exact match on the
chosen letter, which is what this reports.

Two graders, both optional to run together:
  * exact match on the parsed letter -- the official protocol, and the number
    comparable to published MedXpertQA results;
  * judge accuracy -- the SAME one-criterion rubric the in-loop val parquet uses
    (see preprocess_medxpertqa.py), so a checkpoint's number here lines up with
    the val curve of a training arm. Only worth running when a response's final
    choice is genuinely ambiguous to the parser.

The prompt (including the "Final answer: (X)" instruction) is byte-identical to
the val parquet's, so the two paths grade the same task.

Example -- frozen 9B on the dedicated inference box:
    python scripts/self_evolving/eval/medxpertqa_eval.py --subset text \
        --base_url http://point.dd.works:18186/v1 --model Qwen/Qwen3.5-9B \
        --out /scratch/sheng/self_evolving/eval_out/medxpertqa_text_base9b.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path("/scratch/sheng/self_evolving/bench_raw/MedXpertQA")
SUBSET_DIR = {"text": "Text", "mm": "MM"}

# Keep IN SYNC with ANSWER_FORMAT_INSTR in preprocess_medxpertqa.py.
ANSWER_FORMAT_INSTR = (
    "\n\nWork through the case, then end your response with a single final line of "
    "exactly this form:\nFinal answer: (X)\nwhere X is the letter of the single best option."
)

JUDGE_TEMPLATE = """You are grading one criterion against a response.

CRITERION: {criterion}

RESPONSE:
{response}

Did the response meet the criterion? Reply with exactly one word: yes or no."""


def parse_letter(text: str, valid: str) -> str | None:
    """Best-effort extraction of the committed option letter.

    Ordered most explicit first; each pattern is anchored on a commitment phrase so
    an option merely *mentioned* mid-reasoning is never picked up. The final
    fallback (last bare parenthesised letter) fires only when nothing else matched.
    """
    if not text:
        return None
    # Strip a reasoning block so its candidate-weighing never wins over the answer.
    body = text
    if "</think>" in body:
        body = body.rsplit("</think>", 1)[1]
    cls = f"[{valid}]"
    patterns = [
        rf"final answer:\s*\(?({cls})\)?",
        rf"\\boxed\{{\(?({cls})\)?\}}",
        rf"(?:the )?(?:correct |best )?(?:answer|option|choice) is:?\s*\(?({cls})\)?",
        rf"^\s*\(?({cls})\)?\s*$",
    ]
    for pat in patterns:
        m = list(re.finditer(pat, body, re.IGNORECASE | re.MULTILINE))
        if m:
            return m[-1].group(1).upper()
    m = list(re.finditer(rf"\(({cls})\)", body))
    return m[-1].group(1).upper() if m else None


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_messages(ex: dict, subset: str, images_dir: Path, system: str | None) -> list[dict]:
    text = str(ex["question"]) + ANSWER_FORMAT_INSTR
    if subset == "mm":
        parts = []
        for rel in ex.get("images") or []:
            p = images_dir / rel
            if not p.is_file():
                raise FileNotFoundError(p)
            parts.append({"type": "image_url", "image_url": {"url": _data_uri(p)}})
        parts.append({"type": "text", "text": text})
        content: object = parts
    else:
        content = text
    msgs = [{"role": "user", "content": content}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


async def _post(client, url: str, payload: dict, headers: dict, retries: int, timeout: float):
    last = None
    for attempt in range(retries):
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:  # network flake, engine restart, frp blip
            last = f"{type(e).__name__}: {e}"
        await asyncio.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


async def generate_all(rows, args, images_dir: Path):
    import httpx

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    sem = asyncio.Semaphore(args.concurrency)
    done = {"n": 0}
    t0 = time.time()

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one(ex):
            async with sem:
                payload = {
                    "model": args.model,
                    "messages": build_messages(ex, args.subset, images_dir, args.system),
                }
                # TRAPI chat deployments (gpt-chat-latest) reject max_tokens and
                # temperature outright; the proxy is otherwise OpenAI-compatible.
                if not args.trapi:
                    payload["max_tokens"] = args.max_tokens
                    payload["temperature"] = args.temperature
                try:
                    data = await _post(client, url, payload, headers, args.retries, args.timeout)
                    choice = data["choices"][0]
                    msg = choice["message"]
                    text = msg.get("content") or ""
                    # Thinking models served by vLLM put the reasoning channel in a
                    # separate field and leave `content` EMPTY until </think> closes.
                    # A response truncated mid-thought therefore looks blank; keep the
                    # reasoning so the parser can still find a committed answer, and so
                    # a truncation shows up as a long reasoning rather than a mystery.
                    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    reason = choice.get("finish_reason")
                except Exception as e:
                    text, reasoning, reason = "", "", f"error:{type(e).__name__}"
                done["n"] += 1
                if done["n"] % 100 == 0:
                    rate = done["n"] / max(time.time() - t0, 1e-6)
                    print(f"  {done['n']}/{len(rows)} ({rate:.2f}/s)", flush=True)
                return {"id": ex["id"], "response": text, "reasoning": reasoning,
                        "finish_reason": reason}

        return await asyncio.gather(*(one(ex) for ex in rows))


async def judge_all(items, args):
    """Grade the one-criterion rubric with the val judge (matches in-loop val)."""
    import httpx

    url = args.judge_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {args.judge_api_key}"} if args.judge_api_key else {}
    sem = asyncio.Semaphore(args.judge_concurrency)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one(it):
            async with sem:
                criterion = (
                    f"The response's single final answer is option ({it['gold']}): "
                    f"\"{it['gold_text']}\". Award the point ONLY if the response commits to "
                    f"({it['gold']}) as its final choice."
                )
                payload = {
                    "model": args.judge_model,
                    "messages": [{"role": "user", "content": JUDGE_TEMPLATE.format(
                        criterion=criterion, response=it["response"][:20000])}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                }
                try:
                    data = await _post(client, url, payload, headers, args.retries, args.timeout)
                    verdict = (data["choices"][0]["message"]["content"] or "").strip().lower()
                    return verdict.startswith("yes")
                except Exception:
                    return None

        return await asyncio.gather(*(one(it) for it in items))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, choices=sorted(SUBSET_DIR))
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", default=os.environ.get("MEDX_API_KEY", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw_dir", default=str(RAW_DIR))
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=12288,
                    help="thinking models spend most of this on reasoning; too small "
                         "returns an EMPTY answer channel, not a short answer")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--trapi", action="store_true",
                    help="target a TRAPI deployment: omit max_tokens/temperature from the request")
    ap.add_argument("--system", default=None, help="optional system prompt (default: bare, "
                                                   "matching the val parquet)")
    ap.add_argument("--dump_responses", action="store_true")
    # optional judge pass
    ap.add_argument("--judge_base_url", default=None)
    ap.add_argument("--judge_model", default=None)
    ap.add_argument("--judge_api_key", default=os.environ.get("MEDX_JUDGE_API_KEY", ""))
    ap.add_argument("--judge_concurrency", type=int, default=16)
    ap.add_argument("--judge_only_unparsed", action="store_true",
                    help="judge only the responses the parser could not resolve")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    images_dir = Path(args.images_dir) if args.images_dir else raw_dir / "images"
    raw_path = raw_dir / SUBSET_DIR[args.subset] / f"{args.split}.jsonl"
    rows = [json.loads(x) for x in raw_path.read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[medxpertqa] {len(rows)} {args.subset} rows from {raw_path}", flush=True)

    gens = asyncio.run(generate_all(rows, args, images_dir))
    by_id = {g["id"]: g for g in gens}

    items = []
    for ex in rows:
        g = by_id[ex["id"]]
        valid = "".join(sorted(ex["options"]))
        # The answer belongs in `content`; fall back to the reasoning channel only
        # when the answer channel is empty (a response cut off mid-thought), where
        # a stated conclusion is still the model's answer.
        pred = parse_letter(g["response"], valid)
        if pred is None and not g["response"]:
            pred = parse_letter(g.get("reasoning", ""), valid)
        items.append({
            "id": ex["id"],
            "gold": ex["label"],
            "gold_text": ex["options"][ex["label"]],
            "pred": pred,
            "correct": bool(pred == ex["label"]),
            "response": g["response"],
            "reasoning_chars": len(g.get("reasoning", "")),
            "finish_reason": g["finish_reason"],
            "medical_task": ex.get("medical_task"),
            "question_type": ex.get("question_type"),
            "body_system": ex.get("body_system"),
        })

    if args.judge_base_url and args.judge_model:
        targets = [it for it in items if (it["pred"] is None or not args.judge_only_unparsed)]
        print(f"[judge] grading {len(targets)} responses with {args.judge_model}", flush=True)
        verdicts = asyncio.run(judge_all(targets, args))
        for it, v in zip(targets, verdicts):
            it["judge_correct"] = v

    n = len(items)
    acc = sum(it["correct"] for it in items) / n
    unparsed = sum(it["pred"] is None for it in items)
    empty = sum(not it["response"] for it in items)
    truncated = sum(it["finish_reason"] == "length" for it in items)
    errored = sum(str(it["finish_reason"]).startswith("error:") for it in items)

    def _breakdown(key):
        agg = defaultdict(lambda: [0, 0])
        for it in items:
            a = agg[str(it.get(key))]
            a[0] += it["correct"]
            a[1] += 1
        return {k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(agg.items())}

    summary = {
        "subset": args.subset,
        "model": args.model,
        "base_url": args.base_url,
        "n": n,
        "accuracy": round(acc, 4),
        "unparsed": unparsed,
        "empty_responses": empty,
        "truncated": truncated,
        "errored": errored,
        "median_reasoning_chars": sorted(it["reasoning_chars"] for it in items)[len(items) // 2],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "by_medical_task": _breakdown("medical_task"),
        "by_question_type": _breakdown("question_type"),
        "by_body_system": _breakdown("body_system"),
        "pred_distribution": dict(Counter(str(it["pred"]) for it in items).most_common()),
    }
    judged = [it for it in items if it.get("judge_correct") is not None]
    if judged:
        summary["judge_model"] = args.judge_model
        summary["judge_n"] = len(judged)
        summary["judge_accuracy_on_judged"] = round(
            sum(it["judge_correct"] for it in judged) / len(judged), 4)
        # Full-split accuracy taking the judge as truth wherever it ran.
        summary["accuracy_judge_resolved"] = round(sum(
            it["judge_correct"] if it.get("judge_correct") is not None else it["correct"]
            for it in items) / n, 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary}
    if args.dump_responses:
        payload["items"] = items
    else:
        payload["items"] = [{k: v for k, v in it.items() if k != "response"} for it in items]
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
