"""Teacher-written reasoning traces for the MedXpertQA stage-1 SFT set.

The same design as make_distill_traces.py (which is MIMIC/ICD-specific: it mandates
\\boxed{ICD-10 CODE: ...} and matches answers ICD-aware), generalised to the two
source shapes this arm needs:

  * ``mcq``  -- MedQA. The trace ends ``Final answer: (X)``, the exact instruction and
    final-answer form eval/preprocess_medxpertqa.py puts in front of the policy, so
    SFT teaches the shape validation asks for rather than something adjacent.
  * ``open`` -- CLIMB images. The trace ends ``Final answer: <finding>``.

FOUR GATES, all of them load-bearing:

  1. GROUND TRUTH. The teacher is told the answer *confidentially* and asked to reason
     TO it. Any trace whose own final answer disagrees is dropped, so every kept
     example is correct by construction.
  2. NO LEAK. The student never sees the answer, so a trace that says "as given" or
     "the correct answer is stated" teaches a move the student cannot make. Dropped.
  3. LENGTH. A two-line trace teaches nothing and a runaway one teaches rambling.
  4. DIVERSITY. Up to --n_per_question distinct traces, near-duplicates rejected, so
     the later RL stage sees more than one route to the same answer.

The teacher SEES the image for ``open`` rows -- the whole point of the multimodal
half is reasoning that cites what is actually visible.

    CHAT_PROVIDER=trapi python scripts/self_evolving/make_medxpert_traces.py \\
        --source /scratch/sheng/self_evolving/medxpert_sft/source.jsonl \\
        --out    /scratch/sheng/self_evolving/medxpert_sft/traces.jsonl \\
        --api_base http://point.dd.works:18890/v1 --model_name gpt-chat-latest_2026-05-28
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import mimetypes
import os
import random
import re
import sys
from collections import Counter

SYSTEM = (
    "You are a senior physician writing a teaching trace for a medical trainee.\n"
    "You will be given a case and, CONFIDENTIALLY, its verified correct answer.\n"
    "Write the reasoning that ARRIVES at that answer, as if you were solving it "
    "yourself for the first time.\n\n"
    "Format, exactly:\n"
    "<think>\n"
    "{your first-person clinical reasoning}\n"
    "</think>\n\n"
    "{one short paragraph stating the answer and why}\n"
    "{FINAL_LINE}\n\n"
    "Hard rules:\n"
    "- NEVER reveal, hint, or imply that you were given the answer. No 'as given', "
    "'we are told', 'the correct answer is', 'the provided diagnosis'. The trainee "
    "sees only the case, so the reasoning must read as a forward solve.\n"
    "- Weigh the alternatives and say what RULES THEM OUT, using the specific "
    "findings in the case.\n"
    "- Ground every claim in the case's own details (values, timing, exam findings"
    "{IMAGE_CLAUSE}).\n"
    "- Aim for roughly {TARGET} words of reasoning inside <think>."
)

_LEAK_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(as|which is|that is)\s+(given|provided|stated|specified|told)\b",
    r"\b(we|i|you)\s+(are|were|have been)\s+(given|told|provided)\b",
    r"\bground[\s-]?truth\b",
    r"\bthe\s+(correct|right|true|verified|confirmed|known)\s+(answer|diagnosis|finding|option|label)\b",
    r"\b(answer|diagnosis|finding|label|option)\s+(is\s+)?(given|provided|supplied|stated)\b",
    r"\bconfidential(ly)?\b",
    r"\baccording to the (answer|label|key)\b",
    r"\bsince the answer\b",
)]

_FINAL_MCQ = re.compile(r"final answer:\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE)
_FINAL_OPEN = re.compile(r"final answer:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()


def _norm_set(s: str) -> set:
    return {t for t in _norm(s).replace(",", " ").split() if t}


def _answer_ok(kind: str, text: str, gt: str) -> bool:
    """Does the trace's OWN final answer agree with the ground truth?"""
    if kind == "mcq":
        m = _FINAL_MCQ.findall(text)
        return bool(m) and m[-1].strip().upper() == str(gt).strip().upper()
    m = _FINAL_OPEN.findall(text)
    if not m:
        return False
    got, want = _norm(m[-1]), _norm(gt)
    if got == want:
        return True
    # Open-set labels are short phrases ("No DR", "Acute PE"); accept an exact
    # token-set match so trivial wording differences do not throw away a good trace,
    # but never accept a mere overlap -- "No PE" vs "Acute PE" share a token.
    return _norm_set(got) == _norm_set(want)


def _leaks(text: str) -> str:
    for rx in _LEAK_RES:
        m = rx.search(text)
        if m:
            return m.group(0)
    return ""


# Refusal boilerplate. OpenAI-family chat deployments decline to emit visible
# chain-of-thought and answer with a disclaimer plus a short summary ("I can't
# provide detailed internal chain-of-thought reasoning. Instead, ..."). Those pass
# every content gate -- the answer is right, nothing leaks -- while teaching the
# student to REFUSE TO REASON, which is the opposite of the point. Use a teacher
# that writes reasoning (the MIMIC recipe used a locally served Qwen3.6-27B) and
# drop these on sight.
_REFUSAL_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(i|we)\s+(can'?t|cannot|won'?t|am not able to|will not)\b",
    r"\bchain[\s-]?of[\s-]?thought\b",
    r"\b(internal|hidden|private)\s+(reasoning|thought|deliberation)\b",
    r"\bas an ai\b",
    r"\binstead,? here is\b",
)]


def _refuses(text: str) -> str:
    for rx in _REFUSAL_RES:
        m = rx.search(text)
        if m:
            return m.group(0)
    return ""


def _think_words(text: str) -> int:
    """Words inside <think>...</think>; -1 when the block is absent.

    Absent is a HARD reject, not a fallback: the whole product is a reasoning trace
    in the shape the student is trained to imitate, and a bare answer with no block
    teaches the wrong format however good its prose.
    """
    m = _THINK.search(text)
    if not m:
        return -1
    return len(m.group(1).split())


def _too_similar(a: str, b: str, thr: float) -> bool:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return False
    return len(ta & tb) / max(1, len(ta | tb)) >= thr


def _data_uri(path: str, max_pixels: int) -> str:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max_pixels and w * h > max_pixels:
                s = (max_pixels / float(w * h)) ** 0.5
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            raw = buf.getvalue()
        mime = "image/jpeg"
    except Exception:
        with open(path, "rb") as fh:
            raw = fh.read()
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _user_content(row: dict, gt: str, max_pixels: int):
    question = "\n".join(str(m.get("content") or "") for m in row.get("prompt", []))
    text = (f"CASE:\n{question}\n\n"
            f"CONFIDENTIAL -- the verified correct answer is: {gt}\n\n"
            f"Write the teaching trace now.")
    images = row.get("images") or []
    if not images:
        return text
    parts = [{"type": "image_url", "image_url": {"url": _data_uri(p, max_pixels)}}
             for p in images]
    parts.append({"type": "text", "text": text})
    return parts


async def _call(client, args, messages) -> str:
    """Return the trace as ``<think>reasoning</think>\n\nanswer``.

    vLLM's reasoning parser moves the model's thinking OUT of ``content`` into a
    separate ``reasoning`` field, so a thinking teacher returns an answer with no
    <think> block however the prompt asks for one -- and the mandatory-block gate
    would then reject every trace it produced. Reassembling from both channels is
    what makes the teacher's genuine reasoning the imitation target, rather than a
    second, performed one written for the prompt.
    """
    payload = {"model": args.model_name, "messages": messages}
    if args.provider == "trapi":
        payload["max_completion_tokens"] = args.max_tokens
    else:
        payload["max_tokens"] = args.max_tokens
        payload["temperature"] = args.temperature
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    headers = {"Authorization": f"Bearer {args.api_key}"}
    url = args.api_base.rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(args.retries):
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=args.timeout)
            if r.status_code == 200:
                m = r.json()["choices"][0]["message"]
                content = (m.get("content") or "").strip()
                reasoning = (m.get("reasoning") or m.get("reasoning_content") or "").strip()
                if not content and not reasoning:
                    return ""
                if "<think>" in content.lower():
                    return content
                if reasoning:
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                return content
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = f"{type(e).__name__}"
        await asyncio.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))
    raise RuntimeError(f"teacher unreachable: {last}")


async def _run(args) -> int:
    import httpx

    rows = [json.loads(l) for l in open(args.source) if l.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    print(f"[source] {len(rows)} questions", flush=True)

    stats = Counter()
    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    fout = open(args.out, "w")

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one(row):
            kind = (row.get("extra_info") or {}).get("kind", "mcq")
            gt = (row.get("reward_model") or {}).get("ground_truth", "")
            if kind == "mcq":
                gt_show = f"({gt}) {(row.get('extra_info') or {}).get('answer_text','')}".strip()
                final_line = "Final answer: (X)"
            else:
                gt_show = str(gt)
                final_line = "Final answer: <finding>"
            sys_prompt = (SYSTEM
                          .replace("{FINAL_LINE}", final_line)
                          .replace("{TARGET}", str(args.target_words))
                          .replace("{IMAGE_CLAUSE}",
                                   ", and what is visible in the image" if row.get("images") else ""))
            kept: list[str] = []
            async with sem:
                for _ in range(args.oversample):
                    if len(kept) >= args.n_per_question:
                        break
                    try:
                        text = await _call(client, args, [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": _user_content(row, gt_show, args.max_pixels)},
                        ])
                    except Exception:
                        stats["teacher_error"] += 1
                        continue
                    if not text.strip():
                        stats["empty"] += 1
                        continue
                    if not _answer_ok(kind, text, gt):
                        stats["reject_answer"] += 1
                        continue
                    leak = _leaks(text)
                    if leak:
                        stats["reject_leak"] += 1
                        continue
                    if _refuses(text):
                        stats["reject_refusal"] += 1
                        continue
                    w = _think_words(text)
                    if w < 0:
                        stats["reject_no_think_block"] += 1
                        continue
                    if w < args.min_words or w > args.max_words:
                        stats["reject_length"] += 1
                        continue
                    if any(_too_similar(text, k, args.dup_threshold) for k in kept):
                        stats["reject_dup"] += 1
                        continue
                    kept.append(text)
            if not kept:
                stats["no_trace"] += 1
                return
            async with out_lock:
                for t in kept:
                    r = dict(row)
                    r["reference_response"] = t
                    fout.write(json.dumps(r) + "\n")
                    stats["kept"] += 1
                stats["questions_with_trace"] += 1
                n = stats["questions_with_trace"]
                if n % 100 == 0:
                    print(f"  {n} questions done, {stats['kept']} traces "
                          f"(rej ans={stats['reject_answer']} leak={stats['reject_leak']} "
                          f"len={stats['reject_length']} dup={stats['reject_dup']})", flush=True)

        await asyncio.gather(*(one(r) for r in rows))

    fout.close()
    print(f"\nwrote {stats['kept']} traces from {stats['questions_with_trace']}/{len(rows)} "
          f"questions -> {args.out}")
    print(f"stats: {dict(stats)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    # Teacher defaults to the locally served Qwen3.6-27B, NOT a GPT chat deployment:
    # those refuse to emit visible chain-of-thought and return a disclaimer plus a
    # short summary, which is unusable as an imitation target. Same choice the MIMIC
    # distillation made, for the same reason.
    ap.add_argument("--api_base", default=os.environ.get("API_BASE", "http://point.dd.works:18188/v1"))
    ap.add_argument("--api_key", default=os.environ.get("API_KEY", "EMPTY"))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--provider", default=os.environ.get("CHAT_PROVIDER", "vllm"))
    ap.add_argument("--n_per_question", type=int, default=2)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max_tokens", type=int, default=3500,
                    help="a thinking teacher spends most of this on reasoning")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--target_words", type=int, default=350)
    ap.add_argument("--min_words", type=int, default=80)
    ap.add_argument("--max_words", type=int, default=900)
    ap.add_argument("--dup_threshold", type=float, default=0.7)
    ap.add_argument("--max_pixels", type=int, default=1048576)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
