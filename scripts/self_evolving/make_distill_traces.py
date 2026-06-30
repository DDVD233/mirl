"""Generate distillation SFT annotations (reasoning trace + boxed answer).

The *static* distillation counterpart of `static_sft_dataset.py`: instead of an
assistant target that is just `\\boxed{ground_truth}`, each example carries a
full teacher-written reasoning trace ending in the boxed final answer, in the
canonical shape the SFT student is trained to imitate:

    <think>
    {clinical reasoning}
    </think>

    \\boxed{ICD-10 CODE: Diagnosis name}

Per training row (read directly from the REAL RLHF-format `train.jsonl`):
  1. The served model (teacher) is asked to SOLVE the case and write a visible,
     first-person diagnostic reasoning trace, given the confirmed ground-truth
     diagnosis *confidentially* as context — so it reasons to the correct answer.
  2. We REJECT any trace whose own boxed answer does not match the ground truth
     (ICD-code / synonym aware, mirroring the reward's accuracy gate). The final
     boxed answer we store is the EXACT ground truth, so every kept annotation is
     correct by construction.
  3. We REJECT traces that LEAK the fact that the answer was provided (the
     student never sees the ground truth, so the reasoning must read as a forward
     solve — no "the given/known answer", "we are told", "ground truth", etc.).
  4. We target ~1000 tokens of reasoning (soft length gate).
  5. We collect up to `--n_per_question` DISTINCT correct traces per question
     (answer diversity for the later RL stage), then SHUFFLE the whole file.

The teacher SEES the same chest-X-ray / 12-lead-ECG image(s) the student's row
carries (interleaved at the `<image>` placeholders) so the reasoning can cite
real imaging findings; unreadable images degrade to a text placeholder.

Output JSONL rows = the source row (prompt / images / reward_model / extra_info)
plus a `reference_response` field, consumable by `static_trace_sft_dataset.py`
(`StaticTraceSFTDataset`) — the static analog of `SelfEvolvingSFTDataset`.

Example (run on server5, model served locally at :18184):
    API_KEY=$(cat /scratch/sheng/self_evolving/.climb_teacher_key) \
    python scripts/self_evolving/make_distill_traces.py \
        --train_file /scratch/sheng/self_evolving/mimiciv_rare/train.jsonl \
        --out /scratch/sheng/self_evolving/mimiciv_rare/distill_sft_train.jsonl \
        --api_base http://localhost:18184/v1 --model_name Qwen/Qwen3.6-27B \
        --n_per_question 3 --concurrency 16
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
from difflib import SequenceMatcher
from pathlib import Path

import aiohttp

# --------------------------------------------------------------------------- #
# Teacher prompts
# --------------------------------------------------------------------------- #
DISTILL_TEACHER_SYSTEM = (
    "You are a senior physician writing an exemplary diagnostic reasoning trace that a "
    "student model will learn to imitate. You are given a patient's hospital admission and, "
    "CONFIDENTIALLY, the final confirmed primary diagnosis. Write FIRST-PERSON clinical "
    "reasoning that arrives at that diagnosis NATURALLY from the evidence — exactly as if you "
    "were solving the case yourself WITHOUT knowing the answer in advance.\n\n"
    "Rules:\n"
    "- Interpret the salient findings (demographics, vitals, labs, microbiology, procedures, "
    "medications, and any chest X-ray or 12-lead ECG shown). Build a focused differential, then "
    "justify why the correct diagnosis fits and why the leading alternatives do not.\n"
    "- NEVER reveal, restate, or hint that the diagnosis was provided to you. Do NOT use phrases "
    "like \"the given/provided/known/suggested/confirmed answer\", \"we are told\", \"as stated\", "
    "\"ground truth\", or \"thinking process that leads to\". Reason strictly FORWARD, as a "
    "clinician working the case up.\n"
    "- Be decisive: commit to the reasoning. Do not hedge, backtrack, or use words like \"wait\", "
    "\"actually\", \"on second thought\", or \"hmm\".\n"
    "- Length: write about 12-16 sentences (roughly 650-800 words) — thorough but focused, "
    "landing around 1000 tokens total. Do not enumerate every lab; weigh the MOST discriminating "
    "findings.\n"
    "- Do NOT write meta-commentary about the task, the student, or that you are producing a "
    "trace. Output only the clinical reasoning.\n"
    "- Do NOT mention or write any ICD-10 code anywhere in the reasoning, and never refer to "
    "\"the final diagnosis\", \"the code\", or \"the answer\" as something already determined — "
    "derive the disease yourself by name. The ICD-10 code appears ONLY inside the final \\boxed{} "
    "line.\n"
    "- You MUST FINISH with the boxed answer. END your response, on its own final line, with the "
    "answer wrapped in \\boxed{CODE: Diagnosis name} (for example "
    "\\boxed{E70.0: Classical phenylketonuria}), copying the confirmed diagnosis EXACTLY. The "
    "boxed answer is REQUIRED and must be the last thing you write — never omit it."
)


# --------------------------------------------------------------------------- #
# Answer extraction / correctness gate (mirrors generation_server.attach_teacher_trace)
# --------------------------------------------------------------------------- #
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ICD_CODE_RE = re.compile(r"([A-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?)", re.IGNORECASE)
# Stricter ICD detector for free prose: the code must be a STANDALONE token, so
# the "B40" inside a procedure code like "0PB40ZZ" (no boundary before B) is not
# flagged. Requires either a dotted subcode or an isolated 3-char code.
_ICD_IN_PROSE_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[A-Z][0-9][0-9AB](?![A-Za-z0-9.])"
)
_MIN_REASONING_CHARS = 200
_TRACE_FORMAT_RE = re.compile(
    r"\s*<think>\s*(?P<reasoning>.+?)\s*</think>\s*\\boxed\{[^{}]*\}\s*\Z", re.DOTALL
)

# Phrases that betray the answer was handed to the teacher (would teach the
# student a reasoning pattern it cannot reproduce at inference). Case-insensitive.
_LEAK_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"\bground[\s-]?truth\b",
    r"\b(we|you|i)\s*(?:'re|'m| am| are| was)?\s*(?:were\s+)?(told|given|provided|informed)\b",
    r"\bthe\s+(given|provided|known|correct|confirmed|stated|supplied|suggested|reference|"
    r"target|expected|gold|true|actual)\s+(answer|diagnosis|label|result|code)\b",
    # "...provided in the final diagnosis", "the ICD-10 code provided", "code given in..."
    r"\b(provided|given|supplied|specified|listed|furnished|handed|disclosed|revealed|"
    r"indicated|noted)\b[\s\w,'-]{0,30}\b(diagnosis|icd[\s-]?10(?:\s*code)?|code|answer|label)\b",
    r"\b(diagnosis|icd[\s-]?10(?:\s*code)?|code|answer|label)\b[\s\w,'-]{0,30}\b(provided|"
    r"given|supplied|specified|listed|furnished|handed|disclosed|revealed)\b",
    r"\b(as|since)\s+(stated|given|provided|noted|mentioned)\s+(above|in the\s+\w+)\b",
    r"\bthe\s+answer\s+(is|was)\s+(given|provided|already|known)\b",
    r"\b(thinking process|chain of thought)\s+that\s+leads\b",
    r"\bwork(?:ing)?\s+backward",
    r"\bbased on the\s+(provided|given)\s+(answer|diagnosis|label)\b",
    r"\bI\s+(?:already\s+)?know\s+the\s+(answer|diagnosis)\b",
    # backward-reasoning tells: "the final/correct diagnosis points/tells/confirms..."
    r"\bthe\s+(final|correct|confirmed|known|target|expected)\s+(diagnosis|answer|code)\b"
    r"[\s\w,'-]{0,20}\b(points?|tells?|indicat\w+|confirms?|suggests?|reveals?|is\s+autonomic)\b",
)]


def _extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text or "")
    if matches:
        ans = matches[-1]
    else:
        m = re.search(r"\\boxed\{(.+)\}", text or "", re.DOTALL)
        if not m:
            return None
        ans = m.group(1)
    ans = ans.strip()
    tm = re.match(r"\\(?:text|mathrm|mathbf)\{(.*)\}$", ans)
    if tm:
        ans = tm.group(1).strip()
    return ans


def _icd_code(text: str) -> str | None:
    m = _ICD_CODE_RE.search(text or "")
    return m.group(1).upper() if m else None


def _answer_matches(pred: str, gt: str) -> bool:
    """Free-form correctness gate: ICD code match when both carry one, else
    case-insensitive substring/equality (mirrors generation_server._answer_matches)."""
    gtl = (gt or "").strip().lower()
    predl = (pred or "").strip().lower()
    if not gtl or not predl:
        return False
    gc, pc = _icd_code(gt), _icd_code(pred)
    if gc is not None and pc is not None:
        # Accept parent/child subcodes (e.g. GT C83.39 vs teacher C83.3 — both
        # DLBCL); the stored answer is the exact GT, so this only gates whether
        # the reasoning landed on the right disease, not the final label.
        return gc == pc or gc.startswith(pc) or pc.startswith(gc)
    return predl == gtl or gtl in predl or predl in gtl


def _compose_trace(raw: str, boxed: str) -> str:
    """Normalize teacher output into <think>\\n{reasoning}\\n</think>\\n\\n\\boxed{ans}."""
    body = re.sub(r"</?think>", "", raw or "").strip()
    idx = body.rfind("\\boxed")
    reasoning = (body[:idx].strip() if idx != -1 else body).strip()
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{boxed}}}"


def _valid_trace_format(ref: str) -> bool:
    m = _TRACE_FORMAT_RE.fullmatch(ref or "")
    if not m:
        return False
    reasoning = m.group("reasoning").strip()
    if len(reasoning) < _MIN_REASONING_CHARS:
        return False
    if "<think>" in reasoning or "</think>" in reasoning:
        return False
    return True


def _semantic_leak(reasoning: str) -> bool:
    """True if the prose verbally reveals the answer was supplied ("the given
    answer", "we are told", "ground truth", ...). These can't be safely stripped
    (they're whole clauses), so a hit means reject."""
    return any(rx.search(reasoning or "") for rx in _LEAK_RES)


_ICD_BARE = r"[A-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?"
# Strip stray diagnosis-style ICD codes the teacher sometimes drops into the
# prose (e.g. "consistent with DLBCL (C83.39)", "ICD-10: C22.1", "C22.1:"). The
# code belongs only in the final \boxed{} line; removing it salvages an
# otherwise-clean trace instead of discarding it. Order matters: parenthetical
# and "ICD-10:"-prefixed forms first, then bare standalone tokens.
_STRIP_RES = [
    # parenthetical / bracketed code, optionally "ICD-10:"-prefixed
    (re.compile(r"\s*[\(\[]\s*(?:ICD[\s-]?10(?:\s*code)?[:\s]*)?" + _ICD_BARE + r"\s*[\)\]]", re.I), ""),
    # "ICD-10: C22.1" / "ICD-10 code C22.1" inline -> drop the whole phrase
    (re.compile(r"[,;]?\s*\bICD[\s-]?10(?:\s*code)?[:\s]+" + _ICD_BARE, re.I), ""),
    # "C92.00:" / "C22.1 -" (code immediately introducing a name) -> drop both
    (re.compile(r"(?<![A-Za-z0-9])" + _ICD_BARE + r"\s*[:\-]\s*"), ""),
    # bare standalone code token
    (re.compile(r"(?<![A-Za-z0-9])" + _ICD_BARE + r"(?![A-Za-z0-9.])"), ""),
]


def _strip_icd_prose(reasoning: str) -> str:
    s = reasoning or ""
    for rx, repl in _STRIP_RES:
        s = rx.sub(repl, s)
    s = re.sub(r"[\(\[]\s*[\)\]]", "", s)        # empty brackets left behind
    s = re.sub(r"\s+([,.;:)])", r"\1", s)         # space before punctuation
    s = re.sub(r"([(,])\s*[,;]+", r"\1", s)       # collapse orphaned ", ," / "(,"
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _est_tokens(text: str) -> int:
    """Cheap token estimate (no tokenizer dependency). Calibrated against the
    served Qwen3.6 tokenizer on these clinical traces (~3.6 chars/token; medical
    text + ICD codes run denser than the usual 4 chars/token). Good enough for a
    soft ~1000-token length gate."""
    return int(round(len(text or "") / 3.6))


def _reasoning_of(trace: str) -> str:
    m = _TRACE_FORMAT_RE.fullmatch(trace or "")
    return m.group("reasoning").strip() if m else (trace or "")


def _too_similar(reasoning: str, accepted: list[str], threshold: float) -> bool:
    """Diversity gate: reject a near-duplicate of an already-kept reasoning."""
    a = re.sub(r"\s+", " ", reasoning.lower()).strip()
    for prev in accepted:
        b = re.sub(r"\s+", " ", prev.lower()).strip()
        if SequenceMatcher(None, a[:1200], b[:1200]).ratio() >= threshold:
            return True
    return False


# --------------------------------------------------------------------------- #
# Multimodal: show the teacher the same image(s) the student sees
# --------------------------------------------------------------------------- #
def _image_to_data_uri(img, max_pixels: int | None) -> str | None:
    path = img.get("image") if isinstance(img, dict) else img
    mp = img.get("max_pixels") if isinstance(img, dict) else None
    max_pixels = mp or max_pixels
    if not isinstance(path, str) or not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None
    if max_pixels:
        try:
            from PIL import Image

            im = Image.open(io.BytesIO(data)).convert("RGB")
            if im.width * im.height > max_pixels:
                scale = (max_pixels / float(im.width * im.height)) ** 0.5
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if c.get("type") == "text")
    return ""


async def _build_user_content(user_text: str, gt: str, images: list, no_images: bool,
                              max_pixels: int | None):
    """Teacher user message: case text (images interleaved at <image>) + the
    confidential ground-truth footer.

    Image encoding (PIL + file/NFS read) runs via asyncio.to_thread so it never
    blocks the event loop — a single slow image read would otherwise stall every
    concurrent worker (mirrors generation_server._build_teacher_content)."""
    footer = (
        "\n\n--- CONFIDENTIAL (for your reasoning ONLY; the student will NOT see this and you "
        "must NEVER reveal or allude to having been given it) ---\n"
        f"Final confirmed primary diagnosis: {gt}\n"
        "--- end confidential ---\n\n"
        "Now write the forward diagnostic reasoning and end with the required \\boxed{...} answer."
    )
    if no_images or not images:
        return user_text + footer

    parts = re.split(r"(<image>)", user_text)
    content: list = []
    img_idx = 0
    for p in parts:
        if p == "<image>":
            uri = (await asyncio.to_thread(_image_to_data_uri, images[img_idx], max_pixels)
                   if img_idx < len(images) else None)
            content.append(
                {"type": "image_url", "image_url": {"url": uri}} if uri
                else {"type": "text", "text": "<image>"}
            )
            img_idx += 1
        elif p:
            content.append({"type": "text", "text": p})
    while img_idx < len(images):
        uri = await asyncio.to_thread(_image_to_data_uri, images[img_idx], max_pixels)
        if uri:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        img_idx += 1
    content.append({"type": "text", "text": footer})
    return content


# --------------------------------------------------------------------------- #
# Teacher call
# --------------------------------------------------------------------------- #
async def _teacher_solve(session, args, user_content, n: int) -> list[str]:
    """One chat call returning up to `n` candidate reasoning traces (raw text)."""
    payload = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": DISTILL_TEACHER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": n,
        # Direct (non-thinking) decoding: Qwen3.6's thinking mode runs to
        # ~2500-3000 tokens on these dense cases and frequently never reaches the
        # box. Non-thinking + the length rule lands a clean, reliably-boxed trace
        # near the ~1000-token target. We wrap the prose in <think>…</think>
        # ourselves (_compose_trace) for the student-facing format.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("GEN_CHAT_TIMEOUT", "1800")))
    last_err = None
    for attempt in range(4):
        try:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                data = await r.json()
            out = []
            for ch in data.get("choices", []):
                msg = ch.get("message", {}) or {}
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                if reasoning and "<think>" not in content:
                    content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                if content:
                    out.append(content)
            if not out and os.environ.get("DISTILL_DEBUG"):
                fr = [c.get("finish_reason") for c in data.get("choices", [])]
                print(f"[teacher] empty out: status={r.status} nchoices={len(data.get('choices', []))} "
                      f"finish={fr} keys={list((data.get('choices') or [{}])[0].get('message', {}).keys())}",
                      flush=True)
            return out
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_err = e
            if os.environ.get("DISTILL_DEBUG"):
                print(f"[teacher] attempt {attempt} err: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if attempt == 3:
                break
            await asyncio.sleep(2 ** attempt)
    if os.environ.get("DISTILL_DEBUG"):
        print(f"[teacher] call failed: {type(last_err).__name__}: {last_err}")
    return []


# --------------------------------------------------------------------------- #
# Per-question worker: collect up to n_per DISTINCT correct traces
# --------------------------------------------------------------------------- #
async def _make_traces_for_row(session, args, row, stats) -> list[dict]:
    gt = (row.get("reward_model") or {}).get("ground_truth", "")
    if not gt or not str(gt).strip():
        stats["no_gt"] += 1
        return []
    gt = str(gt).strip()

    user_text = None
    for m in row.get("prompt", []):
        if m.get("role") == "user":
            user_text = _text_from_content(m.get("content"))
    if not user_text:
        stats["no_user"] += 1
        return []

    images = row.get("images") or []
    user_content = await _build_user_content(user_text, gt, images, args.no_images, args.max_pixels)

    accepted_reasonings: list[str] = []
    accepted_traces: list[str] = []
    for _ in range(args.max_rounds):
        if len(accepted_traces) >= args.n_per_question:
            break
        cands = await _teacher_solve(session, args, user_content, args.oversample)
        for raw in cands:
            boxed = _extract_boxed(raw)
            if boxed is None:
                stats["no_box"] += 1
                continue
            if not _answer_matches(boxed, gt):
                stats["wrong"] += 1
                continue
            # Store the EXACT ground truth as the final answer so every kept
            # annotation is correct by construction (teacher only had to reason
            # to the right code/synonym, not reproduce the label verbatim).
            trace = _compose_trace(raw, gt)
            reasoning = _reasoning_of(trace)
            if not _valid_trace_format(trace):
                stats["bad_format"] += 1
                continue
            # Verbal "the answer was given" leaks are unrecoverable -> reject.
            if _semantic_leak(reasoning):
                stats["leak"] += 1
                continue
            # Stray ICD code(s) in the prose are recoverable: strip them and keep
            # the trace (the code lives only in the box). Reject only if a code
            # still survives the strip (rare / malformed).
            cleaned = _strip_icd_prose(reasoning)
            if _ICD_IN_PROSE_RE.search(cleaned):
                stats["leak"] += 1
                continue
            if cleaned != reasoning:
                stats["stripped"] += 1
                reasoning = cleaned
                trace = f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{gt}}}"
                if not _valid_trace_format(trace):
                    stats["bad_format"] += 1
                    continue
            ntok = _est_tokens(trace)
            if ntok < args.min_trace_tokens or ntok > args.max_trace_tokens:
                stats["bad_len"] += 1
                continue
            if _too_similar(reasoning, accepted_reasonings, args.dup_threshold):
                stats["dup"] += 1
                continue
            accepted_reasonings.append(reasoning)
            accepted_traces.append(trace)
            if len(accepted_traces) >= args.n_per_question:
                break

    out = []
    for trace in accepted_traces:
        r = dict(row)
        r["reference_response"] = trace
        ei = dict(r.get("extra_info") or {})
        ei["teacher_answer"] = gt
        ei["distill_teacher"] = args.model_name
        r["extra_info"] = ei
        out.append(r)
    if not out:
        stats["rows_empty"] += 1
    elif len(out) < args.n_per_question:
        stats["rows_partial"] += 1
    else:
        stats["rows_full"] += 1
    stats["traces"] += len(out)
    return out


# --------------------------------------------------------------------------- #
def _question_key(row: dict) -> str:
    """Stable id for a source question (the user-turn text)."""
    for m in row.get("prompt", []):
        if m.get("role") == "user":
            return _text_from_content(m.get("content"))[:4000]
    return json.dumps(row.get("prompt", []))[:4000]


def _load_rows(path: str, max_samples: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return rows


async def main_async(args):
    rows = _load_rows(args.train_file, args.max_samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Crash-safe staging: every accepted trace is appended (and flushed) to
    # <out>.partial as it is produced, so a long run survives an interruption.
    # The final shuffled file is written from .partial at the end. On restart we
    # skip questions already present in .partial (resume).
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    done_keys: set[str] = set()
    if partial_path.exists():
        with partial_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_keys.add(_question_key(json.loads(line)))
                    except Exception:
                        pass
    todo = [r for r in rows if _question_key(r) not in done_keys]
    print(f"[distill] {len(rows)} source questions from {args.train_file}; "
          f"{len(done_keys)} already done in {partial_path.name}, {len(todo)} to do",
          flush=True)

    stats = {k: 0 for k in (
        "no_gt", "no_user", "no_box", "wrong", "bad_format", "leak", "bad_len",
        "dup", "stripped", "rows_empty", "rows_partial", "rows_full", "traces")}
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    done = 0

    pf = partial_path.open("a")
    try:
        async with aiohttp.ClientSession() as session:
            async def worker(row):
                nonlocal done
                async with sem:
                    try:
                        res = await _make_traces_for_row(session, args, row, stats)
                    except Exception as e:
                        print(f"[distill] row error: {type(e).__name__}: {e}", flush=True)
                        res = []
                if res:
                    async with write_lock:
                        for r in res:
                            pf.write(json.dumps(r, ensure_ascii=False) + "\n")
                        pf.flush()
                done += 1
                if done % 50 == 0:
                    print(f"[distill] {done}/{len(todo)} questions  "
                          f"traces={stats['traces']}  full={stats['rows_full']} "
                          f"partial={stats['rows_partial']} empty={stats['rows_empty']}  "
                          f"(wrong={stats['wrong']} leak={stats['leak']} dup={stats['dup']} "
                          f"nobox={stats['no_box']} badlen={stats['bad_len']} "
                          f"stripped={stats['stripped']})", flush=True)
                return res

            await asyncio.gather(*[worker(r) for r in todo])
    finally:
        pf.close()

    # Reload the full partial set (this run + any resumed rows) and shuffle —
    # answer diversity is interleaved, not blocked by question — deterministically.
    all_out = _load_rows(str(partial_path), -1)
    random.Random(args.seed).shuffle(all_out)
    with out_path.open("w") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n[distill] DONE", flush=True)
    print(f"  source questions       : {len(rows)}")
    print(f"  annotations written    : {len(all_out)}  -> {out_path}")
    print(f"  questions with 3 traces: {stats['rows_full']}")
    print(f"  questions with 1-2     : {stats['rows_partial']}")
    print(f"  questions with 0       : {stats['rows_empty']}")
    print(f"  rejects: wrong={stats['wrong']} leak={stats['leak']} dup={stats['dup']} "
          f"no_box={stats['no_box']} bad_format={stats['bad_format']} bad_len={stats['bad_len']}")
    print(f"  (staging file {partial_path.name} kept; safe to delete after verifying {out_path.name})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_file", required=True, help="RLHF-format source train.jsonl.")
    p.add_argument("--out", required=True, help="Output distillation annotation JSONL.")
    p.add_argument("--api_base", default=os.environ.get("API_BASE", "http://localhost:18184/v1"))
    p.add_argument("--api_key", default=os.environ.get("API_KEY", "EMPTY"))
    p.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-27B"))
    p.add_argument("--n_per_question", type=int, default=3,
                   help="Distinct correct traces to collect per question.")
    p.add_argument("--oversample", type=int, default=4,
                   help="Candidates sampled per teacher call (vLLM n).")
    p.add_argument("--max_rounds", type=int, default=3,
                   help="Teacher calls per question before giving up topping up to n_per.")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=-1, help="Cap source questions (debug).")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature; higher = more diverse traces per question.")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=1700, help="Teacher generation cap.")
    p.add_argument("--min_trace_tokens", type=int, default=400)
    p.add_argument("--max_trace_tokens", type=int, default=1400)
    p.add_argument("--dup_threshold", type=float, default=0.7,
                   help="Reasoning similarity >= this is rejected as a duplicate.")
    p.add_argument("--no_images", action="store_true", help="Do not show images to the teacher.")
    p.add_argument("--max_pixels", type=int, default=65536, help="Per-image downscale cap.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
