"""
Reward function for self-evolving medical agent training.

Composite weights live in the *_WEIGHT constants below; see `compute_score` for
the full formula. Smooth surrogates `embed_sim` (cosine similarity from the
generation server's retrieval embedding endpoint) and `char_bleu` keep the
gradient above the noise floor when the discrete signals (accuracy / format /
judge) collapse to 0.

In no-label mode, `answer_quality` is derived from `judge_correctness` (5 if
judged correct, 1 if incorrect).
"""

import asyncio
import logging
import os
import random
import re

import aiohttp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


JUDGE_REASONING_PROMPT = """\
You are a medical reasoning evaluator. You are given a question, the model's full response \
(including its reasoning), and the correct answer. Rate the quality of the model's \
REASONING (not the final answer) on a scale from 1 to 5:

1 - No reasoning, or completely irrelevant reasoning
2 - Minimal reasoning with major logical errors
3 - Some relevant reasoning but with gaps or minor errors
4 - Good reasoning that mostly follows from the evidence
5 - Excellent reasoning that is thorough, evidence-based, and logically sound

Directly output your rating inside \\boxed{...}. Do not write any reasoning, explanation, \
or text outside the boxed answer. Example: \\boxed{4}"""


JUDGE_ANSWER_QUALITY_PROMPT = """\
You are a medical answer grader. You are given:
- A question (possibly multiple choice or free response)
- The correct ground-truth answer
- The model's extracted final answer (this is ONLY the boxed answer, not the reasoning)

Your job: rate how well the model's extracted answer ALIGNS with the ground truth on a \
scale 1 to 5. Focus ONLY on the alignment between the two answers — ignore reasoning.

Scale:
1 - Completely wrong / no answer / unrelated
2 - Mostly wrong but has a small overlap with correct answer
3 - Partially correct; captures some but misses key parts, or is overly vague
4 - Essentially correct, minor wording differences or missing nuance
5 - Fully correct and aligned with the ground truth

For MCQ: if the letter matches → 5; if different letter but equivalent content → 4; else 1.
For free response: judge semantic alignment (synonyms, paraphrases count as correct).

Directly output your rating inside \\boxed{...}. Do not write any reasoning, explanation, \
or text outside the boxed answer. Example: \\boxed{5}"""


JUDGE_CORRECTNESS_PROMPT = """\
You are a medical expert evaluating whether a model's answer to a medical question is \
correct. Use the provided question/context and your own medical knowledge to determine \
if the model's extracted answer is correct.

Directly output your verdict inside \\boxed{...} as either \\boxed{correct} or \
\\boxed{incorrect}. Do not write any reasoning, explanation, or text outside the boxed \
answer."""


JUDGE_ACCURACY_LENIENT_PROMPT = """\
You are a medical answer grader. You are given:
- A medical question
- The correct ground-truth answer (typically a disease, often with an ICD code)
- The model's extracted final answer

Decide whether the model's answer identifies the SAME disease as the ground truth. \
Grade GENEROUSLY — the DISEASE NAME is what matters, NOT the exact ICD code or the \
level of coding detail. The ICD code is secondary; if the named disease is the same, \
the answer is CORRECT.

Mark CORRECT (accept) when:
- The disease NAME is the same, even if the ICD code differs \
  (e.g. GT "I82.0: Budd-Chiari syndrome" vs answer "I82.1: Budd-Chiari syndrome" -> CORRECT; \
  GT "B60.0: Babesiosis" vs "B62.0: Babesiosis" -> CORRECT; \
  GT "L73.2: Hidradenitis suppurativa" vs "L73.3: Hidradenitis suppurativa" -> CORRECT)
- Synonyms / abbreviations (e.g. "MI" = "myocardial infarction" = "heart attack"; \
  "factor VIII deficiency" = "hemophilia A")
- ICD code with vs. without descriptive text (e.g. "C22.0" vs "C22.0: Liver cell carcinoma")
- The answer differs only in subtype, site, severity, or remission/relapse status but \
  names the SAME core disease \
  (e.g. GT "C90.00: Multiple myeloma not having achieved remission" vs "Multiple myeloma" \
  -> CORRECT; GT "C92.40: APL not in remission" vs "C92.0: Acute promyelocytic leukemia" -> CORRECT)
- A more general or more specific form when the core diagnosis is right \
  (e.g. "pneumonia" when GT is "bacterial pneumonia")
- The correct disease is clearly stated as the primary/final diagnosis even if a short \
  differential is mentioned
- Misspellings or differences in capitalization / abbreviation

Mark INCORRECT (reject) ONLY when:
- A genuinely DIFFERENT disease (different name, not a synonym/subtype) \
  (e.g. "neutropenia" vs GT "multiple myeloma"; "epilepsy" vs GT "encephalocele")
- A clearly different entity in the same family that is a distinct diagnosis \
  (e.g. "atrial flutter" when GT is "atrial fibrillation"; \
  "Hodgkin lymphoma" when GT is "non-Hodgkin lymphoma")
- Generic non-answers or template placeholders \
  ("unknown", "no diagnosis", "see above", "ICD-10 CODE: Diagnosis name")
- Empty or missing answer

When in doubt and the disease name plausibly refers to the same condition, prefer CORRECT.

Directly output your verdict inside \\boxed{...} as either \\boxed{correct} or \
\\boxed{incorrect}. Do not write any reasoning, explanation, or text outside the boxed \
answer."""


JUDGE_ACCURACY_STRICT_PROMPT = """\
You are a medical answer grader applying a STRICT rubric. You are given:
- A medical question
- The correct ground-truth answer (typically a disease, often with an ICD code)
- The model's extracted final answer

Decide whether the model's answer is EQUIVALENT to the ground truth at the SAME \
LEVEL OF SPECIFICITY. Be strict.

ACCEPT only when the answers refer to exactly the same diagnosis:
- Synonyms or abbreviations of the same disease at the same specificity \
  (e.g. "MI" = "myocardial infarction"; "AML" = "acute myeloid leukemia")
- ICD code with or without its descriptive text (e.g. "C22.0" vs. \
  "C22.0: Liver cell carcinoma")
- Trivial spelling / capitalization differences

REJECT:
- A more general or more specific subtype than the GT \
  (e.g. "pneumonia" when GT is "bacterial pneumonia"; \
  "leukemia" when GT is "acute myeloblastic leukemia in remission")
- A different ICD code, even within the same family \
  (e.g. C22.0 vs C22.1; I48.0 vs I48.91)
- A different disease — even if clinically related \
  (e.g. "atrial flutter" when GT is "atrial fibrillation")
- Generic non-answers ("unknown", "no diagnosis", "see above")
- Empty or missing answer

Directly output your verdict inside \\boxed{...} as either \\boxed{correct} or \
\\boxed{incorrect}. Do not write any reasoning, explanation, or text outside the boxed \
answer."""


# Reward component weights (sum to 1.0).
# accuracy = strict normalized exact match; judge_acc_lenient and
# judge_acc_strict = LLM-judged disease match at two strictness levels (both
# reported so we can decide later which to keep in the final composite).
# Both fire when ground truth is available; the LLM judges give partial credit
# the strict-string match misses (a real positive on rare ICD codes is still
# rare with strict match).
# embed_sim and char_bleu are smooth surrogates that fire even when the
# discrete signals collapse to 0; they keep reward shaping above the noise floor.
# Re-weighted (2026-06-29) to sharpen the gradient toward CORRECTNESS after a flat-training
# diagnosis: embed_sim/char_bleu sat ~constant (0.66/0.19) regardless of policy quality and
# diluted the signal, so they are cut to near-zero; accuracy / lenient / answer_quality are
# raised; FORMAT is raised to 0.20 to restore boxing/commitment pressure (only ~48% of outputs
# were emitting \boxed, flooring exact-match acc). Weights sum to 1.0.
ACCURACY_WEIGHT = 0.20
JUDGE_ACCURACY_LENIENT_WEIGHT = 0.10
JUDGE_ACCURACY_STRICT_WEIGHT = 0.05
REASONING_WEIGHT = 0.15
ANSWER_QUALITY_WEIGHT = 0.25
FORMAT_WEIGHT = 0.20
EMBED_SIM_WEIGHT = 0.05
CHAR_BLEU_WEIGHT = 0.00

DEBUG_PRINT_PROB = 0.01


def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer from \\boxed{...} (last occurrence)."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


# Fallback answer cues for responses that state a final diagnosis WITHOUT \boxed{}.
# In practice ~half of the model's responses finish with a clear diagnosis line
# ("Final Diagnosis: ...", "ICD-10: ...") but no \boxed, which the boxed-only
# extractor dropped to an empty answer -> auto-zero (the judge never ran). We
# keep format_ok tied to \boxed (training pressure toward clean format) but let
# the accuracy/judge path recover these answers so they are actually graded.
_ANSWER_CUE_RE = re.compile(
    r"(?:final\s+diagnosis|primary\s+diagnosis|most\s+likely\s+diagnosis|"
    r"final\s+answer|diagnosis|icd[-\s]?10(?:\s*code)?|conclusion|answer)"
    r"\s*(?:is|:|=|-)\s*",
    re.IGNORECASE,
)


def _clean_answer(s: str) -> str:
    """Strip markdown / boilerplate from a captured answer span (first line only)."""
    s = s.strip().splitlines()[0] if s.strip() else ""
    s = s.replace("**", "").replace("`", "").strip()
    # drop a trailing sentence after the diagnosis (keep code + name, cut prose)
    s = re.split(r"\s+(?:because|since|as the|which|given)\b", s, maxsplit=1)[0]
    return s.strip(" .*:-\t")


def extract_final_answer(text: str) -> str | None:
    """Best-effort final-answer extraction.

    Prefer \\boxed{...}; otherwise fall back to the LAST explicit diagnosis cue
    ("Final Diagnosis: ...", "ICD-10: ...", etc.). Returns None if nothing found.
    """
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    last = None
    for m in _ANSWER_CUE_RE.finditer(text):
        span = text[m.end():]
        cleaned = _clean_answer(span)
        if cleaned and len(cleaned) >= 2:
            last = cleaned
    if last:
        return last
    # Last resort: no \boxed and no explicit diagnosis cue (e.g. the model ran out
    # of tokens mid-reasoning). Feed the tail of the response so the judge can still
    # see the disease the model was converging on, instead of an auto-zero. This is
    # EVAL-ONLY (gated on REWARD_EVAL_LENIENT_ONLY): in the training reward it would
    # be a reward-hacking surface (the model could ramble instead of committing to a
    # boxed/cued diagnosis and still earn partial credit), so training stops at cues.
    if os.environ.get("REWARD_EVAL_LENIENT_ONLY", "0") == "1":
        tail = re.sub(r"\s+", " ", text.replace("**", "").replace("`", "")).strip()
        tail = tail[-200:].strip()
        return tail if len(tail) >= 2 else None
    return None


def check_format(text: str) -> bool:
    return bool(re.search(r"\\boxed\{[^}]*\}", text))


def check_accuracy(solution_str: str, ground_truth: str) -> tuple[bool, str | None]:
    """Normalized string match between extracted final answer and ground_truth."""
    extracted = extract_final_answer(solution_str)
    if extracted is None:
        return False, None
    gt = ground_truth.strip().lower()
    pred = extracted.strip().lower()
    # For MCQ, pick first letter
    if len(gt) == 1 and gt in "abcd":
        pred_letter = re.sub(r"[^a-d]", "", pred)[:1]
        return pred_letter == gt, extracted
    return pred == gt, extracted


# Global concurrency cap on judge API calls, per worker process. Without it the async
# reward loop `asyncio.gather`s over the whole batch AND over each rubric's criteria, so
# validation fires ~(#samples x #criteria) judge calls at once and bursts past TRAPI's
# global rate limit (~2000 req/60s shared) -> 429s / long backoff / the hang we hit.
# The semaphore throttles concurrent calls (spreads them over time). Keyed by event loop
# so it stays valid across a worker's loop lifecycle. Tune via REWARD_JUDGE_CONCURRENCY.
_JUDGE_SEMS: dict = {}


def _judge_sem() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _JUDGE_SEMS.get(loop)
    if sem is None:
        # Default 6 (NOT read only from driver env — Ray actors don't inherit it) so the
        # cap holds per reward worker regardless: 6 x 8 workers = 48 concurrent judge calls.
        n = int(os.environ.get("REWARD_JUDGE_CONCURRENCY", "6"))
        sem = asyncio.Semaphore(n)
        _JUDGE_SEMS[loop] = sem
    return sem


async def _call_api(
    api_base: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 256,
    provider: str = "",
    timeout_s: float | None = None,
) -> str:
    """Call the chat API with thinking disabled.

    Judges only need a short verdict (boxed letter / rating / correct vs.
    incorrect). Letting the model think first burns thousands of tokens
    per call and serializes the chat server behind the gen_server +
    rollout traffic — too slow to be practical. We pass
    ``chat_template_kwargs.enable_thinking=False`` so the Qwen3 chat
    template skips the reasoning block entirely; the model emits the
    boxed answer directly.
    """
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Provider-specific extensions for disabling thinking. The OpenAI HTTP
    # shape is identical across providers; only the "skip reasoning" knob
    # differs:
    #   vllm  : chat_template_kwargs.enable_thinking=False
    #   kimi  : thinking={"type": "disabled"} (and you MUST omit temperature
    #           — Kimi non-thinking mode fixes it to 0.6 and 400s on any
    #           other value)
    #   other : drop both; let the provider's own default handle it.
    # `provider` arg (when given) overrides the global env so a single process
    # can mix a vllm self-judge (training) and a TRAPI gpt-chat-latest judge
    # (validation) — they need different request shaping.
    provider = (provider or os.environ.get("CHAT_PROVIDER", "vllm")).lower()
    payload: dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    if provider == "vllm":
        payload["temperature"] = 0.0
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif provider == "kimi":
        payload["thinking"] = {"type": "disabled"}
    elif provider == "trapi":
        # TRAPI Azure-OpenAI models (Kimi, gpt-5.x, ...): no temperature; want
        # `max_completion_tokens` (gpt-5.x 400 on `max_tokens`). gpt-5.x reasoning
        # models take reasoning_effort="none" to skip reasoning, but the high-throughput
        # gpt-chat-latest deployment REJECTS "none" (only "medium"), so omit it there.
        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = max_tokens
        # Reasoning models (gpt-5.x, o-series) take reasoning_effort="none" to skip
        # reasoning. CHAT deployments (gpt-chat-latest, gpt-5.3-chat, ...) reject it
        # (only accept their default), so omit for any "chat" model.
        if "chat" not in model_name.lower():
            payload["reasoning_effort"] = "none"
    else:  # openai-compatible / generic
        payload["temperature"] = 0.0

    # Each attempt gets its own 300s budget; up to 4 attempts with
    # exponential backoff (1s, 2s, 4s) — worst case ~21 min before we
    # fall back to the 0.0 default. The chat server is shared with the
    # gen_server proposer/generator/validator (which use thinking and
    # can run for minutes per request), so individual judge requests
    # occasionally sit in the queue long past the model's own latency.
    # `timeout_s` overrides that budget per caller. The specification-gap referee
    # runs in the trainer DRIVER and blocks the step, so it needs a much tighter
    # budget than a reward worker's grading call — and REWARD_JUDGE_TIMEOUT is
    # shared with those workers, so it cannot be lowered globally.
    timeout_total = (float(timeout_s) if timeout_s
                     else float(os.environ.get("REWARD_JUDGE_TIMEOUT", "300")))
    timeout = aiohttp.ClientTimeout(total=timeout_total)
    last_err: Exception | None = None
    # Throttle concurrent judge calls (rate-limit safety) — only the network wait is
    # held under the semaphore; retries/backoff happen inside so a slow call doesn't
    # permanently occupy a slot beyond its attempts.
    async with _judge_sem():
        for attempt in range(4):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        # TRAPI (and other Azure-OpenAI proxies) return auth / quota /
                        # content-filter failures as HTTP 200 with an {"error": ...} body,
                        # which slips past raise_for_status(). Detect it explicitly so it
                        # is retried + logged, not turned into a silent KeyError.
                        if not isinstance(data, dict) or "choices" not in data:
                            err = data.get("error", data) if isinstance(data, dict) else data
                            raise RuntimeError(f"judge API error response: {str(err)[:300]}")
                        msg = data["choices"][0]["message"]
                        # With thinking disabled the answer is in `content`; we
                        # still fall back to `reasoning_content` defensively in
                        # case the server ignored the flag.
                        content = msg.get("content") or msg.get("reasoning_content") or ""
                        return content.strip()
            except Exception as e:  # broadened: NEVER swallow — every failure is logged
                last_err = e
                # Log every failed attempt (status/type) so rate-limits (429) / auth /
                # timeouts / error-body responses on the judge endpoint are visible.
                status = getattr(e, "status", None)
                logger.warning(
                    "judge call attempt %d/4 failed (provider=%s, model=%s, base=%s): %s%s",
                    attempt + 1, provider, model_name, api_base,
                    f"HTTP {status} " if status is not None else "", repr(e),
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
    raise RuntimeError(f"unreachable, last_err={last_err}")


def _extract_judge_verdict(text: str) -> str:
    """Pull 'correct'/'incorrect' from a judge response.

    Prefer the boxed answer (the format the prompt asks for); fall back to
    a substring match on the raw response. Returns "" if neither matches —
    callers should treat that as a judge failure rather than a verdict.
    """
    boxed = extract_boxed_answer(text)
    candidate = (boxed or text or "").lower()
    if "incorrect" in candidate:
        return "incorrect"
    if "correct" in candidate:
        return "correct"
    return ""


def _extract_judge_rating(text: str, default: float) -> float:
    """Pull an integer 1-5 from a judge response.

    Prefer the boxed answer; fall back to the first 1-5 digit in the raw
    text. Returns ``default`` if nothing usable is found.
    """
    boxed = extract_boxed_answer(text)
    for source in (boxed, text):
        if not source:
            continue
        m = re.search(r"[1-5]", source)
        if m:
            return float(m.group())
    return default


async def judge_reasoning(
    api_base: str, api_key: str, model_name: str,
    question: str, context: str, response: str, ground_truth: str,
) -> float:
    """Rate reasoning quality 1-5."""
    gt_line = f"Correct answer: {ground_truth}\n" if ground_truth else ""
    ctx_line = f"Context: {context[:1500]}\n" if context else ""
    user_prompt = (
        f"Question: {question}\n"
        f"{ctx_line}"
        f"{gt_line}"
        f"Model's full response:\n{response[:3000]}\n\n"
        "Rate the reasoning quality (1-5):"
    )
    try:
        content = await _call_api(
            api_base, api_key, model_name, JUDGE_REASONING_PROMPT, user_prompt, max_tokens=64
        )
        return _extract_judge_rating(content, default=3.0)
    except Exception as e:
        logger.warning(f"judge_reasoning failed: {type(e).__name__}: {e!r}")
        return 3.0


async def judge_answer_quality(
    api_base: str, api_key: str, model_name: str,
    question: str, ground_truth: str, extracted_answer: str, options: dict = None,
) -> float:
    """Rate alignment between ground truth and model's extracted boxed answer 1-5."""
    if not extracted_answer:
        return 1.0
    options_line = ""
    if options:
        options_line = "Options: " + ", ".join(f"{k}. {v}" for k, v in options.items()) + "\n"
    user_prompt = (
        f"Question: {question}\n"
        f"{options_line}"
        f"Ground truth answer: {ground_truth}\n"
        f"Model's extracted answer: {extracted_answer}\n\n"
        "Rate the alignment (1-5):"
    )
    try:
        content = await _call_api(
            api_base, api_key, model_name, JUDGE_ANSWER_QUALITY_PROMPT, user_prompt, max_tokens=64
        )
        return _extract_judge_rating(content, default=1.0)
    except Exception as e:
        logger.warning(f"judge_answer_quality failed: {type(e).__name__}: {e!r}")
        return 1.0


async def _report_to_gen_server(
    gen_server_url: str, question_id: str, accuracy: float,
) -> None:
    """POST per-question accuracy to the generation server's /report endpoint.

    Called after compute_score so the server can update its sliding accuracy
    window (used to calibrate next-question difficulty) and the per-id log.
    Failures are logged and swallowed — reward scoring must not depend on
    the gen-server being reachable.
    """
    if not gen_server_url or not question_id:
        return
    url = f"{gen_server_url.rstrip('/')}/report"
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url, json={"question_id": question_id, "accuracy": float(accuracy)}
            ) as resp:
                resp.raise_for_status()
    except Exception as e:
        logger.warning(f"report to gen server failed: {e}")


async def embedding_similarity(
    embed_api_base: str,
    embed_api_key: str,
    embed_model: str,
    prediction: str,
    reference: str,
) -> float:
    """Cosine similarity between prediction and reference via an OpenAI-compatible
    embeddings endpoint (the same one used by the generation server's retriever).

    Both strings are sent in a single ``/embeddings`` call (``input=[pred, ref]``)
    and the cosine is computed locally. Both inputs are expected to be short
    (extracted boxed answers + ground truths) — we truncate to 2000 chars to match
    the gen_server's _embed_text convention. Returns ``[0, 1]``; 0 on failure.
    """
    if not embed_api_base or not embed_model or not prediction or not reference:
        return 0.0
    url = f"{embed_api_base.rstrip('/')}/embeddings"
    payload = {"model": embed_model, "input": [prediction[:2000], reference[:2000]]}
    headers = {"Authorization": f"Bearer {embed_api_key or 'EMPTY'}"}
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
        a = data["data"][0]["embedding"]
        b = data["data"][1]["embedding"]
        # Pure-Python cosine; embeddings are ~2k dims, this is fast enough and
        # avoids forcing numpy on every reward worker.
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b, strict=True):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        sim = dot / ((na**0.5) * (nb**0.5))
        return max(0.0, min(1.0, sim))
    except Exception as e:
        logger.warning(f"embedding_similarity failed: {type(e).__name__}: {e}")
        return 0.0


def char_bleu(prediction: str, reference: str) -> float:
    """Character-level BLEU-4 with smoothing, returns a value in [0, 1].

    Robust to short strings via NLTK's smoothing method 1, which dampens
    the harsh zero-precision behaviour BLEU otherwise has on tiny outputs.
    """
    if not prediction or not reference:
        return 0.0
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except Exception as e:
        logger.warning(f"char_bleu: nltk unavailable: {e}")
        return 0.0
    pred_chars = list(prediction.lower())
    ref_chars = list(reference.lower())
    if not pred_chars or not ref_chars:
        return 0.0
    smooth = SmoothingFunction().method1
    try:
        return float(
            sentence_bleu(
                [ref_chars],
                pred_chars,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
        )
    except Exception as e:
        logger.warning(f"char_bleu failed: {e}")
        return 0.0


async def _judge_accuracy_with_prompt(
    api_base: str, api_key: str, model_name: str,
    system_prompt: str, label: str,
    question: str, ground_truth: str, extracted_answer: str,
    options: dict | None = None,
) -> float:
    """Internal helper: LLM-judged binary accuracy with the given system prompt.

    Different from judge_correctness: this TAKES the ground truth and asks the
    judge whether the extracted answer matches it. judge_correctness is for the
    no-label path and decides correctness from question + model knowledge alone.
    """
    if not extracted_answer:
        return 0.0
    options_line = ""
    if options:
        options_line = "Options: " + ", ".join(f"{k}. {v}" for k, v in options.items()) + "\n"
    user_prompt = (
        f"Question: {question}\n"
        f"{options_line}"
        f"Ground truth answer: {ground_truth}\n"
        f"Model's extracted answer: {extracted_answer}\n\n"
        "Is the model's answer correct under the rubric above?"
    )
    try:
        content = await _call_api(
            api_base, api_key, model_name, system_prompt, user_prompt, max_tokens=64
        )
        verdict = _extract_judge_verdict(content)
        return 1.0 if verdict == "correct" else 0.0
    except Exception as e:
        logger.warning(f"judge_accuracy ({label}) failed: {type(e).__name__}: {e!r}")
        return 0.0


async def judge_accuracy_lenient(
    api_base: str, api_key: str, model_name: str,
    question: str, ground_truth: str, extracted_answer: str,
    options: dict | None = None,
) -> float:
    """Lenient: synonyms / ICD-code variants / broader-or-narrower subtype
    of the same core diagnosis are all accepted. Different diseases rejected."""
    return await _judge_accuracy_with_prompt(
        api_base, api_key, model_name,
        JUDGE_ACCURACY_LENIENT_PROMPT, "lenient",
        question, ground_truth, extracted_answer, options,
    )


async def judge_accuracy_strict(
    api_base: str, api_key: str, model_name: str,
    question: str, ground_truth: str, extracted_answer: str,
    options: dict | None = None,
) -> float:
    """Strict: same diagnosis at the same level of specificity. Subtype
    mismatch (broader OR narrower than GT) is REJECTED."""
    return await _judge_accuracy_with_prompt(
        api_base, api_key, model_name,
        JUDGE_ACCURACY_STRICT_PROMPT, "strict",
        question, ground_truth, extracted_answer, options,
    )


async def judge_correctness(
    api_base: str, api_key: str, model_name: str,
    question: str, context: str, response: str, extracted_answer: str,
) -> float:
    """Binary correctness when no ground truth available."""
    if not extracted_answer:
        return 0.0
    ctx_line = f"Context: {context[:1500]}\n" if context else ""
    user_prompt = (
        f"{ctx_line}"
        f"Question: {question}\n\n"
        f"Model's full response:\n{response[:2500]}\n\n"
        f"Model's extracted answer: {extracted_answer}\n\n"
        "Is the model's answer correct?"
    )
    try:
        content = await _call_api(
            api_base, api_key, model_name, JUDGE_CORRECTNESS_PROMPT, user_prompt, max_tokens=64
        )
        verdict = _extract_judge_verdict(content)
        return 1.0 if verdict == "correct" else 0.0
    except Exception as e:
        logger.warning(f"judge_correctness failed: {type(e).__name__}: {e!r}")
        return 0.0


async def _evolve_addon(
    composite_score: float,
    question: str,
    solution_str: str,
    ground_truth: str,
    api_base: str,
    api_key: str,
    model_name: str,
    evolve_dir: str,
    w_judge: float,
    w_func: float,
) -> tuple[float, float, float]:
    """Reward-evolution ADD-ON (training-only). Fold the evolvable judge prompt + executable
    function ON TOP of the existing composite instead of replacing it.

    Returns ``(total, dynamic_judge, dynamic_function)`` where
    ``total = (composite + w_judge*dynamic_judge + w_func*dynamic_function) / (1 + w_judge + w_func)``
    in [0,1]. The full composite (acc / lenient / strict / answer_quality / reasoning / format /
    embed / bleu) stays the BASE reward; the evolvable judge prompt (dynamic_judge) and executable
    function (dynamic_function) are additive signals on top. Reads ``{evolve_dir}/current/``.
    """
    from verl.utils.reward_score import reward_evolution as RE  # lazy import avoids a cycle

    # Expose judge/embed credentials to the evolved function (it only gets the fixed 3-arg
    # signature, so it reads endpoints from the environment). Milvus creds come from the run
    # script's env exports; setdefault never clobbers those. See FUNCTION_TOOLING_GUIDE.
    for k, v in (
        ("REWARD_JUDGE_API_BASE", api_base),
        ("REWARD_JUDGE_API_KEY", api_key),
        ("REWARD_JUDGE_MODEL", model_name),
    ):
        if v:
            os.environ.setdefault(k, str(v))

    judge_prompt, fn, _fn_src = RE.get_current_artifacts(evolve_dir)
    # The function may now do blocking network I/O (judge / Milvus / tool download). Run it in a
    # worker thread so a slow call does not stall this worker's event loop — keeping the per-sample
    # reward batch genuinely concurrent (a generous REWARD_FN_TIMEOUT is then acceptable).
    dynamic_judge, dynamic_function = await asyncio.gather(
        RE.score_with_judge_prompt(
            api_base, api_key, model_name, judge_prompt, question, solution_str, ground_truth
        ),
        asyncio.to_thread(RE.safe_call_function, fn, question, solution_str, ground_truth),
    )
    denom = 1.0 + w_judge + w_func
    total = (composite_score + w_judge * dynamic_judge + w_func * dynamic_function) / denom
    return total, dynamic_judge, dynamic_function


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    api_base: str = "",
    api_key: str = "EMPTY",
    model_name: str = "",
    embed_api_base: str = "",
    embed_api_key: str = "EMPTY",
    embed_model: str = "",
    gen_server_url: str = "",
    evolve_enable: bool = False,
    evolve_dir: str = "",
    evolve_w_judge: float = 0.7,
    evolve_w_func: float = 0.3,
    rubric_mode: bool = False,
    provider: str = "",
    val_api_base: str = "",
    val_api_key: str = "EMPTY",
    val_model_name: str = "",
    val_provider: str = "",
    **kwargs,
) -> dict:
    """Compute composite reward.

    Components:
    - accuracy (0-1): exact/normalized match, or LLM-judged correct/incorrect for no-label.
    - answer_quality (1-5 → normalized to 0-1): LLM-judged alignment of extracted answer
      with ground truth. For no-label mode, derived from judge_correctness (5 or 1).
    - reasoning_quality (1-5 → normalized to 0-1): LLM-judged reasoning quality.
    - format_ok (0-1): \\boxed{} format present.
    - embed_sim (0-1): cosine similarity between the extracted boxed answer and the
      ground truth via an OpenAI-compatible embeddings endpoint (the same retriever
      used by the generation server). Smooth signal; 0 if server unreachable.
    - char_bleu (0-1): character-level BLEU-4 with smoothing between extracted answer
      and ground truth (smooth signal that fires when accuracy collapses to 0).

    composite = 0.10*acc + 0.05*judge_lenient + 0.05*judge_strict
              + 0.20*(answer_q/5) + 0.15*(reasoning/5) + 0.15*format
              + 0.20*embed_sim + 0.10*char_bleu
    """
    extra_info = extra_info or {}

    # === Rubric mode (HealthBench-Professional task+rubric co-generation) ===
    # When enabled (reward_kwargs.rubric_mode) or the data_source is a HealthBench
    # set, the reward is PURELY the co-generated rubric graded by an LLM judge —
    # the 8-component composite and the judge/function reward-evolution add-on are
    # bypassed entirely. Training grades with self (server 5); validation grades
    # with the gpt-chat-latest judge (the rubric scorer branches on _is_validation).
    if rubric_mode or str(data_source or "").startswith("healthbench"):
        from verl.utils.reward_score import healthbench_pro

        result = await healthbench_pro.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            provider=provider,
            val_api_base=val_api_base,
            val_api_key=val_api_key,
            val_model_name=val_model_name,
            val_provider=val_provider,
            # D1 (2026-08-04 audit): these were accepted into **kwargs and never
            # forwarded, so the outage-rescue branch in healthbench_pro was DEAD
            # CODE — during the 2026-08-01 server5 outage every judge failure
            # silently scored 0.0 instead of falling back.
            fallback_api_base=kwargs.get("fallback_api_base", ""),
            fallback_api_key=kwargs.get("fallback_api_key", "EMPTY"),
            fallback_model_name=kwargs.get("fallback_model_name", ""),
            fallback_provider=kwargs.get("fallback_provider", ""),
            # Retrieval-coverage judge (the second graded component). Defaults to
            # the TRAIN judge when unset — deliberately not the val judge, so a
            # val-judge outage cannot perturb training.
            cov_api_base=kwargs.get("cov_api_base", ""),
            cov_api_key=kwargs.get("cov_api_key", "EMPTY"),
            cov_model_name=kwargs.get("cov_model_name", ""),
            cov_provider=kwargs.get("cov_provider", ""),
        )
        # Feed the rubric fraction back to the gen server for difficulty
        # calibration (training only; validation rows have no gen-server entry).
        server_url = gen_server_url or os.environ.get("GEN_SERVER_URL", "")
        question_id = extra_info.get("question_id", "")
        # _is_validation does not reliably survive Ray dispatch, so ALSO gate on
        # data_source: real-benchmark rows must never reach /report (a 525-item
        # val burst flushes the gen server's 64-item accuracy window and poisons
        # its difficulty controller — observed in healthbench_diag_seeded).
        is_benchmark_row = str(data_source or "").startswith("healthbench_professional")
        if server_url and question_id and not extra_info.get("_is_validation", False) \
                and not is_benchmark_row:
            await _report_to_gen_server(server_url, question_id, float(result.get("acc", 0.0)))
        return result

    question = extra_info.get("question", "")
    context = extra_info.get("context", "")
    options = extra_info.get("options", {}) if isinstance(extra_info.get("options"), dict) else {}
    has_label = bool(ground_truth and ground_truth.strip())

    # 1. Format check
    format_ok = 1.0 if check_format(solution_str) else 0.0

    # 2. Accuracy check
    # `accuracy` is strict normalized exact match (or LLM correctness in no-label).
    # `judge_acc_lenient` and `judge_acc_strict` are LLM-judged disease matches at
    # two strictness levels — only meaningful with a ground truth. In no-label
    # mode they mirror `accuracy` so the weight isn't wasted.
    extracted_answer = extract_final_answer(solution_str)
    judge_acc_lenient = 0.0
    judge_acc_strict = 0.0
    if has_label:
        is_correct, extracted_answer = check_accuracy(solution_str, ground_truth)
        accuracy = 1.0 if is_correct else 0.0
        if api_base and extracted_answer:
            judge_acc_lenient = await judge_accuracy_lenient(
                api_base=api_base, api_key=api_key, model_name=model_name,
                question=question, ground_truth=ground_truth,
                extracted_answer=extracted_answer, options=options,
            )
            judge_acc_strict = await judge_accuracy_strict(
                api_base=api_base, api_key=api_key, model_name=model_name,
                question=question, ground_truth=ground_truth,
                extracted_answer=extracted_answer, options=options,
            )
        else:
            judge_acc_lenient = float(accuracy)
            judge_acc_strict = float(accuracy)
    elif api_base and extracted_answer:
        accuracy = await judge_correctness(
            api_base=api_base, api_key=api_key, model_name=model_name,
            question=question, context=context,
            response=solution_str, extracted_answer=extracted_answer,
        )
        judge_acc_lenient = accuracy
        judge_acc_strict = accuracy
    else:
        accuracy = 0.0

    # 3. Answer quality (1-5, always use judge for consistency)
    if has_label and api_base and extracted_answer:
        answer_quality = await judge_answer_quality(
            api_base=api_base, api_key=api_key, model_name=model_name,
            question=question, ground_truth=ground_truth,
            extracted_answer=extracted_answer, options=options,
        )
    elif has_label and extracted_answer:
        # No API: derive from exact match
        answer_quality = 5.0 if accuracy == 1.0 else 1.0
    elif not has_label:
        # No-label: derive from judge_correctness (accuracy)
        answer_quality = 5.0 if accuracy == 1.0 else 1.0
    else:
        answer_quality = 1.0

    # 4. Reasoning quality
    reasoning_score = 3.0
    if api_base:
        reasoning_score = await judge_reasoning(
            api_base=api_base, api_key=api_key, model_name=model_name,
            question=question, context=context,
            response=solution_str, ground_truth=ground_truth,
        )

    # 5. Embedding semantic similarity (smooth signal). Embed only the extracted
    # boxed answer and the ground truth (no reasoning trace), batched into one
    # /embeddings call.
    embed_sim = 0.0
    if embed_api_base and embed_model and ground_truth and extracted_answer:
        embed_sim = await embedding_similarity(
            embed_api_base, embed_api_key, embed_model, extracted_answer, ground_truth
        )

    # 6. Character-level BLEU (smooth signal, in-process)
    char_bleu_score = 0.0
    if ground_truth and extracted_answer:
        char_bleu_score = char_bleu(extracted_answer, ground_truth)

    # 7. Composite
    score = (
        ACCURACY_WEIGHT * accuracy
        + JUDGE_ACCURACY_LENIENT_WEIGHT * judge_acc_lenient
        + JUDGE_ACCURACY_STRICT_WEIGHT * judge_acc_strict
        + ANSWER_QUALITY_WEIGHT * (answer_quality / 5.0)
        + REASONING_WEIGHT * (reasoning_score / 5.0)
        + FORMAT_WEIGHT * format_ok
        + EMBED_SIM_WEIGHT * embed_sim
        + CHAR_BLEU_WEIGHT * char_bleu_score
    )

    # === Reward-evolution ADD-ON (optional, training-only) ===
    # When enabled and NOT during validation, fold the evolvable judge prompt + executable
    # function ON TOP of the composite above (the composite stays the base reward; the new
    # judge/function are additive signals, not the sole reward). Validation always uses the
    # pure composite so val metrics stay comparable. Off by default => composite unchanged.
    dynamic_judge = 0.0
    dynamic_function = 0.0
    evolve_on = bool(evolve_enable and evolve_dir and not extra_info.get("_is_validation", False))
    if evolve_on:
        try:
            score, dynamic_judge, dynamic_function = await _evolve_addon(
                composite_score=score,
                question=question,
                solution_str=solution_str,
                ground_truth=ground_truth,
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                evolve_dir=evolve_dir,
                w_judge=evolve_w_judge,
                w_func=evolve_w_func,
            )
        except Exception as e:  # never let the add-on break the (working) composite reward
            logger.warning(f"evolve add-on failed, using composite only: {type(e).__name__}: {e}")
            evolve_on = False

    if random.random() < DEBUG_PRINT_PROB:
        mode = "label" if has_label else "no-label"
        print(f"\n{'=' * 60}")
        print(f"[REWARD DEBUG] mode={mode}  format={extra_info.get('format', '?')}")
        print(f"  question: {question}")
        print(f"  ground_truth: {ground_truth!r}")
        print(f"  extracted: {extracted_answer!r}")
        print(f"  accuracy={accuracy:.1f}  judge_lenient={judge_acc_lenient:.1f}  "
              f"judge_strict={judge_acc_strict:.1f}  "
              f"answer_q={answer_quality:.1f}  reasoning={reasoning_score:.1f}  "
              f"format={format_ok:.1f}  embed_sim={embed_sim:.2f}  "
              f"char_bleu={char_bleu_score:.2f}")
        print(f"  total_score={score:.3f}")
        print(f"  response: {solution_str}")
        print(f"{'=' * 60}\n")

    # Feed accuracy back to the generation server so it can keep its
    # sliding-window difficulty calibration and per-id log up to date.
    # gen_server_url comes from reward_kwargs in the run script; falls back
    # to env var so ad-hoc evals can opt in / out without re-launching.
    server_url = gen_server_url or os.environ.get("GEN_SERVER_URL", "")
    question_id = (extra_info or {}).get("question_id", "")
    if server_url and question_id:
        await _report_to_gen_server(server_url, question_id, accuracy)

    # HOMOGENEOUS_DEFAULTS first: the trainer snapshots the reward key set from
    # SAMPLE 0 only, so any key this branch omits that the healthbench branch returns
    # is silently dropped batch-wide (or KeyErrors, depending on which sample is 0).
    # This branch used to differ from healthbench_pro by 8 keys, so a genuinely mixed
    # batch already broke.
    # Lazy import, mirroring the convention on both sides of this pair (healthbench_pro
    # imports `_call_api` from here lazily for the same reason).
    from verl.utils.reward_score.healthbench_pro import HOMOGENEOUS_DEFAULTS

    result = {
        **HOMOGENEOUS_DEFAULTS,
        "score": score,
        "acc": accuracy,
        "judge_acc_lenient": judge_acc_lenient,
        "judge_acc_strict": judge_acc_strict,
        "answer_quality": answer_quality,
        "reasoning_quality": reasoning_score,
        "format_ok": format_ok,
        "embed_sim": embed_sim,
        "char_bleu": char_bleu_score,
        "extracted_answer": extracted_answer or "",
        # ALWAYS emitted, 0.0 when the add-on is off or threw. `evolve_on` is cleared
        # PER SAMPLE inside the exception handler above, so emitting these
        # conditionally meant a batch where sample 0 succeeded and sample k threw
        # raised KeyError and killed the step (and the reverse order silently dropped
        # both metrics). Byte-identity of a dict is not worth a crash class.
        "dynamic_judge": dynamic_judge,
        "dynamic_function": dynamic_function,
    }
    return result
