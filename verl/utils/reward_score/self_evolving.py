"""
Reward function for self-evolving medical agent training.

Combines FOUR components:
- Accuracy (0.3): Extracted answer matches ground truth (exact/normalized match),
  OR if ground_truth is empty, LLM judge deems it correct.
- Reasoning quality (0.2): LLM judge rates the <think>...</think> reasoning 1-5.
- Answer quality (0.3): LLM judge rates alignment between the EXTRACTED BOXED
  ANSWER ONLY (reasoning stripped) and the ground truth, 1-5. This applies to
  BOTH free-response and MCQ for consistent reward format.
- Format (0.2): \\boxed{} format present (binary 0/1).

If ground truth is empty (no-label mode), answer_quality is derived from
judge_correctness (5 if judged correct, 1 if incorrect).
"""

import logging
import os
import random
import re

import aiohttp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


JUDGE_REASONING_PROMPT = """\
You are a medical reasoning evaluator. You are given a question, the model's full response \
(including reasoning inside <think></think>), and the correct answer. Rate the quality \
of the model's REASONING (not the final answer) on a scale from 1 to 5:

1 - No reasoning, or completely irrelevant reasoning
2 - Minimal reasoning with major logical errors
3 - Some relevant reasoning but with gaps or minor errors
4 - Good reasoning that mostly follows from the evidence
5 - Excellent reasoning that is thorough, evidence-based, and logically sound

Output ONLY a single integer from 1 to 5."""


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

Output ONLY a single integer from 1 to 5."""


JUDGE_CORRECTNESS_PROMPT = """\
You are a medical expert evaluating whether a model's answer to a medical question is \
correct. Use the provided question/context and your own medical knowledge to determine \
if the model's extracted answer is correct.

Output ONLY one of: "correct" or "incorrect"."""


# Reward component weights (sum to 1.0)
ACCURACY_WEIGHT = 0.3
REASONING_WEIGHT = 0.2
ANSWER_QUALITY_WEIGHT = 0.3
FORMAT_WEIGHT = 0.2

DEBUG_PRINT_PROB = 0.01


def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer from \\boxed{...} (last occurrence)."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def check_format(text: str) -> bool:
    return bool(re.search(r"\\boxed\{[^}]*\}", text))


def check_accuracy(solution_str: str, ground_truth: str) -> tuple[bool, str | None]:
    """Normalized string match between extracted boxed answer and ground_truth."""
    extracted = extract_boxed_answer(solution_str)
    if extracted is None:
        return False, None
    gt = ground_truth.strip().lower()
    pred = extracted.strip().lower()
    # For MCQ, pick first letter
    if len(gt) == 1 and gt in "abcd":
        pred_letter = re.sub(r"[^a-d]", "", pred)[:1]
        return pred_letter == gt, extracted
    return pred == gt, extracted


async def _call_api(
    api_base: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 16,
) -> str:
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()


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
            api_base, api_key, model_name, JUDGE_REASONING_PROMPT, user_prompt, max_tokens=16
        )
        m = re.search(r"[1-5]", content)
        if m:
            return float(m.group())
        return 3.0
    except Exception as e:
        logger.warning(f"judge_reasoning failed: {e}")
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
            api_base, api_key, model_name, JUDGE_ANSWER_QUALITY_PROMPT, user_prompt, max_tokens=16
        )
        m = re.search(r"[1-5]", content)
        if m:
            return float(m.group())
        return 1.0
    except Exception as e:
        logger.warning(f"judge_answer_quality failed: {e}")
        return 1.0


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
            api_base, api_key, model_name, JUDGE_CORRECTNESS_PROMPT, user_prompt, max_tokens=16
        )
        c = content.lower()
        return 1.0 if "correct" in c and "incorrect" not in c else 0.0
    except Exception as e:
        logger.warning(f"judge_correctness failed: {e}")
        return 0.0


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    api_base: str = "",
    api_key: str = "EMPTY",
    model_name: str = "",
    **kwargs,
) -> dict:
    """Compute composite reward.

    Components:
    - accuracy (0-1): exact/normalized match, or LLM-judged correct/incorrect for no-label.
    - answer_quality (1-5 → normalized to 0-1): LLM-judged alignment of extracted answer
      with ground truth. For no-label mode, derived from judge_correctness (5 or 1).
    - reasoning_quality (1-5 → normalized to 0-1): LLM-judged reasoning quality.
    - format_ok (0-1): \\boxed{} format present.

    composite = 0.3*acc + 0.3*(answer_q/5) + 0.2*(reasoning/5) + 0.2*format
    """
    extra_info = extra_info or {}
    question = extra_info.get("question", "")
    context = extra_info.get("context", "")
    options = extra_info.get("options", {}) if isinstance(extra_info.get("options"), dict) else {}
    has_label = bool(ground_truth and ground_truth.strip())

    # 1. Format check
    format_ok = 1.0 if check_format(solution_str) else 0.0

    # 2. Accuracy check
    extracted_answer = extract_boxed_answer(solution_str)
    if has_label:
        is_correct, extracted_answer = check_accuracy(solution_str, ground_truth)
        accuracy = 1.0 if is_correct else 0.0
    elif api_base and extracted_answer:
        accuracy = await judge_correctness(
            api_base=api_base, api_key=api_key, model_name=model_name,
            question=question, context=context,
            response=solution_str, extracted_answer=extracted_answer,
        )
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

    # 5. Composite
    score = (
        ACCURACY_WEIGHT * accuracy
        + ANSWER_QUALITY_WEIGHT * (answer_quality / 5.0)
        + REASONING_WEIGHT * (reasoning_score / 5.0)
        + FORMAT_WEIGHT * format_ok
    )

    if random.random() < DEBUG_PRINT_PROB:
        mode = "label" if has_label else "no-label"
        print(f"\n{'=' * 60}")
        print(f"[REWARD DEBUG] mode={mode}  format={extra_info.get('format', '?')}")
        print(f"  question: {question[:200]}")
        print(f"  ground_truth: {ground_truth!r}")
        print(f"  extracted: {extracted_answer!r}")
        print(f"  accuracy={accuracy:.1f}  answer_q={answer_quality:.1f}  "
              f"reasoning={reasoning_score:.1f}  format={format_ok:.1f}")
        print(f"  total_score={score:.3f}")
        print(f"  response (first 300): {solution_str[:300]}")
        print(f"{'=' * 60}\n")

    return {
        "score": score,
        "acc": accuracy,
        "answer_quality": answer_quality,
        "reasoning_quality": reasoning_score,
        "format_ok": format_ok,
        "extracted_answer": extracted_answer or "",
    }
