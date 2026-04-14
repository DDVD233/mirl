"""
Reward function for self-evolving medical agent training.

Combines three components:
- Accuracy (0.4): Whether the extracted answer matches ground truth,
  or if ground truth is empty, whether an LLM judge deems it correct.
- Reasoning quality (0.3): API-based judgment of reasoning on a 1-5 scale
- Format (0.3): Whether the answer is in \\boxed{} format
"""

import logging
import os
import random
import re

import aiohttp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

JUDGE_REASONING_PROMPT = """\
You are a biomedical reasoning evaluator. You will be given a question, context, \
the model's response, and the correct answer. Rate the quality of the model's \
reasoning on a scale from 1 to 5:

1 - No reasoning or completely irrelevant reasoning
2 - Minimal reasoning with major logical errors
3 - Some relevant reasoning but with gaps or minor errors
4 - Good reasoning that mostly follows from the evidence
5 - Excellent reasoning that is thorough, evidence-based, and logically sound

Output ONLY a single integer from 1 to 5."""

JUDGE_CORRECTNESS_PROMPT = """\
You are a biomedical expert evaluating whether a model's answer to a medical question \
is correct. You must use the provided context AND your own medical knowledge to determine \
the correct answer, then judge whether the model's answer matches.

The model answered a yes/no/maybe question about a biomedical context. You need to:
1. Read the context carefully
2. Determine what the correct answer should be (yes, no, or maybe)
3. Check if the model's extracted answer matches

Output ONLY one of: "correct" or "incorrect"."""

# Weights for reward components
ACCURACY_WEIGHT = 0.4
REASONING_WEIGHT = 0.3
FORMAT_WEIGHT = 0.3

DEBUG_PRINT_PROB = 0.01


def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer from \\boxed{...} in the text."""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip().lower()
    return None


def check_format(text: str) -> bool:
    """Check if the response contains \\boxed{...} format."""
    return bool(re.search(r'\\boxed\{[^}]*\}', text))


def check_accuracy(solution_str: str, ground_truth: str) -> tuple[bool, str | None]:
    """Check if extracted answer matches ground truth via normalized string matching.

    Returns (is_correct, extracted_answer).
    """
    extracted = extract_boxed_answer(solution_str)
    if extracted is None:
        return False, None

    gt = ground_truth.strip().lower()
    pred = extracted.strip().lower()

    return pred == gt, extracted


async def _call_api(api_base: str, api_key: str, model_name: str,
                    system_prompt: str, user_prompt: str,
                    max_tokens: int = 16) -> str:
    """Make an async API call and return the response content."""
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
    """Call external API to judge reasoning quality (1-5)."""
    gt_line = f"Correct answer: {ground_truth}\n" if ground_truth else ""
    user_prompt = (
        f"Question: {question}\n"
        f"Context: {context[:2000]}\n"
        f"{gt_line}"
        f"Model's response:\n{response[:3000]}\n\n"
        "Rate the reasoning quality (1-5):"
    )

    try:
        content = await _call_api(api_base, api_key, model_name,
                                  JUDGE_REASONING_PROMPT, user_prompt)
        match = re.search(r'[1-5]', content)
        if match:
            return float(match.group())
        logger.warning(f"Could not parse judge rating from: {content}")
        return 3.0
    except Exception as e:
        logger.warning(f"Judge reasoning API failed: {e}. Defaulting to 3.0")
        return 3.0


async def judge_correctness(
    api_base: str, api_key: str, model_name: str,
    question: str, context: str, response: str, extracted_answer: str,
) -> float:
    """Call external API to judge if the answer is correct (no ground truth available)."""
    user_prompt = (
        f"Context: {context[:2000]}\n\n"
        f"Question: {question}\n\n"
        f"Model's full response:\n{response[:3000]}\n\n"
        f"Model's extracted final answer: {extracted_answer}\n\n"
        "Based on the context and your medical knowledge, is the model's answer correct?"
    )

    try:
        content = await _call_api(api_base, api_key, model_name,
                                  JUDGE_CORRECTNESS_PROMPT, user_prompt,
                                  max_tokens=32)
        return 1.0 if "correct" in content.lower() and "incorrect" not in content.lower() else 0.0
    except Exception as e:
        logger.warning(f"Judge correctness API failed: {e}. Defaulting to 0.0")
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
    """Compute reward score combining accuracy, reasoning quality, and format.

    If ground_truth is empty, uses LLM judge for correctness instead of string matching.

    Returns:
        dict with keys: score, acc, reasoning_quality, format_ok, extracted_answer
    """
    extra_info = extra_info or {}
    question = extra_info.get("question", "")
    context = extra_info.get("context", "")
    has_label = bool(ground_truth and ground_truth.strip())

    # 1. Format check
    format_ok = 1.0 if check_format(solution_str) else 0.0

    # 2. Accuracy check
    extracted_answer = extract_boxed_answer(solution_str)

    if has_label:
        # Standard: compare against ground truth
        is_correct, extracted_answer = check_accuracy(solution_str, ground_truth)
        accuracy = 1.0 if is_correct else 0.0
    elif api_base and extracted_answer:
        # No label: ask LLM judge to determine correctness
        accuracy = await judge_correctness(
            api_base=api_base, api_key=api_key, model_name=model_name,
            question=question, context=context,
            response=solution_str, extracted_answer=extracted_answer,
        )
    else:
        # No label, no API or no extracted answer
        accuracy = 0.0

    # 3. Reasoning quality (API-based)
    reasoning_score = 3.0
    if api_base:
        reasoning_score = await judge_reasoning(
            api_base=api_base, api_key=api_key, model_name=model_name,
            question=question, context=context,
            response=solution_str, ground_truth=ground_truth,
        )

    # 4. Composite score
    score = (
        ACCURACY_WEIGHT * accuracy
        + REASONING_WEIGHT * (reasoning_score / 5.0)
        + FORMAT_WEIGHT * format_ok
    )

    # Debug printing at DEBUG_PRINT_PROB probability
    if random.random() < DEBUG_PRINT_PROB:
        label_mode = "label" if has_label else "no-label (judge)"
        print(f"\n{'='*60}")
        print(f"[REWARD DEBUG] mode={label_mode}")
        print(f"  question: {question[:200]}")
        print(f"  ground_truth: {ground_truth!r}")
        print(f"  extracted_answer: {extracted_answer!r}")
        print(f"  accuracy: {accuracy:.1f}  reasoning: {reasoning_score:.1f}  format: {format_ok:.1f}")
        print(f"  total_score: {score:.3f}")
        print(f"  response (first 300 chars): {solution_str[:300]}")
        print(f"{'='*60}\n")

    return {
        "score": score,
        "acc": accuracy,
        "reasoning_quality": reasoning_score,
        "format_ok": format_ok,
        "extracted_answer": extracted_answer or "",
    }