"""
Reward function for self-evolving medical agent training.

Combines three components:
- Accuracy (0.5): Whether the extracted answer matches ground truth
- Reasoning quality (0.4): API-based judgment of reasoning on a 1-5 scale
- Format (0.1): Whether the answer is in \\boxed{} format
"""

import logging
import os
import re

import aiohttp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

JUDGE_SYSTEM_PROMPT = """\
You are a biomedical reasoning evaluator. You will be given a question, context, \
the model's response, and the correct answer. Rate the quality of the model's \
reasoning on a scale from 1 to 5:

1 - No reasoning or completely irrelevant reasoning
2 - Minimal reasoning with major logical errors
3 - Some relevant reasoning but with gaps or minor errors
4 - Good reasoning that mostly follows from the evidence
5 - Excellent reasoning that is thorough, evidence-based, and logically sound

Output ONLY a single integer from 1 to 5."""

# Weights for reward components
ACCURACY_WEIGHT = 0.5
REASONING_WEIGHT = 0.4
FORMAT_WEIGHT = 0.1


def extract_boxed_answer(text: str) -> str | None:
    """Extract the answer from \\boxed{...} in the text."""
    # Search from the end of the text (last boxed answer)
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip().lower()
    return None


def check_format(text: str) -> bool:
    """Check if the response contains \\boxed{...} format."""
    return bool(re.search(r'\\boxed\{[^}]*\}', text))


def check_accuracy(solution_str: str, ground_truth: str) -> tuple[bool, str | None]:
    """Check if extracted answer matches ground truth.

    Returns (is_correct, extracted_answer).
    """
    extracted = extract_boxed_answer(solution_str)
    if extracted is None:
        return False, None

    gt = ground_truth.strip().lower()
    # Normalize common variants
    answer_map = {"yes": "yes", "no": "no", "maybe": "maybe", "y": "yes", "n": "no"}
    normalized_extracted = answer_map.get(extracted, extracted)
    normalized_gt = answer_map.get(gt, gt)

    return normalized_extracted == normalized_gt, extracted


async def call_judge_api(
    api_base: str,
    api_key: str,
    model_name: str,
    question: str,
    context: str,
    response: str,
    ground_truth: str,
) -> float:
    """Call external API to judge reasoning quality (1-5)."""
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    user_prompt = (
        f"Question: {question}\n"
        f"Context: {context[:2000]}\n"  # Truncate long contexts
        f"Correct answer: {ground_truth}\n"
        f"Model's response:\n{response[:3000]}\n\n"  # Truncate long responses
        "Rate the reasoning quality (1-5):"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Extract integer from response
                match = re.search(r'[1-5]', content)
                if match:
                    return float(match.group())
                logger.warning(f"Could not parse judge rating from: {content}")
                return 3.0
    except Exception as e:
        logger.warning(f"Judge API call failed: {e}. Defaulting to 3.0")
        return 3.0


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

    Returns:
        dict with keys: score, acc, reasoning_quality, format_ok, extracted_answer
    """
    extra_info = extra_info or {}

    # 1. Format check
    format_ok = 1.0 if check_format(solution_str) else 0.0

    # 2. Accuracy check
    is_correct, extracted_answer = check_accuracy(solution_str, ground_truth)
    accuracy = 1.0 if is_correct else 0.0

    # 3. Reasoning quality (API-based)
    reasoning_score = 3.0  # default
    if api_base:
        question = extra_info.get("question", "")
        # Try to get context from the extra_info or use empty string
        context = extra_info.get("context", "")
        reasoning_score = await call_judge_api(
            api_base=api_base,
            api_key=api_key,
            model_name=model_name,
            question=question,
            context=context,
            response=solution_str,
            ground_truth=ground_truth,
        )

    # 4. Composite score
    score = (
        ACCURACY_WEIGHT * accuracy
        + REASONING_WEIGHT * (reasoning_score / 5.0)
        + FORMAT_WEIGHT * format_ok
    )

    return {
        "score": score,
        "acc": accuracy,
        "reasoning_quality": reasoning_score,
        "format_ok": format_ok,
        "extracted_answer": extracted_answer or "",
    }
