"""Reward scoring for medical/CLIMB tasks (condition diagnosis).

Ground truth and predictions are comma-separated condition phrases
(e.g. "Pleural Effusion, Support Devices").
Uses F1 over condition sets + token similarity + format compliance.

Returns a dict so the reward manager can log evaluation-only metrics
alongside the training reward.

Training reward = 0.5 * f1 + 0.3 * similarity + 0.2 * format
"""

import re


def extract_boxed_answer(predict_str: str) -> str | None:
    """Extract the content inside the last \\boxed{...} in the prediction."""
    idx = predict_str.rfind("\\boxed{")
    if idx < 0:
        return None
    depth = 0
    i = idx + len("\\boxed{") - 1
    while i < len(predict_str):
        if predict_str[i] == "{":
            depth += 1
        elif predict_str[i] == "}":
            depth -= 1
            if depth == 0:
                return predict_str[idx + len("\\boxed{"):i]
        i += 1
    return None


def format_reward(predict_str: str) -> float:
    """Check for <think>...</think>...\\boxed{...} format."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def parse_conditions(text: str) -> set[str]:
    """Parse a multi-condition answer string into a normalized set."""
    text = text.strip().lower()
    for sep in [", ", " and ", " & ", ",", "&"]:
        if sep in text:
            return {c.strip() for c in text.split(sep) if c.strip()}
    return {text} if text else set()


def f1_over_conditions(pred_conditions: set[str], gt_conditions: set[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 over condition sets."""
    if not pred_conditions and not gt_conditions:
        return 1.0, 1.0, 1.0
    if not pred_conditions or not gt_conditions:
        return 0.0, 0.0, 0.0

    tp = len(pred_conditions & gt_conditions)
    precision = tp / len(pred_conditions)
    recall = tp / len(gt_conditions)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def jaccard_similarity(pred_text: str, gt_text: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    pred_tokens = set(pred_text.lower().split())
    gt_tokens = set(gt_text.lower().split())

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    return len(pred_tokens & gt_tokens) / len(pred_tokens | gt_tokens)


def compute_score(
    predict_str: str,
    ground_truth: str,
    f1_weight: float = 0.5,
    sim_weight: float = 0.3,
    format_weight: float = 0.2,
) -> dict:
    """Compute reward and evaluation metrics for medical condition diagnosis.

    Training reward = f1_weight * f1 + sim_weight * jaccard + format_weight * format

    Returns:
        dict with keys:
            score: float     - the training reward
            acc: float       - exact-match accuracy (0 or 1)
            f1: float        - F1 over condition sets
            precision: float - precision over condition sets
            recall: float    - recall over condition sets
            jaccard: float   - token-level Jaccard similarity
            format: float    - format compliance (0 or 1)
    """
    boxed = extract_boxed_answer(predict_str)
    pred_text = (boxed or "").lower()
    gt_text = ground_truth.lower()

    pred_conditions = parse_conditions(pred_text)
    gt_conditions = parse_conditions(gt_text)

    fmt = format_reward(predict_str)
    acc = 1.0 if pred_conditions == gt_conditions else 0.0
    precision, recall, f1 = f1_over_conditions(pred_conditions, gt_conditions)
    jacc = jaccard_similarity(pred_text, gt_text)

    score = f1_weight * f1 + sim_weight * jacc + format_weight * fmt

    return {
        "score": score,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "jaccard": jacc,
        "similarity": 0.0,
        "format": fmt,
    }
