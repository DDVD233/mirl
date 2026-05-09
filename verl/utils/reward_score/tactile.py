"""Reward scoring for tactile video QA (multi-choice, possibly multi-label).

Ground truth and predictions are comma-separated uppercase letters (e.g. "A", "B,C,D").
Order-agnostic comparison via set operations.

Returns a dict so the reward manager can log evaluation-only metrics (f1, precision, recall)
alongside the training reward.

Training reward = acc_weight * acc + sim_weight * jaccard + format_weight * format
"""

import re


def extract_boxed_answer(predict_str: str) -> str | None:
    """Extract the content inside the last \\boxed{...} in the prediction."""
    idx = predict_str.rfind("\\boxed{")
    if idx < 0:
        return None
    # Walk forward to find matching closing brace
    depth = 0
    i = idx + len("\\boxed{") - 1  # points at '{'
    while i < len(predict_str):
        if predict_str[i] == "{":
            depth += 1
        elif predict_str[i] == "}":
            depth -= 1
            if depth == 0:
                return predict_str[idx + len("\\boxed{"):i]
        i += 1
    return None


def parse_labels(answer_str: str) -> set[str]:
    """Parse a comma-separated answer string into a set of normalized labels."""
    if not answer_str:
        return set()
    return {label.strip().upper() for label in answer_str.split(",") if label.strip()}


def format_reward(predict_str: str) -> float:
    """Check for <think>...</think>...\\boxed{...} format."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def acc_reward(pred_labels: set[str], gt_labels: set[str]) -> float:
    """Exact-match accuracy: 1.0 iff predicted label set equals ground truth."""
    return 1.0 if pred_labels == gt_labels else 0.0


def jaccard_similarity(pred_labels: set[str], gt_labels: set[str]) -> float:
    """Jaccard similarity between predicted and ground truth label sets."""
    if not pred_labels and not gt_labels:
        return 1.0
    if not pred_labels or not gt_labels:
        return 0.0
    return len(pred_labels & gt_labels) / len(pred_labels | gt_labels)


def f1_score(pred_labels: set[str], gt_labels: set[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 between predicted and ground truth label sets."""
    if not pred_labels and not gt_labels:
        return 1.0, 1.0, 1.0
    if not pred_labels or not gt_labels:
        return 0.0, 0.0, 0.0
    tp = len(pred_labels & gt_labels)
    precision = tp / len(pred_labels)
    recall = tp / len(gt_labels)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_score(
    predict_str: str,
    ground_truth: str,
    acc_weight: float = 0.5,
    sim_weight: float = 0.4,
    format_weight: float = 0.1,
) -> dict:
    """Compute reward and evaluation metrics for tactile video QA.

    Training reward = acc_weight * acc + sim_weight * jaccard + format_weight * format

    Returns:
        dict with keys:
            score: float        - the training reward
            acc: float          - exact-match accuracy (0 or 1)
            jaccard: float      - Jaccard similarity
            f1: float           - F1 score (evaluation only)
            precision: float    - precision (evaluation only)
            recall: float       - recall (evaluation only)
            format: float       - format compliance (0 or 1)
    """
    boxed = extract_boxed_answer(predict_str)
    pred_labels = parse_labels(boxed)
    gt_labels = parse_labels(ground_truth)

    fmt = format_reward(predict_str)
    acc = acc_reward(pred_labels, gt_labels)
    jacc = jaccard_similarity(pred_labels, gt_labels)
    precision, recall, f1 = f1_score(pred_labels, gt_labels)

    score = acc_weight * acc + sim_weight * jacc + format_weight * fmt

    return {
        "score": score,
        "acc": acc,
        "jaccard": jacc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "format": fmt,
    }
