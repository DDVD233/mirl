"""Reward scoring for MIMIC clinical QA.

Supports two formats:
- Multiple choice (q3, q5, q6): exact string match on \\boxed{} content.
- Long response (q1, q2): BioBERT embedding cosine similarity between
  \\boxed{} content and ground truth. Queries a separate BioBERT server
  to avoid torch dispatch mode conflicts inside Ray workers.

Start the server before training:
    python -m verl.utils.reward_score.biobert_server [--port 5100] [--device cuda]

Set BIOBERT_SERVER_URL env var if not using default (http://localhost:5100).
"""

import os
import re

import requests

# Long-response qa_types (free-text diagnosis answers)
LONG_RESPONSE_TYPES = {"1", "2"}

BIOBERT_SERVER_URL = os.environ.get("BIOBERT_SERVER_URL", "http://localhost:5100")


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


# ---------------------------------------------------------------------------
# BioBERT embedding similarity via external server
# ---------------------------------------------------------------------------


def embedding_similarity(pred_text: str, gt_text: str) -> float:
    """Cosine similarity between BioBERT embeddings via external server."""
    if not pred_text or not gt_text:
        print(f"[mimic reward] embedding_similarity: empty input, pred={repr(pred_text[:100])}, gt={repr(gt_text[:100])}, sim=0.0")
        return 0.0
    try:
        resp = requests.post(
            f"{BIOBERT_SERVER_URL}/similarity",
            json={"pred": pred_text, "gt": gt_text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data["similarity"]
        elapsed = data["time"]
        print(f"[mimic reward] embedding_similarity: sim={result:.4f}, time={elapsed:.3f}s, pred={repr(pred_text[:80])}, gt={repr(gt_text[:80])}")
        return result
    except Exception as e:
        print(f"[mimic reward] embedding_similarity: server error: {e}, returning 0.0")
        return 0.0


# ---------------------------------------------------------------------------
# Multiple choice scoring
# ---------------------------------------------------------------------------
def mc_accuracy(predict_str: str, ground_truth: str) -> float:
    """Exact match of boxed answer against ground truth (case-insensitive)."""
    boxed = extract_boxed_answer(predict_str)
    if boxed is None:
        return 0.0
    return 1.0 if boxed.strip().upper() == ground_truth.strip().upper() else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_score(
    predict_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_weight: float = 0.1,
) -> dict:
    """Compute reward for MIMIC clinical QA.

    For multiple choice (q3/q5/q6):
        score = (1 - format_weight) * acc + format_weight * format

    For long response (q1/q2):
        score = (1 - format_weight) * similarity + format_weight * format

    Returns dict with score and diagnostic metrics.
    """
    qa_type = extra_info.get("qa_type", "") if extra_info else ""
    fmt = format_reward(predict_str)

    if qa_type in LONG_RESPONSE_TYPES:
        boxed = extract_boxed_answer(predict_str)
        sim = embedding_similarity(boxed or "", ground_truth)
        acc = 0.0
        score = (1.0 - format_weight) * sim + format_weight * fmt
    else:
        acc = mc_accuracy(predict_str, ground_truth)
        sim = 0.0
        score = (1.0 - format_weight) * acc + format_weight * fmt

    return {
        "score": score,
        "acc": acc,
        "similarity": sim,
        "format": fmt,
    }