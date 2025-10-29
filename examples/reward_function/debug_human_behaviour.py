from typing import List, Dict, Optional
import re
import numpy as np
# from sentence_transformers import SentenceTransformer  # optional

# ---------------------------
# Helpers (unchanged)
# ---------------------------
def extract_boxed_content(text: str) -> str:
    """Extract content within \\boxed{} or similar boxing notations."""
    boxed_match = re.search(r"\\boxed{([^}]*)}", text)
    if boxed_match:
        return boxed_match.group(1)

    markdown_match = re.search(r"\[(.*?)\]", text)
    if markdown_match:
        return markdown_match.group(1)

    answer_match = re.search(r"<answer>(.*?)</answer>", text)
    if answer_match:
        return answer_match.group(1)

    return text

def format_reward(response: str) -> float:
    """
    Require pattern: <think>...</think> ... \boxed{...}
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    """Exact match on normalized label string."""
    return 1.0 if response == ground_truth else 0.0

def cosine_similarity_reward(pred_label: str, ground_truth: str, model=None) -> float:
    """
    If you later enable SBERT, wire it here. For now we keep a stub/constant to avoid OOM risk.
    """
    # Example SBERT usage:
    # embeddings = model.encode([pred_label, ground_truth], convert_to_numpy=True)
    # pred_norm = embeddings[0] / np.linalg.norm(embeddings[0])
    # gt_norm   = embeddings[1] / np.linalg.norm(embeddings[1])
    # cos_sim = float(np.dot(pred_norm, gt_norm))
    # return max(0.0, min(1.0, cos_sim))
    return 0.6

# ---------------------------
# New: soft overlong punishment (same logic as example)
# ---------------------------
def soft_overlong_punishment(response_length: int,
                             max_response_length: int,
                             overlong_buffer_length: int) -> float:
    """
    Piecewise:
      <= (max - buffer): 0
      in ((max - buffer), max]: linear penalty down to -1 at max
      > max: -1
    """
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    elif response_length <= max_response_length:
        return (expected_len - response_length) / float(overlong_buffer_length)
    else:
        return -1.0

# ---------------------------
# Main batch scorer (now with overlong penalty)
# ---------------------------
def human_behaviour_compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[str],
    *,
    # Overlength controls (required to activate punishment)
    max_response_length: Optional[int] = 512,
    overlong_buffer_length: int = 256,
    overlong_penalty_factor: float = 1.0,
    # Optional true token-lengths (preferred). If None, we fall back to len(text).
    response_lengths: Optional[List[int]] = None,
    # Optional: pass a loaded SBERT model if you want real cosine (kept off by default to avoid OOM).
    # embedding_model: Optional[SentenceTransformer] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Returns a list of dicts with:
      - score: overall (includes overlong penalty if max_response_length is provided)
      - standard_score, format_score, similarity_score
      - overlong_score (0..-1), present when max_response_length is provided
    """
    batch_scores = []
    format_weight = 0.2

    # embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')

    # If response_lengths not provided, approximate with raw string length
    if response_lengths is None:
        response_lengths = [len(s or "") for s in solution_strs]

    for i, (data_source, predict_str, ground_truth, extra_info) in enumerate(
        zip(data_sources, solution_strs, ground_truths, extra_infos)
    ):
        # Normalize formatting quirks then extract <boxed{...}>
        full_response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str or "")
        pred_label = extract_boxed_content(full_response).lower()
        gt_norm = (ground_truth or "").lower()

        # Base components
        format_score = format_reward(full_response)
        standard_score = accuracy_reward(pred_label, gt_norm)
        similarity_score = cosine_similarity_reward(pred_label, gt_norm, model=None)  # keep None unless you pass a model

        # Combine (before penalty)
        overall_score = standard_score + format_weight * format_score + 0.5 * similarity_score

        # Optional overlong penalty (only if max_response_length is provided)
        overlong_score = 0.0
        if max_response_length is not None:
            resp_len = int(response_lengths[i])
            overlong_score = soft_overlong_punishment(
                response_length=resp_len,
                max_response_length=int(max_response_length),
                overlong_buffer_length=int(overlong_buffer_length),
            )
            overall_score += overlong_penalty_factor * overlong_score

        scores = {
            "score": overall_score,
            "standard_score": standard_score,
            "format_score": format_score,
            "similarity_score": similarity_score,
            "overlong_score": overlong_score,
        }
        batch_scores.append(scores)

    return batch_scores

# ---------------------------
# Quick demo
# ---------------------------
if __name__ == "__main__":
    response_str = (
        "<think>…</think>\\boxed{anger} trailing text that might be long..."
    )
    data_sources = ["sample_audio.wav"]
    solution_strs = [response_str]
    ground_truths = ["anger"]
    extra_infos = [""]

    scores = human_behaviour_compute_score_batch(
        data_sources, solution_strs, ground_truths, extra_infos,
        max_response_length=512,           # set your model’s max response tokens
        overlong_buffer_length=128,        # grace window
        overlong_penalty_factor=1.0,       # weight of the penalty
        response_lengths=[len(response_str)]  # or true token length from your tokenizer
    )
    print(scores)