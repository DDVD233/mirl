from typing import List, Dict
import re

def format_reward(response: str) -> float:
    """
    Check whether the response matches the expected format.
    Here we require something like <think>...</think> ... \boxed{...}
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Simple accuracy: exact match to ground truth string.
    """
    return 1.0 if response == ground_truth else 0.0

def human_behaviour_compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[str],
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute human behaviour scoring for batch inputs.

    Args:
        data_sources: List of data sources (unused here, but kept for interface compatibility)
        solution_strs: List of model prediction strings
        ground_truths: List of ground truth strings
        extra_infos: List of extra information (unused here, kept for compatibility)

    Returns:
        List of score dictionaries
    """
    batch_scores = []
    format_weight = 0.1

    for data_source, predict_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        # Normalize response formatting (e.g., qwen2.5vl quirks)
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)

        # Compute individual components
        format_score = format_reward(response)
        standard_score = accuracy_reward(response, ground_truth)

        # Weighted overall score
        overall_score = (1 - format_weight) * standard_score + format_weight * format_score

        scores = {
            "score": overall_score,
            "standard_score": standard_score,
            "format_score": format_score,
        }
        batch_scores.append(scores)

    return batch_scores
