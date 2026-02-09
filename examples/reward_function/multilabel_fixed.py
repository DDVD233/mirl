from typing import List, Dict, Optional
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

# Load embedding model once
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def extract_boxed_content(text: str) -> str:
    """Extract content within \boxed{} or similar boxing notations."""
    patterns = [
        r"\\boxed{([^}]*)}",
        r"\[(.*?)\]",
        r"<answer>(.*?)</answer>",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return text

def format_reward(response: str) -> float:
    """
    Check whether the response matches the expected format.
    Uses DOTALL to handle multi-line reasoning and accounts for optional 
    leading/trailing whitespace.
    """
    # Restored logic to handle multi-line strings and basic structural checks
    pattern = re.compile(r".*<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.match(pattern, response) else 0.0

def accuracy_reward(pred: str, gt: str) -> float:
    return 1.0 if pred == gt else 0.0

def cosine_similarity_reward(pred: str, gt: str) -> float:
    emb = _EMBED_MODEL.encode([pred, gt], convert_to_numpy=True)
    v1 = emb[0] / np.linalg.norm(emb[0])
    v2 = emb[1] / np.linalg.norm(emb[1])
    return float(np.dot(v1, v2))

def _score_one(raw_response: str, gt: str) -> Dict[str, float]:
    # 1. Normalize response to handle Qwen quirks (extra spaces in tags)
    normalized_response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)
    
    # 2. Extract answer and clean up for comparison
    pred_content = extract_boxed_content(normalized_response)
    pred_clean = pred_content.lower().strip()
    gt_clean = gt.lower().strip()

    # 3. Compute component scores
    fmt = format_reward(normalized_response)
    acc = accuracy_reward(pred_clean, gt_clean)
    sim = cosine_similarity_reward(pred_clean, gt_clean)

    # 4. Final weighted score
    score = acc + 0.2 * fmt + 0.5 * sim

    return {
        "score": score,
        "standard_score": acc,
        "format_score": fmt,
        "similarity_score": sim,
    }

def human_behaviour_compute_score_batch(
    *,
    data_sources: Optional[List[str]] = None,
    solution_strs: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None,
    extra_infos: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    # Compatibility for single mode
    data_source: Optional[str] = None,
    solution_str: Optional[List[str]] = None,
    ground_truth: Optional[List[str]] = None,
    **kwargs
):
    # ---- SINGLE ITEM PATH ----
    if solution_strs is None and solution_str is not None:
        return _score_one(solution_str[0], ground_truth[0])

    # ---- BATCH PATH ----
    results = []
    for pred_text, gt_text in zip(solution_strs, ground_truths):
        scores = _score_one(pred_text, gt_text)
        results.append(scores)

    return results