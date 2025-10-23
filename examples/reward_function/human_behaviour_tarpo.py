from typing import List, Dict, Optional
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Lazy initialization for SentenceTransformer
_sentence_transformer_loaded = False
_STModel: Optional["SentenceTransformer"] = None


def _ensure_st_model():
    """Load SentenceTransformer only once (lazy load)."""
    global _sentence_transformer_loaded, _STModel
    if not _sentence_transformer_loaded:
        _STModel = SentenceTransformer('all-MiniLM-L6-v2')
        _sentence_transformer_loaded = True
    return _STModel


def extract_boxed_content(text: str) -> str:
    """Extract content within \boxed{}, [ ], or <answer>...</answer>."""
    for pattern in [
        r"\\boxed{([^}]*)}",      # \boxed{...}
        r"\[(.*?)\]",             # [ ... ]
        r"<answer>(.*?)</answer>" # <answer>...</answer>
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return text


def format_reward(response: str) -> float:
    """Reward 1.0 if format matches expected <think>...</think> ... \boxed{...}."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Exact string match (case-insensitive handled externally)."""
    return 1.0 if response == ground_truth else 0.0

def cosine_similarity_reward(pred_label: str, ground_truth: str, model) -> float:
    """
    Compute cosine similarity between two strings using embeddings.
    Returns scaled score in [0, 1].
    """
    import numpy as np

    embeddings = model.encode([pred_label, ground_truth], convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Scale from [-1, 1] → [0, 1]
    return (cos_sim + 1.0) / 2.0


def _parse_type_from_task_id(task_id: str) -> str:
    """Extract '<type>' from '<task>_<type>'."""
    _, _, tail = task_id.rpartition("_")
    t = (tail if _ else task_id).strip().lower()
    if t in {"cls", "classification"}:
        return "cls"
    if t in {"qa", "qna", "q&a"}:
        return "qa"
    return "cls"


def human_behaviour_compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[str],
    task_ids: List[str],
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute scores for each response:
      - type=cls → exact string match only (no cosine)
      - type=qa  → cosine similarity only
      - always includes format score
    """
    assert len(solution_strs) == len(ground_truths) == len(task_ids), "Input length mismatch."

    format_weight = 0.2

    need_cosine = any(_parse_type_from_task_id(tid) == "qa" for tid in task_ids)
    st_model = _ensure_st_model() if need_cosine else None

    batch_scores = []
    for predict_str, ground_truth, task_id in zip(solution_strs, ground_truths, task_ids):
        task_type = _parse_type_from_task_id(task_id)

        full_response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)
        pred_label = extract_boxed_content(full_response).strip().lower()
        gt_norm = ground_truth.strip().lower()

        format_score = format_reward(full_response)

        if task_type == "cls":
            label_score = accuracy_reward(pred_label, gt_norm)
            label_weight = 1.0 - format_weight
            overall_score = label_weight * label_score + format_weight * format_score
            similarity_score = 0.0
        else:  # QA task
            similarity_score = cosine_similarity_reward(pred_label, gt_norm, st_model)
            similarity_weight = 1.0 - format_weight
            overall_score = similarity_weight * similarity_score + format_weight * format_score
            # placeholders 
            label_score = 0.0

        batch_scores.append({
            "score": overall_score,
            "standard_score": label_score,
            "format_score": format_score,
            "similarity_score": similarity_score,
            "task_type": task_type,
        })

    return batch_scores


if __name__ == "__main__":
    cls_response = "<think>Reasoning.....</think>\\boxed{anger}"
    qa_response = "<think>Reasoning....</think>\\boxed{The Eiffel Tower is in Paris.}"
    qa_response_two = "<think>Reasoning....</think>\\boxed{good.}"

    scores = human_behaviour_compute_score_batch(
        data_sources=["", "", ""],
        solution_strs=[cls_response, qa_response, qa_response_two],
        ground_truths=["anger", "The Eiffel Tower is located in Paris.", "bad."],
        extra_infos=["", "", ""],
        task_ids=["sen_cls", "intent_qa", "mime_qa"]
    )
    print(scores)
