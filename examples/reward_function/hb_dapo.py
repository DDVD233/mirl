# Copyright 2025 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Optional, Any
import re
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
    """Extract content within \\boxed{}, [ ], or <answer>...</answer>."""
    for pattern in [
        r"\\boxed{([^}]*)}",      # \boxed{...}
        r"\[(.*?)\]",             # [ ... ]
        r"<answer>(.*?)</answer>" # <answer>...</answer>
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return text


def _parse_type_from_task_id(task_id: str) -> str:
    """
    Robustly extract the trailing task type from strings like:
      "chsimsv2_sentiment_intensity_cls", "intent_qa", "mime_qna", "foo_q&a"
    Rules:
      - Only the FINAL token matters.
      - Accept exactly: "cls", "qa", "qna", "q&a" (case-insensitive).
      - Do NOT treat "classification" (or similar) as "cls".
      - Default to "cls" if no recognized suffix is found.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        return "cls"

    s = task_id.strip().strip("_").lower()
    tokens = re.split(r"[_\W]+", s)
    last = tokens[-1] if tokens else ""

    if last == "cls":
        return "cls"
    if last in {"qa", "qna", "q&a"}:
        return "qa"
    return "cls"


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Exact string match (case-insensitive)."""
    return 1.0 if response == ground_truth else -1.0


def cosine_similarity_reward(pred_label: str, ground_truth: str, model) -> float:
    """
    Compute cosine similarity between two strings using embeddings.
    Returns scaled score in [-1, 1] to match DAPO's range.
    """
    embeddings = model.encode([pred_label, ground_truth], convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Keep in [-1, 1] range to match DAPO's accuracy_reward range
    return cos_sim


def soft_overlong_punishment(
    response_length: int,
    max_response_length: int,
    overlong_buffer_length: int,
) -> float:
    """
    Piecewise penalty in [-1, 0]:
      - If len <= (max - buffer):        0.0   (no penalty)
      - If (max - buffer) < len <= max:  linear from 0.0 down to -1.0
      - If len > max:                    -1.0  (full penalty)
    """
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    if response_length <= max_response_length:
        # Linearly decreases from 0 → -1 across the buffer window
        return (expected_len - response_length) / float(overlong_buffer_length)
    return -1.0


def hb_dapo_compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Any],
    task_ids: Optional[List[str]] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute scores for each response following DAPO pattern:
      - type=cls → exact string match (returns 1.0 or -1.0)
      - type=qa  → cosine similarity (returns value in [-1, 1])
      - No format score (to preserve DAPO's intuition)
      - Returns: score, accuracy_normalized, similarity_score, task_type

    Note: Overlong penalties are handled by the reward manager, not here.
    """
    if task_ids is None:
        # default all to CLS semantics if not provided
        task_ids = ["cls"] * len(solution_strs)

    assert len(solution_strs) == len(ground_truths) == len(task_ids), "Input length mismatch."

    # Check if we need cosine similarity
    need_cosine = any(_parse_type_from_task_id(tid) == "qa" for tid in task_ids)
    st_model = _ensure_st_model() if need_cosine else None

    batch_scores = []
    for i, (predict_str, ground_truth, task_id) in enumerate(zip(solution_strs, ground_truths, task_ids)):
        task_type = _parse_type_from_task_id(task_id)

        # Normalize tag spacing, then extract boxed content
        full_response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str or "")
        pred_label = extract_boxed_content(full_response).strip().lower()
        gt_norm = (ground_truth or "").strip().lower()

        # Compute accuracy based on task type
        if task_type == "cls":
            # Exact match: returns 1.0 or -1.0 (DAPO range)
            accuracy_score = accuracy_reward(pred_label, gt_norm)
            similarity_score = 0.0
        else:  # QA task
            # Cosine similarity: returns value in [-1, 1]
            similarity_score = cosine_similarity_reward(pred_label, gt_norm, st_model)
            accuracy_score = similarity_score

        # Normalize accuracy to [0, 1] range for tracking
        accuracy_normalized = 0.5 * (accuracy_score + 1.0)

        batch_scores.append({
            "score": accuracy_score,
            "accuracy_normalized": accuracy_normalized,
            "similarity_score": similarity_score,
            "task_type": task_type,
        })

    return batch_scores


if __name__ == "__main__":
    # Test examples
    cls_response = "<think>Reasoning.....</think>\\boxed{anger}"
    qa_response = "<think>Reasoning....</think>\\boxed{The Eiffel Tower is in Paris.}"
    qa_response_short = "<think>Reasoning....</think>\\boxed{good.}"

    scores = hb_dapo_compute_score(
        data_sources=["", "", ""],
        solution_strs=[cls_response, qa_response, qa_response_short],
        ground_truths=["anger", "The Eiffel Tower is located in Paris.", "bad."],
        extra_infos=["", "", ""],
        task_ids=["sen_intensity_data_cls", "intent_qa", "mime_qa"],
    )

    print("Test Results:")
    for i, score in enumerate(scores):
        print(f"\nExample {i+1}:")
        for key, value in score.items():
            print(f"  {key}: {value}")
