# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Deterministic reward scorer for CLIMB multimodal classification.

CLIMB answers are label phrase(s) drawn from a fixed per-question option list
(single-label for most modalities, multi-label for chest X-ray). Scoring is a
cheap, API-free set comparison — no LLM judge — so it is identical between
training reward and validation, and fast enough to run on every rollout.

IMPORTANT — return-key contract: this function returns the EXACT same dict keys
as ``verl.utils.reward_score.self_evolving.compute_score``. A training batch can
mix CLIMB rows (``data_source`` ``climb_*``) and text rows (``self_evolving``),
and verl's DataProto requires every row in a batch to carry the same non-tensor
keys. Introducing a key only on CLIMB rows would make the batch ragged and trip
DataProto's length assertion, so partial-credit signals are mapped onto the
existing reward slots rather than added as new keys. The per-modality F1 the
evaluation reports is computed downstream in
``verl/trainer/ppo/climb_metrics.py`` from ``extracted_answer`` + the ground
truth, so no extra reward key is needed.
"""

import re

from verl.utils.climb import exact_match, f1_set, parse_label_set
from verl.utils.reward_score.self_evolving import extract_boxed_answer


def _extract_prediction(solution_str: str) -> tuple[str, bool]:
    """Return (predicted-answer-text, had_boxed). Prefer the \\boxed{} answer;
    fall back to the last non-empty line so a missing box still scores."""
    boxed = extract_boxed_answer(solution_str)
    if boxed is not None:
        return boxed, True
    lines = [ln.strip() for ln in re.split(r"[\n\r]+", solution_str or "") if ln.strip()]
    return (lines[-1] if lines else ""), False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Score a CLIMB classification response.

    - acc / exact_acc: 1.0 iff the predicted label set exactly equals the
      ground-truth set (perfect match — the headline per-modality accuracy).
    - sample F1: token-set F1 between predicted and gt labels (partial credit
      for multi-label cases), surfaced through the lenient/quality slots.
    - score: blended training reward = 0.5*exact + 0.4*F1 + 0.1*format.
    """
    extra_info = extra_info or {}
    options = extra_info.get("options") or None

    pred_text, had_boxed = _extract_prediction(solution_str)
    pred_set = parse_label_set(pred_text, options)
    gt_set = parse_label_set(ground_truth, options)

    em = exact_match(pred_set, gt_set)
    f1 = f1_set(pred_set, gt_set)
    format_ok = 1.0 if had_boxed else 0.0
    score = 0.5 * em + 0.4 * f1 + 0.1 * format_ok

    # Keys mirror self_evolving.compute_score exactly (see module docstring).
    return {
        "score": float(score),
        "acc": float(em),                 # headline accuracy / gen-server feedback
        "judge_acc_lenient": float(f1),   # partial-credit (F1) signal
        "judge_acc_strict": float(em),
        "exact_acc": float(em),
        "answer_quality": 5.0 * f1,       # 1-5 scale slot, reused for F1
        "reasoning_quality": 3.0,
        "format_ok": float(format_ok),
        "embed_sim": 0.0,
        "char_bleu": float(f1),
        "extracted_answer": pred_text or "",
    }
