"""MedThinkVQA reward: exact match on the diagnosis letter.

The whole reason this dataset is worth running alongside HealthBench-Pro is that
the answer is one of five letters, so the reward is a string comparison. No judge,
no rubric, no per-criterion grading — which removes every failure mode that has
cost us runs so far: judge outages that look like bad answers, rubrics the
curriculum can quietly make easier, and a training signal that drifts because the
grader drifts. It also costs nothing per rollout.

FIVE OPTIONS MEANS A 20% FLOOR. Random guessing scores 0.2, and the train answer
distribution is not uniform (A 724 ... D 1866 of 7347), so always answering "D"
scores 0.254. Any accuracy claim has to clear those, and `guess_rate` is exported
so a policy that has learned the prior rather than the medicine is visible rather
than inferred.

FORMAT IS SCORED SEPARATELY FROM CORRECTNESS. `format_ok` says whether a parsable
letter was produced at all. Folding a formatting failure into "wrong answer"
hides the difference between a model that cannot diagnose and one that cannot
follow the output contract — early in RL the second dominates, and the two want
opposite fixes.
"""

from __future__ import annotations

import os
import re

# Accept the letter however the model chose to present it, but never guess for it.
# Order matters: \boxed{} is the contract, the rest are fallbacks in descending
# confidence, and a bare letter anywhere in the text is NOT accepted (a response
# discussing "option C" while concluding D would otherwise score as C).
_BOXED = re.compile(r"\\boxed\s*\{\s*([A-Ea-e])\s*(?:[.):]\s*[^}]*)?\}")
_ANSWER_IS = re.compile(
    r"(?:final\s+)?answer\s*(?:is)?\s*[:=]?\s*\(?\s*([A-Ea-e])\s*[).:\s]", re.I)
_OPTION_IS = re.compile(r"\boption\s*\(?\s*([A-Ea-e])\s*\)?\b", re.I)

VALID = "ABCDE"


def _strip_thinking(text: str) -> str:
    """Drop the private reasoning channel before parsing.

    Without this a model that explores "maybe \\boxed{B}" mid-thought and then
    concludes E is scored on the exploration. The reasoning is never graded.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def extract_letter(solution_str: str) -> tuple[str, str]:
    """Return (letter, how). letter is "" when nothing parsable was produced."""
    ans = _strip_thinking(solution_str or "")
    for rx, how in ((_BOXED, "boxed"), (_ANSWER_IS, "answer_is"), (_OPTION_IS, "option_is")):
        m = None
        for m in rx.finditer(ans):      # last match: the conclusion, not the setup
            pass
        if m:
            return m.group(1).upper(), how
    return "", "none"


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Exact-match reward with the diagnostics needed to interpret it."""
    extra_info = extra_info or {}
    gt = str(ground_truth or "").strip().upper()[:1]
    letter, how = extract_letter(solution_str)
    ok = bool(letter) and letter in VALID
    correct = 1.0 if (ok and letter == gt) else 0.0

    # A response with no parsable letter scores 0, but is counted separately so a
    # formatting collapse cannot be mistaken for a capability collapse.
    fmt_penalty = float(os.environ.get("MTV_FORMAT_PENALTY", "0.0"))
    score = correct - (0.0 if ok else fmt_penalty)

    ans = _strip_thinking(solution_str or "")
    return {
        "score": float(score),
        "acc": correct,
        "format_ok": 1.0 if ok else 0.0,
        # Rate at which the policy lands on the majority answer. If accuracy and
        # this rise together the model is learning the label prior, not the task.
        "guess_rate": 1.0 if letter == "D" else 0.0,
        "parsed_boxed": 1.0 if how == "boxed" else 0.0,
        "answer_chars": float(len(ans)),
        "think_chars": float(max(0, len(solution_str or "") - len(ans))),
        "think_closed": 1.0 if "</think>" in (solution_str or "") else 0.0,
        "n_images_used": float(extra_info.get("n_images_used", 0) or 0),
        "n_images_total": float(extra_info.get("n_images_total", 0) or 0),
        "extracted_answer": (letter or "?"),
    }
