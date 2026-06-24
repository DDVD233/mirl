"""Rubric-based reward for HealthBench-Professional-style training data.

This scores an open-ended clinician-chat response against a set of
physician-style rubric criteria (each with a signed point value), exactly the
way HealthBench / HealthBench Professional grade:

    raw = sum(points of met criteria) / sum(positive points)          (can be <0)
    length_adjusted = raw - penalty_per_500 * (chars - center) / 500
    score = clip(length_adjusted, 0, 1)

A model-based grader (the same chat endpoint the medical judge uses) decides
met / not-met for each criterion independently. The rubric items travel on the
training row in ``extra_info["rubric_items"]`` (list of {criterion_text|criterion,
points}); they are produced by ``scripts/self_evolving/healthbench_gen.py``.

This module is dispatched from ``self_evolving.compute_score`` for any
``data_source`` starting with "healthbench" and returns the SAME key set as
``self_evolving`` / ``climb`` so mixed training batches stay homogeneous for
DataProto. It does NOT affect any other data source.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

# Length-adjustment constants from the HealthBench Professional paper.
LENGTH_ADJ_CENTER = float(os.environ.get("HB_LENGTH_CENTER", "2000"))
LENGTH_ADJ_PENALTY_PER_500 = float(os.environ.get("HB_LENGTH_PENALTY_PER_500", "0.0147"))

GRADER_SYSTEM = "You are a careful clinical grader. Return ONLY a JSON object."

GRADER_TEMPLATE = """\
You score the last assistant response in a clinician-AI conversation against a single rubric item.

# Conversation
{conversation}

# Rubric item
[{points}] {criterion}

# Instructions
Return ONLY a json object with fields "explanation" (string) and "criteria_met" (boolean).
- criteria_met is true iff the assistant response meets the criterion. Consider all parts of a
  multi-part criterion; if any part is not met, return false.
- "such as" / "for example" / "including" lists are illustrative — the response need not contain
  every listed example to meet the criterion.
- For criteria phrased as undesirable behavior (often with negative points), criteria_met is true
  ONLY if the response actually exhibits that undesirable behavior.
Return just the json, no markdown.
"""


def _rubric_items(extra_info: dict) -> list[dict]:
    items = extra_info.get("rubric_items")
    if items is None:
        rm = extra_info.get("reward_model") or {}
        items = rm.get("rubric_items")
    out = []
    for it in items or []:
        crit = it.get("criterion_text", it.get("criterion"))
        pts = it.get("points")
        if crit is None or pts is None:
            continue
        out.append({"criterion": str(crit), "points": float(pts)})
    return out


def _conversation_text(extra_info: dict, solution_str: str) -> str:
    """Render the clinician turn(s) + the model's response for the grader."""
    conv = extra_info.get("conversation")
    lines = []
    if isinstance(conv, list) and conv:
        for m in conv:
            if isinstance(m, dict) and m.get("content"):
                lines.append(f"{m.get('role', 'user')}: {m['content']}")
    else:
        q = extra_info.get("question") or extra_info.get("prompt") or ""
        if q:
            lines.append(f"user: {q}")
    lines.append(f"assistant: {solution_str}")
    return "\n\n".join(lines)


def _parse_met(text: str) -> bool | None:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", (text or "").strip())
    try:
        d = json.loads(cleaned)
    except Exception:
        m = re.search(r'"criteria_met"\s*:\s*(true|false)', cleaned, re.I)
        if m:
            return m.group(1).lower() == "true"
        return None
    val = d.get("criteria_met")
    return val if isinstance(val, bool) else None


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    api_base: str = "",
    api_key: str = "EMPTY",
    model_name: str = "",
    **kwargs,
) -> dict:
    extra_info = extra_info or {}
    items = _rubric_items(extra_info)
    response_text = solution_str or ""
    fmt_ok = 1.0 if response_text.strip() else 0.0

    # No rubric or no grader configured -> neutral, homogeneous result.
    total_pos = sum(it["points"] for it in items if it["points"] > 0)
    if not items or total_pos <= 0 or not api_base:
        return _result(0.0, 0.0, response_text)

    # Lazy import to avoid an import cycle (self_evolving imports us).
    from verl.utils.reward_score.self_evolving import _call_api

    conversation = _conversation_text(extra_info, response_text)

    async def grade(it: dict) -> float:
        prompt = GRADER_TEMPLATE.format(
            conversation=conversation, points=it["points"], criterion=it["criterion"]
        )
        try:
            raw = await _call_api(api_base, api_key, model_name, GRADER_SYSTEM, prompt, max_tokens=512)
        except Exception:
            return 0.0  # unreachable grader -> treat as not met (no credit)
        met = _parse_met(raw)
        return it["points"] if met else 0.0

    achieved = sum(await asyncio.gather(*[grade(it) for it in items]))
    raw = achieved / total_pos  # may be negative if negative criteria fire

    chars = len(response_text)
    length_adjusted = raw - LENGTH_ADJ_PENALTY_PER_500 * ((chars - LENGTH_ADJ_CENTER) / 500.0)
    return _result(raw, length_adjusted, response_text)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _result(raw: float, length_adjusted: float, response_text: str) -> dict:
    """Map the rubric signals onto the canonical key set shared with
    self_evolving / climb (homogeneous DataProto batches)."""
    score = _clip01(length_adjusted)
    raw01 = _clip01(raw)
    fmt_ok = 1.0 if (response_text or "").strip() else 0.0
    return {
        "score": float(score),               # training reward (length-adjusted, clipped)
        "acc": float(raw01),                  # headline rubric fraction / gen-server feedback
        "judge_acc_lenient": float(raw01),    # raw rubric fraction
        "judge_acc_strict": float(score),     # length-adjusted (the HB-Pro primary metric)
        "exact_acc": float(raw01),
        "answer_quality": 1.0 + 4.0 * raw01,  # 1-5 slot
        "reasoning_quality": 3.0,
        "format_ok": float(fmt_ok),
        "embed_sim": 0.0,
        "char_bleu": float(raw01),
        "extracted_answer": (response_text or "")[:512],
    }
