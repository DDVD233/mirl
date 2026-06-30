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
import logging
import os
import random
import re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Length-adjustment constants from the HealthBench Professional paper.
LENGTH_ADJ_CENTER = float(os.environ.get("HB_LENGTH_CENTER", "2000"))
LENGTH_ADJ_PENALTY_PER_500 = float(os.environ.get("HB_LENGTH_PENALTY_PER_500", "0.0147"))

# Probability of printing a full per-item rubric grading trace (debug).
DEBUG_PRINT_PROB = float(os.environ.get("HB_DEBUG_PRINT_PROB", "0.0"))

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
    provider: str = "",
    val_api_base: str = "",
    val_api_key: str = "EMPTY",
    val_model_name: str = "",
    val_provider: str = "",
    **kwargs,
) -> dict:
    """Score an open-ended clinician-chat response against its co-generated rubric.

    The judge is chosen by training vs validation:
      - training (``extra_info["_is_validation"]`` falsy): self / server-5 judge
        (``api_base`` / ``model_name`` / ``provider``).
      - validation (``_is_validation`` true): the gpt-chat-latest TRAPI judge
        (``val_api_base`` / ``val_model_name`` / ``val_provider``), so the val
        metric matches the official HealthBench Professional protocol.

    Each rubric item is graded INDEPENDENTLY with a single binary judge call
    (``criteria_met`` true/false). The point arithmetic is done here in Python —
    the judge never sees the whole rubric and never sums points:
        achieved  = Σ points of met criteria (negative items subtract)
        total_pos = Σ positive points
        raw       = achieved / total_pos      (≈ achieved/10 since positives sum ~10)
        score     = clip(raw [- length penalty in val], 0, 1)
    """
    extra_info = extra_info or {}
    items = _rubric_items(extra_info)
    response_text = solution_str or ""

    is_val = bool(extra_info.get("_is_validation", False))
    if is_val and val_api_base:
        eff_base, eff_key, eff_model, eff_provider = (
            val_api_base, val_api_key, val_model_name, val_provider
        )
    else:
        eff_base, eff_key, eff_model, eff_provider = (
            api_base, api_key, model_name, provider
        )

    # No rubric or no grader configured -> neutral, homogeneous result.
    total_pos = sum(it["points"] for it in items if it["points"] > 0)
    if not items or total_pos <= 0 or not eff_base:
        return _result(0.0, 0.0, response_text)

    # Lazy import to avoid an import cycle (self_evolving imports us).
    from verl.utils.reward_score.self_evolving import _call_api

    conversation = _conversation_text(extra_info, response_text)

    async def grade(it: dict) -> tuple[float, bool | None]:
        prompt = GRADER_TEMPLATE.format(
            conversation=conversation, points=it["points"], criterion=it["criterion"]
        )
        try:
            raw = await _call_api(
                eff_base, eff_key, eff_model, GRADER_SYSTEM, prompt,
                max_tokens=512, provider=eff_provider,
            )
        except Exception:
            return 0.0, None  # unreachable grader -> treat as not met (no credit)
        met = _parse_met(raw)
        return (it["points"] if met else 0.0), met

    graded = await asyncio.gather(*[grade(it) for it in items])
    achieved = sum(pts for pts, _met in graded)
    raw = achieved / total_pos  # may be negative if negative criteria fire

    # Length adjustment is the HealthBench-Pro primary metric, applied ONLY for
    # validation (to match the official benchmark). The training reward is the
    # pure clipped rubric fraction so the policy isn't length-shaped by the proxy.
    chars = len(response_text)
    if is_val:
        length_adjusted = raw - LENGTH_ADJ_PENALTY_PER_500 * ((chars - LENGTH_ADJ_CENTER) / 500.0)
    else:
        length_adjusted = raw

    if random.random() < DEBUG_PRINT_PROB:
        mode = "VAL" if is_val else "train"
        print(f"\n{'=' * 60}")
        print(f"[RUBRIC REWARD {mode}] judge={eff_model}@{eff_base}  "
              f"use_case={extra_info.get('use_case', '?')}  n_items={len(items)}")
        for (pts, met), it in zip(graded, items):
            print(f"  [{it['points']:+.0f}] met={met}  {it['criterion'][:90]}")
        print(f"  achieved={achieved:.1f}  total_pos={total_pos:.1f}  "
              f"raw={raw:.3f}  len_adj={length_adjusted:.3f}  chars={chars}")
        print(f"{'=' * 60}\n")

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
