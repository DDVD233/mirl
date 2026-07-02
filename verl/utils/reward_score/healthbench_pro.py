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

# TRAINING-ONLY anti-runaway-thinking shaping. The solver always thinks (the chat
# template opens the reasoning channel), so a response with no ``</think>`` means
# the reasoning ate the whole token budget and there IS no answer. Analysis of run
# ij3ccucc showed this — not answer quality — drove the val decline (unclosed items
# score ~-0.55 length-adjusted because the raw reasoning dump got graded).
#  - unclosed thinking -> fixed score, NO judge calls (grading a dump is meaningless);
#  - think text beyond a free budget pays a small linear penalty, so the policy
#    learns to bound its reasoning instead of drifting toward the cap.
# Validation is untouched (official HealthBench-Pro protocol, comparable across runs).
HB_UNCLOSED_THINK_SCORE = float(os.environ.get("HB_UNCLOSED_THINK_SCORE", "0.0"))
HB_THINK_FREE_CHARS = float(os.environ.get("HB_THINK_FREE_CHARS", "10000"))
HB_THINK_PENALTY_PER_1K = float(os.environ.get("HB_THINK_PENALTY_PER_1K", "0.02"))
HB_THINK_PENALTY_MAX = float(os.environ.get("HB_THINK_PENALTY_MAX", "0.3"))

# Probability of printing a full per-item rubric grading trace (debug).
DEBUG_PRINT_PROB = float(os.environ.get("HB_DEBUG_PRINT_PROB", "0.0"))

# Grader system message. The official simple-evals HealthBench grader sends a
# single user message (no system); we add a minimal neutral system so the contract
# is explicit for local judges. It does not alter the grading rules below.
GRADER_SYSTEM = "You are a careful grader. Return only the requested JSON object."

# VERBATIM official simple-evals HealthBench GRADER_TEMPLATE
# (openai/simple-evals healthbench_eval.py). Placeholders <<conversation>> and
# <<rubric_item>> are filled by str.replace (the template contains literal JSON
# braces, so str.format must not be used). A rubric item renders as "[points] criterion"
# exactly like RubricItem.__str__. This is what makes our training/val grader follow
# the HealthBench Professional paper.
GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""


def _strip_thinking(text: str) -> str:
    """Return only the final answer, dropping any reasoning/thinking channel.

    The solver runs with thinking ENABLED, so its response is
    ``<think> ... </think> <answer>`` (or, if the chat template streams reasoning
    without an open tag, ``... </think> <answer>``). We grade and length-measure
    ONLY the answer:
      - if a ``</think>`` close tag exists, take everything after the LAST one;
      - else if an unclosed ``<think>`` exists (truncated reasoning, no answer),
        drop it (treated as an empty answer -> low score, which is correct);
      - else return the text unchanged.
    """
    if not text:
        return ""
    low = text.lower()
    close = low.rfind("</think>")
    if close != -1:
        return text[close + len("</think>"):].strip()
    open_i = low.find("<think>")
    if open_i != -1:
        return text[:open_i].strip()
    return text.strip()


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
        score     = clip(raw - length penalty, 0, 1)

    The solver runs with THINKING ENABLED, so its raw output is
    ``<think>...</think> answer``. Grading and length-measurement use ONLY the
    answer (thinking stripped): the judge sees just the final answer, and the
    length penalty counts only the answer's characters — so reasoning length is
    never rewarded or penalized. The length penalty (HealthBench-Pro style) is
    applied for BOTH training and validation.
    """
    extra_info = extra_info or {}
    items = _rubric_items(extra_info)
    response_text = solution_str or ""
    # Strip the thinking channel: the judge grades — and length counts — only the
    # final answer, not the reasoning.
    answer_text = _strip_thinking(response_text)
    think_closed = "</think>" in response_text.lower()
    think_chars = max(0, len(response_text) - len(answer_text))

    is_val = bool(extra_info.get("_is_validation", False))

    # Runaway thinking (training only): reasoning never closed -> no answer exists.
    # Fixed score, no judge calls. Val keeps the official grading path untouched.
    if not is_val and not think_closed:
        return _result(HB_UNCLOSED_THINK_SCORE, HB_UNCLOSED_THINK_SCORE, "",
                       think_closed=False, think_chars=len(response_text))
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
        return _result(0.0, 0.0, answer_text,
                       think_closed=think_closed, think_chars=think_chars)

    # Lazy import to avoid an import cycle (self_evolving imports us).
    from verl.utils.reward_score.self_evolving import _call_api

    conversation = _conversation_text(extra_info, answer_text)

    async def grade(it: dict) -> tuple[float, bool | None]:
        # Render exactly like official simple-evals: replace <<conversation>> and
        # <<rubric_item>> (rubric item = "[points] criterion", RubricItem.__str__).
        rubric_item = f"[{it['points']:g}] {it['criterion']}"
        prompt = (GRADER_TEMPLATE
                  .replace("<<conversation>>", conversation)
                  .replace("<<rubric_item>>", rubric_item))
        try:
            raw = await _call_api(
                eff_base, eff_key, eff_model, GRADER_SYSTEM, prompt,
                max_tokens=512, provider=eff_provider,
            )
        except Exception:
            return 0.0, None  # unreachable grader -> treat as not met (no credit)
        # Defensive: if the judge itself emits a thinking channel, keep only the
        # main text before parsing the verdict JSON.
        met = _parse_met(_strip_thinking(raw))
        return (it["points"] if met else 0.0), met

    graded = await asyncio.gather(*[grade(it) for it in items])
    achieved = sum(pts for pts, _met in graded)
    raw = achieved / total_pos  # may be negative if negative criteria fire
    # Per-criterion verdicts, forwarded (as JSON) to the gen server's /evolve loop
    # so the meta-optimizer sees WHICH criteria failed, not just the total score.
    rubric_met = [
        {"criterion": it["criterion"], "points": it["points"], "met": met}
        for (_pts, met), it in zip(graded, items)
    ]

    # HealthBench-Pro length adjustment on the ANSWER length (thinking stripped),
    # applied for both training and validation so longer answers pay the same
    # penalty the benchmark uses.
    chars = len(answer_text)
    length_adjusted = raw - LENGTH_ADJ_PENALTY_PER_500 * ((chars - LENGTH_ADJ_CENTER) / 500.0)

    # Training-only thinking-budget penalty: reasoning beyond the free budget pays
    # a small linear cost (capped) so bounded thinking is preferred to drift.
    if not is_val and think_chars > HB_THINK_FREE_CHARS:
        length_adjusted -= min(
            HB_THINK_PENALTY_MAX,
            HB_THINK_PENALTY_PER_1K * (think_chars - HB_THINK_FREE_CHARS) / 1000.0,
        )

    if random.random() < DEBUG_PRINT_PROB:
        mode = "VAL" if is_val else "train"
        print(f"\n{'=' * 60}")
        print(f"[RUBRIC REWARD {mode}] judge={eff_model}@{eff_base}  "
              f"use_case={extra_info.get('use_case', '?')}  n_items={len(items)}")
        for (pts, met), it in zip(graded, items):
            print(f"  [{it['points']:+.0f}] met={met}  {it['criterion'][:90]}")
        print(f"  achieved={achieved:.1f}  total_pos={total_pos:.1f}  "
              f"raw={raw:.3f}  len_adj={length_adjusted:.3f}  answer_chars={chars} "
              f"(resp_chars={len(response_text)})")
        print(f"{'=' * 60}\n")

    return _result(raw, length_adjusted, answer_text,
                   think_closed=think_closed, think_chars=think_chars,
                   rubric_met=rubric_met)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _result(raw: float, length_adjusted: float, response_text: str,
            think_closed: bool = True, think_chars: int = 0,
            rubric_met: list | None = None) -> dict:
    """Map the rubric signals onto the canonical key set shared with
    self_evolving / climb (homogeneous DataProto batches)."""
    score = _clip01(length_adjusted)
    raw01 = _clip01(raw)
    fmt_ok = 1.0 if (response_text or "").strip() else 0.0
    return {
        "score": float(score),               # training reward (length-adjusted, clipped [0,1])
        # Both headline numbers, reported every step so validation shows raw AND
        # length-adjusted accuracy side by side:
        "acc_raw": float(raw01),              # raw rubric fraction (no length penalty), clipped
        "acc_len_adj": float(score),          # length-adjusted (HB-Pro primary metric), clipped
        # UNCLIPPED (signed) variants — the official simple-evals metric does NOT
        # clip per example (negative criteria can push a response below 0), so the
        # mean of these is the paper-comparable number:
        "acc_raw_signed": float(raw),
        "acc_len_adj_signed": float(length_adjusted),
        "acc": float(raw01),                  # raw (gen-server difficulty feedback uses this)
        "judge_acc_lenient": float(raw01),    # raw rubric fraction
        "judge_acc_strict": float(score),     # length-adjusted
        "exact_acc": float(raw01),
        "answer_quality": 1.0 + 4.0 * raw01,  # 1-5 slot
        "reasoning_quality": 3.0,
        "format_ok": float(fmt_ok),
        "embed_sim": 0.0,
        "char_bleu": float(raw01),
        "extracted_answer": (response_text or "")[:512],
        # Thinking telemetry (wandb: reward/think_closed/mean = closure rate) and
        # per-criterion verdicts for the /evolve loop. Only healthbench batches
        # carry these keys; batches in this run are healthbench-only.
        "think_closed": 1.0 if think_closed else 0.0,
        "think_chars": float(think_chars),
        "rubric_met": json.dumps(rubric_met or []),
    }
