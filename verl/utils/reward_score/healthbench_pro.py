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
``data_source`` starting with "healthbench". It does NOT affect any other data
source.

KEY-SET HOMOGENEITY (read before adding a return key). The trainer snapshots
``reward_extra_keys`` from **sample 0 only** (reward_loop.py / agent_loop.py:
``keys = list(infos[0].keys())`` then ``[info[k] for info in infos]``). So a key
present on sample 0 but missing on sample k is a ``KeyError`` that kills the step,
and a key missing on sample 0 is silently dropped for the whole batch. The rule
that makes this structural rather than reviewed:

    **Never build a result dict outside ``_result()``. New keys are added as
    ``_result`` parameters with defaults.**

``HOMOGENEOUS_DEFAULTS`` below is the shared floor spliced into the other
``self_evolving`` branches so genuinely mixed batches stack.
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

# Distinct (val/train, provider, base, model) judge configs already logged (log once each).
_JUDGE_LOGGED: set = set()

# Cumulative count of criteria that could not be graded at all (see D3). Kept for
# the log line; the ACTIONABLE signal is the per-sample `judge_fail` reward key,
# because a process-global counter never reaches wandb and a judge outage is
# otherwise indistinguishable from a batch of genuinely bad answers.
_JUDGE_FAILURES = [0]

# Every numeric key any branch of self_evolving.compute_score may return, with a
# neutral default. Splice into the branches that do not go through `_result` so a
# mixed batch (healthbench + mimic/climb rows) stacks instead of raising KeyError
# on the sample-0 key snapshot. See the module docstring.
HOMOGENEOUS_DEFAULTS: dict = {
    "acc_raw": 0.0,
    "acc_len_adj": 0.0,
    "acc_raw_signed": 0.0,
    "acc_len_adj_signed": 0.0,
    "exact_acc": 0.0,
    "think_closed": 1.0,
    "think_chars": 0.0,
    "rubric_met": "[]",
    "judge_fail": 0.0,
    # Retrieval telemetry / reward (populated by the retrieval agent loop; 0 when
    # the rollout did not retrieve or the run has retrieval disabled).
    "retrieval_used": 0.0,
    "retrieval_coverage": 0.0,
    "retrieval_judged": 0.0,
    "retrieval_bonus": 0.0,
    "n_search": 0.0,
    "n_queries": 0.0,
    "retrieval_hits": 0.0,
    "retrieval_error": 0.0,
    "retrieval_truncated": 0.0,
    "answer_rescued": 0.0,
    "budget_exhausted": 0.0,
    "retrieval_ctx_chars": 0.0,
    # self_evolving evolve add-on (returned only when evolve_on, which is decided
    # PER SAMPLE inside an exception handler -> same latent KeyError class).
    "dynamic_judge": 0.0,
    "dynamic_function": 0.0,
}

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
# ---- retrieval-quality reward (TRAINING ONLY) ----
# Weight of the second graded component. The bonus is ADDITIVE and CENTERED, never
# a normalized weighted sum: with norm_adv_by_std_in_grpo=False (Dr.GRPO) dividing
# by (1+w) would silently scale every advantage by 1/(1+w) — a learning-rate cut
# that would be misread as "the retrieval term hurt".
HB_RETRIEVAL_WEIGHT = float(os.environ.get("HB_RETRIEVAL_WEIGHT", "0.0"))
# 1 = center each rollout's coverage on the mean coverage of the OTHER searching
# rollouts in its GRPO group (folded in the trainer, where uid exists). That makes
# the deltas sum to ~0 within the group, so the search/no-search decision stays
# arbitrated by the answer reward alone and coverage only ranks QUERY QUALITY among
# rollouts that did search. 0 = fold here against the fixed counterfactual below,
# for paths with no groups.
HB_RETRIEVAL_GROUP_BASELINE = os.environ.get("HB_RETRIEVAL_GROUP_BASELINE", "1") == "1"
# Counterfactual coverage attributed to NOT searching. Only used as the fallback
# baseline when a group has fewer than 2 searching rollouts. Set it near the
# observed reward/retrieval_coverage/mean; too high trains the model away from
# searching on exactly the hard-knowledge tasks retrieval exists for (~26% of
# criteria are facts the KB simply does not hold, where even a perfect query scores 0).
HB_RETRIEVAL_NOSEARCH_COVERAGE = float(os.environ.get("HB_RETRIEVAL_NOSEARCH_COVERAGE", "0.35"))

HB_UNCLOSED_THINK_SCORE = float(os.environ.get("HB_UNCLOSED_THINK_SCORE", "0.0"))
HB_THINK_FREE_CHARS = float(os.environ.get("HB_THINK_FREE_CHARS", "10000"))
HB_THINK_PENALTY_PER_1K = float(os.environ.get("HB_THINK_PENALTY_PER_1K", "0.02"))
HB_THINK_PENALTY_MAX = float(os.environ.get("HB_THINK_PENALTY_MAX", "0.3"))
# TRAINING-ONLY floor of the reward ("score"). v4 post-mortem: with the floor at 0,
# every response whose length penalty exceeds its rubric fraction scores exactly 0 —
# "slightly bad" and "catastrophic" become indistinguishable, so once the policy
# drifts long there is no gradient ordering to bring it back, and whole GRPO groups
# go zero-variance (v4 ended with clip_ratio 1.0, entropy 3e-5, every group
# identical). A negative floor (e.g. -0.5) keeps bad rollouts ORDERED. Validation
# metrics are untouched (signed variants are already reported separately).
# Set HB_SCORE_MIN=none (or off/inf/-inf) to remove the training floor ENTIRELY, so the
# reward is never clipped from below and every rollout keeps its true ordering however bad
# it is. That is the strongest form of the fix above: a floor at -0.5 still collapses
# everything worse than -0.5, and the observed worst val task reaches -2.06.
#
# The cost to watch: with norm_adv_by_std_in_grpo=False the advantage is score minus the
# group mean, so a single catastrophic rollout at -3 against peers near +0.4 produces an
# advantage ~10x the typical magnitude. That is a real gradient spike, not a clipping
# artifact -- if actor/grad_norm or entropy destabilises, put a floor back rather than
# re-clipping silently.
_score_min_raw = (os.environ.get("HB_SCORE_MIN", "0.0") or "").strip().lower()
if _score_min_raw in ("none", "off", "inf", "-inf", "disabled"):
    HB_SCORE_MIN = float("-inf")
else:
    HB_SCORE_MIN = float(_score_min_raw)

# Repetition penalty (training only; see the v16 note in compute_score).
# `dup fraction` = share of 8-gram positions that repeat an earlier 8-gram.
HB_REP_FREE_FRAC = float(os.environ.get("HB_REP_FREE_FRAC", "0.15"))
HB_REP_PENALTY_SLOPE = float(os.environ.get("HB_REP_PENALTY_SLOPE", "1.0"))
HB_REP_PENALTY_MAX = float(os.environ.get("HB_REP_PENALTY_MAX", "0.5"))
HB_REP_LOOP_FRAC = float(os.environ.get("HB_REP_LOOP_FRAC", "0.5"))

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


# Tool-use turns (retrieval agent loop) leave ``<tool_call>``/``<tool_response>``
# spans in the decoded response, because the reward manager decodes by
# attention_mask (loss-masked tool tokens still appear). Strip them before the
# thinking split so neither the retrieval query nor the retrieved passages leak
# into the graded / length-counted answer.
# The closing tag is OPTIONAL (``|\Z``). A rollout that hits the response cap
# mid-``<tool_response>`` leaves an unterminated span; requiring the close tag left
# the entire raw passage block in the "answer", where it was counted as thinking
# chars, fed to the repetition penalty (retrieved passages share 8-grams across
# sources, so `_dup_ngram_frac` could exceed HB_REP_LOOP_FRAC and floor the reward),
# and — with no later ``</think>`` — could be graded AS the answer. Single
# continuous retrieval trajectories make truncation mid-span routine.
_TOOL_SPAN_RE = re.compile(
    r"<tool_call>.*?(?:</tool_call>|\Z)|<tool_response>.*?(?:</tool_response>|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _strip_tool_spans(text: str) -> str:
    """Remove ``<tool_call>...</tool_call>`` and ``<tool_response>...</tool_response>``."""
    if not text:
        return ""
    return _TOOL_SPAN_RE.sub("", text)


def _strip_thinking(text: str) -> str:
    """Return only the final answer, dropping reasoning and tool-use turns.

    The solver runs with thinking ENABLED (and, in retrieval mode, tool use), so a
    response looks like ``<think>..</think><tool_call>..</tool_call>`` then a
    tool turn then ``<think>..</think> <answer>``. We grade and length-measure
    ONLY the final answer:
      - tool-call / tool-response spans are stripped first;
      - if a ``</think>`` close tag exists, take everything after the LAST one;
      - else if an unclosed ``<think>`` exists (truncated reasoning, no answer),
        drop it (treated as an empty answer -> low score, which is correct);
      - else return the text unchanged.
    """
    if not text:
        return ""
    text = _strip_tool_spans(text)
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


def _task_text(extra_info: dict) -> str:
    """The clinician request alone (no model response) — what the coverage judge
    needs to decide whether the retrieved evidence is on target."""
    conv = extra_info.get("conversation")
    if isinstance(conv, list) and conv:
        parts = [str(m.get("content") or "") for m in conv
                 if isinstance(m, dict) and m.get("role") == "user"]
        if parts:
            return "\n\n".join(parts)
    return str(extra_info.get("question") or extra_info.get("prompt") or "")


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
    fallback_api_base: str = "",
    fallback_api_key: str = "EMPTY",
    fallback_model_name: str = "",
    fallback_provider: str = "",
    cov_api_base: str = "",
    cov_api_key: str = "EMPTY",
    cov_model_name: str = "",
    cov_provider: str = "",
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
    # The trained trajectory is not always the graded one. The retrieval agent loop
    # trains a fraction of rollouts on the phase-1 retrieval DECISION (a tool call),
    # while the thing worth scoring is still the answer that decision led to; the
    # loop passes that answer through `graded_answer`.
    #
    # Only the ANSWER is overridden. think_closed / think_chars stay derived from the
    # generated response, because they describe what the policy actually produced —
    # and because `graded_answer` arrives already think-stripped, using it here would
    # mark every decision sample as unclosed-think and hand it the fixed penalty
    # score, teaching the model that searching is always wrong.
    graded_answer = extra_info.get("graded_answer")
    if isinstance(graded_answer, str) and graded_answer.strip():
        answer_text = graded_answer
    # Count only reasoning chars toward the think-length penalty — exclude
    # tool-call/tool-response spans (retrieval mode) so retrieved passages are
    # neither rewarded nor penalized as "thinking".
    think_chars = max(0, len(_strip_tool_spans(response_text)) - len(answer_text))

    # Validation detection. The `_is_validation` meta_info flag set by the trainer does
    # NOT reliably survive the async reward-loop's Ray dispatch (chunk -> .remote ->
    # worker), so grading silently fell back to the TRAIN self-judge for val (bug: val
    # graded with Qwen, never TRAPI). Route robustly by data_source instead: the REAL
    # HealthBench-Professional eval set is always `healthbench_professional/*`; the
    # self-generated training tasks are `healthbench_self`. So the official data always
    # gets the official (TRAPI) judge regardless of the flag.
    is_val = bool(extra_info.get("_is_validation", False)) or str(data_source or "").startswith(
        "healthbench_professional"
    )
    # `is_val` above is the GRADING-PROTOCOL switch (judge routing, length
    # adjustment, benchmark clip) and must keep exactly that meaning. It is NOT a
    # usable "is this a validation rollout?" test: on a train-on-val run
    # (train_files == val_files == healthbench_professional/*) it is True for every
    # TRAINING row too. Anything training-only — the retrieval reward, the coverage
    # judge — must gate on `is_train`, which reads the real trainer flag (now
    # delivered via meta_info, see agent_loop._compute_score).
    is_train = not bool(extra_info.get("_is_validation", False))

    # Retrieval telemetry from the agent loop. These arrive only inside the
    # `tool_extra_fields` object column and are copied into extra_info by the reward
    # manager; they are pure passthrough here so every sample carries every key.
    rtel = {k: _as_float(extra_info.get(k)) for k in (
        "n_search", "n_queries", "retrieval_hits", "retrieval_error",
        "retrieval_truncated", "answer_rescued", "budget_exhausted",
    )}
    retrieval_context = extra_info.get("retrieval_context") or ""
    rtel["retrieval_used"] = 1.0 if str(retrieval_context).strip() else 0.0
    rtel["retrieval_ctx_chars"] = float(len(str(retrieval_context)))

    # Runaway thinking (training only): reasoning never closed -> no answer exists.
    # Fixed score, no judge calls. Val keeps the official grading path untouched.
    if not is_val and not think_closed:
        return _result(HB_UNCLOSED_THINK_SCORE, HB_UNCLOSED_THINK_SCORE, "",
                       think_closed=False, think_chars=len(response_text), is_val=False,
                       **rtel)
    if is_val and val_api_base:
        eff_base, eff_key, eff_model, eff_provider = (
            val_api_base, val_api_key, val_model_name, val_provider
        )
    else:
        eff_base, eff_key, eff_model, eff_provider = (
            api_base, api_key, model_name, provider
        )
    # Confirm WHICH judge is actually used (once per distinct config) so we can verify
    # validation really grades with the TRAPI gpt-5.x judge and not the vllm/kimi self-judge.
    _sig = ("VAL" if is_val else "TRAIN", eff_provider, eff_base, eff_model)
    if _sig not in _JUDGE_LOGGED:
        _JUDGE_LOGGED.add(_sig)
        logger.warning("RUBRIC JUDGE [%s]: provider=%s base=%s model=%s",
                       _sig[0], eff_provider or "(default)", eff_base, eff_model)

    # No rubric or no grader configured -> neutral, homogeneous result.
    total_pos = sum(it["points"] for it in items if it["points"] > 0)
    if not items or total_pos <= 0 or not eff_base:
        return _result(0.0, 0.0, answer_text,
                       think_closed=think_closed, think_chars=think_chars, is_val=is_val,
                       **rtel)

    # Lazy import to avoid an import cycle (self_evolving imports us).
    from verl.utils.reward_score.self_evolving import _call_api

    conversation = _conversation_text(extra_info, answer_text)

    # Per-SAMPLE ungradable-criterion count. Exported as the `judge_fail` reward key
    # so `reward/judge_fail/mean` makes a judge outage one glance instead of a log
    # grep; the module-global _JUDGE_FAILURES never reaches wandb.
    fails = [0]

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
        except Exception as e:
            # DO NOT silently swallow judge failures — a struggling/misconfigured judge
            # (e.g. TRAPI auth/rate-limit) otherwise looks like a stream of "not met"
            # verdicts with no error. Log it loudly (with endpoint/model) so it's visible.
            logger.warning(
                "JUDGE CALL FAILED (%s) provider=%s base=%s model=%s: %s: %s",
                "VAL" if is_val else "TRAIN", eff_provider, eff_base, eff_model,
                type(e).__name__, e,
            )
            # FALLBACK judge (e.g. the local teacher when the primary is a remote
            # proxy). Without this, a proxy outage silently zeroes EVERY reward in
            # the batch — for the TRAIN judge that trains the policy on garbage,
            # which is far worse than a blind val point. Only used on failure.
            if fallback_api_base:
                try:
                    raw = await _call_api(
                        fallback_api_base, fallback_api_key, fallback_model_name,
                        GRADER_SYSTEM, prompt, max_tokens=512,
                        provider=fallback_provider,
                    )
                    logger.warning("judge fallback OK -> %s@%s", fallback_model_name,
                                   fallback_api_base)
                    met = _parse_met(_strip_thinking(raw))
                    return (it["points"] if met else 0.0), met
                except Exception as e2:
                    logger.warning("judge FALLBACK also failed: %s: %s",
                                   type(e2).__name__, e2)
            # D3 (2026-08-04 audit): an ungradable criterion is indistinguishable
            # from a genuine "not met" in the reward, so a judge outage looks like
            # a batch of bad answers. Count it loudly so the rate is checkable.
            _JUDGE_FAILURES[0] += 1
            fails[0] += 1
            logger.error("judge UNGRADABLE (cumulative=%d) -> scoring 0 points",
                         _JUDGE_FAILURES[0])
            return 0.0, None  # grader unreachable -> treat as not met (no credit)
        # Defensive: if the judge itself emits a thinking channel, keep only the
        # main text before parsing the verdict JSON.
        met = _parse_met(_strip_thinking(raw))
        if met is None:
            # The judge answered but unparseably — also an ungradable criterion,
            # and previously uncounted, so even the log undercounted the failure rate.
            _JUDGE_FAILURES[0] += 1
            fails[0] += 1
        return (it["points"] if met else 0.0), met

    # Majority-vote grading (D1 noise study, 2026-08-03): a single grading call
    # flips on 2.65% of criteria; because rubrics average only ~2.5 items, one
    # flip moves the reward by ~0.6 (~2 group SDs), so ~40% of GRPO groups carry
    # a rollout with a wrong-signed advantage. 3-vote majority cuts residual
    # criterion error 0.89% -> 0.024% (noise SD 0.111 -> ~0.018). Raising n per
    # group does NOT help — each rollout is still graded once.
    # HB_JUDGE_VOTES=1 (default) keeps the old single-call behavior.
    # HB_JUDGE_VOTES_ADAPTIVE=1 spends the extra votes ONLY on the criterion
    # profile the study found noisy (>110 chars or multi-clause disjunctions:
    # ~35% of items, ~65% of the flips) — ~1.4x cost instead of 3x.
    _votes = max(1, int(os.environ.get("HB_JUDGE_VOTES", "1")))
    _adaptive = os.environ.get("HB_JUDGE_VOTES_ADAPTIVE", "1") == "1"

    def _is_noisy_item(it) -> bool:
        # `_rubric_items` normalizes the row to {"criterion", "points"}, so reading
        # only "criterion_text" here made t always "" -> never noisy -> with the
        # default HB_JUDGE_VOTES_ADAPTIVE=1 every item collapsed to n=1 and
        # HB_JUDGE_VOTES was silently a no-op. Accept both spellings.
        t = str(it.get("criterion") or it.get("criterion_text") or "")
        return len(t) > 110 or "at least one of" in t.lower()

    async def grade_voted(it):
        n = _votes
        if _votes > 1 and _adaptive and not _is_noisy_item(it):
            n = 1
        if n == 1:
            return await grade(it)
        results = await asyncio.gather(*[grade(it) for _ in range(n)])
        mets = [m for _pts, m in results if m is not None]
        if not mets:
            return results[0]
        met = sum(mets) > len(mets) / 2.0
        return (it["points"] if met else 0.0), met

    graded = await asyncio.gather(*[grade_voted(it) for it in items])
    achieved = sum(pts for pts, _met in graded)
    raw = achieved / total_pos  # may be negative if negative criteria fire
    # Per-criterion verdicts, forwarded (as JSON) to the gen server's /evolve loop
    # so the meta-optimizer sees WHICH criteria failed, not just the total score.
    rubric_met = [
        {"criterion": it["criterion"], "points": it["points"], "met": met}
        for (_pts, met), it in zip(graded, items)
    ]

    # HealthBench-Pro length adjustment on the ANSWER length (thinking stripped).
    # Applied on VALIDATION always (it IS the benchmark metric). On TRAINING it
    # is now opt-in (HB_TRAIN_LENGTH_ADJ=1): the unclamped linear term is the
    # only deterministic part of the training reward, so under judge-noise GRPO
    # the policy learns "shorter" first — v15's over-compression fingerprint —
    # and below the 2000-char center the "penalty" is a length BONUS.
    #
    # ON BY DEFAULT FOR TRAINING TOO (dvd, 2026-08-06): the training reward should be
    # the metric we are scored on. Length adjustment is part of HealthBench-Pro, so
    # optimizing raw rubric fraction while measuring length-adjusted score trains
    # against a different objective than the one reported — the generation runs did
    # exactly that (train rows are `healthbench_self`, so `is_val` was False and the
    # term was absent from training) and their val gain still turned out to be ~45%
    # length, earned incidentally rather than optimized.
    #
    # The known risk, kept here because it is real: this linear term is the only
    # DETERMINISTIC part of a judge-noisy reward, so the policy learns "shorter"
    # first. That is fine when a model is verbose (the 9B base wrote 10.6k chars,
    # penalty 0.25) and useless-to-harmful when it is not (the 27B SFT writes 4.4k,
    # penalty 0.07, and its acc_raw fell 0.492 -> 0.451). Watch acc_raw, not just
    # acc: if acc rises while acc_raw falls, the policy is buying score with brevity.
    # Set HB_TRAIN_LENGTH_ADJ=0 to train on raw rubric content instead.
    chars = len(answer_text)
    if not is_train:
        # VALIDATION: the official HealthBench-Professional term, two-sided and
        # unchanged. Held-out numbers must stay comparable with published results, so
        # nothing below touches this branch.
        length_adjusted = raw - LENGTH_ADJ_PENALTY_PER_500 * ((chars - LENGTH_ADJ_CENTER) / 500.0)
    elif os.environ.get("HB_TRAIN_LENGTH_ADJ", "1") == "1":
        # TRAINING: penalty ONLY above the centre. The two-sided form paid for brevity:
        # below 2000 chars (chars - CENTER) is negative, so the term ADDS reward, up to
        # +0.057 for a 63-char stub, with no reference to content at all.
        #
        # That is not theoretical. Measured on this run at step 165: 35 val answers are
        # under 800 chars, 84% of their positive criteria unmet, 31 of the 35 red_teaming
        # -- and their think length is ABOVE average, so the model reasoned and then
        # emitted a stub. Over the run, answers shrank 4159 -> 2554 chars and that
        # accounted for 27% of the entire val gain while positive-criteria credit moved
        # 2.4pp. On a wrapper task where every rollout scores ~0 on content, brevity is
        # the only term with variance, so GRPO has nothing else to climb.
        #
        # The comment above already warned "if acc rises while acc_raw falls, the policy
        # is buying score with brevity". It did exactly that; a one-sided term removes
        # the purchase while keeping the anti-verbosity pressure the term exists for.
        length_adjusted = raw - LENGTH_ADJ_PENALTY_PER_500 * (
            max(0.0, chars - LENGTH_ADJ_CENTER) / 500.0)
    else:
        length_adjusted = raw

    # Training-only thinking-budget penalty: reasoning beyond the free budget pays
    # a small linear cost (capped) so bounded thinking is preferred to drift.
    if not is_val and think_chars > HB_THINK_FREE_CHARS:
        length_adjusted -= min(
            HB_THINK_PENALTY_MAX,
            HB_THINK_PENALTY_PER_1K * (think_chars - HB_THINK_FREE_CHARS) / 1000.0,
        )

    # Training-only repetition penalty (v16). v15's dominant val failure was
    # n-gram repetition loops (answers repeating one sentence to the token cap:
    # 62/525 val answers, mean score 0.118) that greedy decoding amplifies but
    # sampled training rollouts only hint at — so the reward must catch the
    # attractor early. Penalize when the answer's or thinking's duplicated-8gram
    # fraction exceeds a free threshold; a hard floor for outright loops.
    if not is_val and HB_REP_PENALTY_MAX > 0.0:
        rep = max(_dup_ngram_frac(answer_text), _dup_ngram_frac(_strip_tool_spans(response_text)))
        if rep > HB_REP_FREE_FRAC:
            length_adjusted -= min(
                HB_REP_PENALTY_MAX,
                HB_REP_PENALTY_SLOPE * (rep - HB_REP_FREE_FRAC),
            )
        if rep >= HB_REP_LOOP_FRAC:  # unambiguous loop: floor the reward
            length_adjusted = min(length_adjusted, HB_SCORE_MIN)

    # ---- second graded component: retrieval quality ----------------------------
    # Points-weighted fraction of THIS task's rubric that the retrieved evidence
    # actually supplies, graded independently of whether the answer used it. Gated
    # on `is_train` (NOT `is_val`): on a train-on-val run every row's data_source is
    # `healthbench_professional/*`, so `is_val` is True for training rollouts too and
    # gating on it would silently disable the whole component.
    #
    # The bonus itself is applied group-relative in the trainer, where the GRPO group
    # (uid) exists — the reward manager is strictly per-sample. With
    # HB_RETRIEVAL_GROUP_BASELINE=0 we fold it here instead, against a fixed
    # counterfactual, for paths that have no groups (e.g. the SFT-loss trainer).
    retrieval_coverage, retrieval_judged, retrieval_bonus = 0.0, 0.0, 0.0
    if is_train and HB_RETRIEVAL_WEIGHT > 0.0 and str(retrieval_context).strip():
        from verl.utils.reward_score.retrieval_coverage import score_coverage
        cov, judged = await score_coverage(
            items,
            _task_text(extra_info),
            str(retrieval_context),
            cov_api_base or eff_base,
            cov_api_key if cov_api_base else eff_key,
            cov_model_name or eff_model,
            cov_provider if cov_api_base else eff_provider,
        )
        retrieval_coverage, retrieval_judged = cov, (1.0 if judged else 0.0)
        # A judge failure must degrade to "retrieval reward temporarily off", never
        # to "this rollout retrieved nothing useful".
        if judged and not HB_RETRIEVAL_GROUP_BASELINE:
            retrieval_bonus = HB_RETRIEVAL_WEIGHT * (cov - HB_RETRIEVAL_NOSEARCH_COVERAGE)

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
                   rubric_met=rubric_met, is_val=is_val,
                   judge_fail=fails[0] / max(1, len(items)),
                   retrieval_coverage=retrieval_coverage,
                   retrieval_judged=retrieval_judged,
                   retrieval_bonus=retrieval_bonus, **rtel)


def _dup_ngram_frac(text: str, n: int = 8) -> float:
    """Fraction of word n-gram positions that duplicate an earlier n-gram.
    ~0 for healthy prose; approaches 1 for sentence-repetition loops."""
    words = (text or "").split()
    if len(words) < 2 * n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _as_float(x, default: float = 0.0) -> float:
    """Best-effort numeric coercion for passthrough telemetry (None / str / bool)."""
    try:
        if x is None or isinstance(x, str) and not x.strip():
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _result(raw: float, length_adjusted: float, response_text: str,
            think_closed: bool = True, think_chars: int = 0,
            rubric_met: list | None = None, is_val: bool = False,
            judge_fail: float = 0.0,
            retrieval_bonus: float = 0.0,
            retrieval_coverage: float = 0.0, retrieval_judged: float = 0.0,
            retrieval_used: float = 0.0, retrieval_ctx_chars: float = 0.0,
            n_search: float = 0.0, n_queries: float = 0.0,
            retrieval_hits: float = 0.0, retrieval_error: float = 0.0,
            retrieval_truncated: float = 0.0, answer_rescued: float = 0.0,
            budget_exhausted: float = 0.0) -> dict:
    """Map the rubric signals onto the canonical key set (homogeneous DataProto
    batches). EVERY return path in this module goes through here — see the module
    docstring on why that is a hard rule and not a style preference.

    `retrieval_bonus` is the ONLY thing that may move the training scalar; it is
    added to `score` and never to `length_adjusted`, so every `acc*` metric — and
    therefore the whole validation curve — keeps its exact prior definition.
    """
    # Training reward may go below 0 (HB_SCORE_MIN) so bad rollouts stay ordered;
    # validation keeps the benchmark's [0, 1] clip.
    lo = 0.0 if is_val else min(0.0, HB_SCORE_MIN)
    # The [lo, 1] range bounds the ANSWER score. The retrieval bonus is a separate
    # additive term, so the clip is widened by its weight — otherwise an answer
    # already near an edge silently truncates the bonus, and asymmetrically: with
    # training scores at ~0.92 only the POSITIVE bonuses were being cut, rewarding
    # bad retrieval relative to good. (Same fix as the trainer-side group fold.)
    _w = HB_RETRIEVAL_WEIGHT if retrieval_bonus else 0.0
    score = max(lo - _w, min(1.0 + _w, float(length_adjusted) + float(retrieval_bonus)))
    raw01 = _clip01(raw)
    fmt_ok = 1.0 if (response_text or "").strip() else 0.0
    # HEADLINE val metric (`val-core/acc/mean`): the OFFICIAL HealthBench-Professional
    # score — length-adjusted, negatives subtracted, UNCLIPPED (openai/simple-evals does
    # not clip per example; the plain mean of these is the paper number). Previously this
    # reported clipped-raw (no length penalty, negatives floored at 0), which ran ~0.18 too
    # high. TRAINING keeps clipped-raw for the gen-server difficulty feedback (unchanged).
    acc_headline = float(length_adjusted) if is_val else float(raw01)
    # Splicing the shared floor FIRST is what makes homogeneity structural: this dict
    # is a superset of HOMOGENEOUS_DEFAULTS by construction, so adding a key there can
    # never leave this branch short of it.
    return {
        **HOMOGENEOUS_DEFAULTS,
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
        "acc": acc_headline,                  # val-core/acc: official len-adj signed (val) / raw (train)
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
        # Fraction of this sample's criteria the judge could not grade at all. A
        # nonzero mean here means the reward is measuring the judge, not the policy.
        "judge_fail": float(judge_fail),
        # ---- retrieval: telemetry + the separately-graded second component ----
        # `retrieval_coverage` is the points-weighted fraction of this task's rubric
        # criteria that the RETRIEVED CONTEXT factually supplies — graded independently
        # of whether the answer then used them, so query quality has its own signal.
        "retrieval_used": float(retrieval_used),
        "retrieval_coverage": float(retrieval_coverage),
        "retrieval_judged": float(retrieval_judged),   # 0 => coverage judge gave no verdict
        "retrieval_bonus": float(retrieval_bonus),
        "retrieval_ctx_chars": float(retrieval_ctx_chars),
        "n_search": float(n_search),
        "n_queries": float(n_queries),
        "retrieval_hits": float(retrieval_hits),
        "retrieval_error": float(retrieval_error),
        "retrieval_truncated": float(retrieval_truncated),
        "answer_rescued": float(answer_rescued),
        "budget_exhausted": float(budget_exhausted),
    }
