"""Retrieval-quality reward: does the retrieved context SUPPLY what the rubric grades?

This is the second, separately-graded component of the retrieval RL run. The answer
grader asks "did the response satisfy the criterion"; this asks "was the fact the
criterion rewards present in the evidence the model retrieved" — independently of
whether the model then used it. Grading them separately is what gives the QUERY
tokens their own learning signal: a rollout can retrieve well and answer badly, and
the two show up as different numbers.

The prompt is lifted verbatim from ``scripts/self_evolving/kb/multiquery_probe.py``,
which is the protocol the supply measurements were made with (single query supplies
the graded fact 10.8% of the time, a 2-query plan 35.9%). Keeping one definition
matters: if the offline probe and the online reward drifted, the reward would be
optimizing a target the measurements never validated.

Scoring: points-weighted over ``abs(points)`` so negative criteria (things the answer
must NOT do, which retrieval can also help avoid) count as the grading targets they
are. Behaviour-only criteria stay in the denominator, so coverage is structurally low
on non-factual tasks — under the group-relative baseline (see the trainer fold) that
offset cancels within the GRPO group.

FAILURE POLICY: a judge failure returns ``judged=False`` and the caller applies NO
bonus, leaving the answer reward fully intact. It must never look like "coverage 0".
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

COVERAGE_SYSTEM = """\
You are auditing whether a set of retrieved passages supplies what a grading rubric \
criterion rewards. You are precise and sceptical: a passage that is merely on-topic does \
NOT count; it must actually STATE the fact the criterion rewards (or enough of it that a \
competent physician reading only these passages would satisfy the criterion).

Return ONLY valid JSON, no prose, no markdown fence."""

COVERAGE_TEMPLATE = """\
# Clinician request
{task}

# Retrieved passages (all the model would see)
{passages}

# Criteria
{criteria}

# Task
For EACH criterion output an object:
{{"idx": <int>, "supplied": true|false, "passage": "<the passage number that states it, or null>", \
"why": "<12 words max>"}}

"supplied" is true ONLY if some passage above STATES the rewarded fact. If the criterion \
rewards a pure behaviour (asking a follow-up, tone, formatting) with no factual branch, \
set "supplied": false and why "behaviour".

Return {{"results": [...]}}"""

# Context handed to the coverage judge. The brief is ~2-3k chars; the cap only
# guards against an unsummarized fallback payload.
CTX_CHARS = int(os.environ.get("HB_COVERAGE_CTX_CHARS", "12000"))
MAX_TOKENS = int(os.environ.get("HB_COVERAGE_MAX_TOKENS", "900"))


def _parse(raw: str) -> dict[int, bool] | None:
    t = re.sub(r"^```json\s*|\s*```$", "", (raw or "").strip())
    try:
        d = json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j <= i:
            return None
        try:
            d = json.loads(t[i:j + 1])
        except Exception:
            return None
    rows = d.get("results") if isinstance(d, dict) else None
    if not isinstance(rows, list):
        return None
    out: dict[int, bool] = {}
    for r in rows:
        if isinstance(r, dict) and "idx" in r:
            try:
                out[int(r["idx"])] = bool(r.get("supplied"))
            except (TypeError, ValueError):
                continue
    return out or None


async def score_coverage(
    items: list[dict],
    task_text: str,
    context: str,
    api_base: str,
    api_key: str,
    model_name: str,
    provider: str = "",
) -> tuple[float, bool]:
    """Return (coverage in [0,1], judged).

    `items` are the normalized rubric rows ({"criterion", "points"}). `context` is
    exactly the evidence text the model saw. `judged` is False whenever the judge
    gave no usable verdict — the caller must then apply no bonus at all.
    """
    if not items or not (context or "").strip() or not api_base:
        return 0.0, False

    # Lazy import: self_evolving owns the shared judge semaphore + provider payload
    # shapes, and importing it at module scope would close an import cycle.
    from verl.utils.reward_score.self_evolving import _call_api

    criteria = "\n".join(
        f"{i}. [{it['points']:+g} pts] {it['criterion']}" for i, it in enumerate(items)
    )
    prompt = COVERAGE_TEMPLATE.format(
        task=(task_text or "")[:4000],
        passages=(context or "")[:CTX_CHARS],
        criteria=criteria,
    )
    try:
        raw = await _call_api(
            api_base, api_key, model_name, COVERAGE_SYSTEM, prompt,
            max_tokens=MAX_TOKENS, provider=provider,
        )
    except Exception as e:
        logger.warning("coverage judge FAILED (%s@%s): %s: %s",
                       model_name, api_base, type(e).__name__, e)
        return 0.0, False

    from verl.utils.reward_score.healthbench_pro import _strip_thinking

    verdicts = _parse(_strip_thinking(raw))
    if not verdicts:
        logger.warning("coverage judge returned unparseable JSON: %r", (raw or "")[:200])
        return 0.0, False

    denom = sum(abs(it["points"]) for it in items)
    if denom <= 0:
        return 0.0, False
    num = sum(abs(it["points"]) for i, it in enumerate(items) if verdicts.get(i))
    return max(0.0, min(1.0, num / denom)), True
