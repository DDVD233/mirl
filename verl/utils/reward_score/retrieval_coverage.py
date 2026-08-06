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

# ---------------------------------------------------------------------------
# EVOLVABLE COVERAGE PROMPT
#
# The coverage judge defines what "good retrieval" means, so it is the natural
# thing to evolve when the goal is to teach the policy to retrieve better. The
# gen server's /evolve_retrieval rewrites this file; the reward re-reads it on
# mtime change, exactly like the generation prompts.
#
# EVERY evolved prompt is validated on BOTH sides — the server refuses to commit
# one that fails, and this module refuses to load one that fails. That is not
# redundant: they guard different events (a bad rewrite vs. a hand-edited or
# truncated file), and the cost of getting it wrong is a reward that silently
# returns 0 coverage for every rollout, which is indistinguishable from a policy
# that cannot retrieve.
# ---------------------------------------------------------------------------
COVERAGE_PROMPT_FILE = os.environ.get("HB_COVERAGE_PROMPT_FILE", "")

# The template is str.format-ed with exactly these fields, and the parser reads
# results[].idx/.supplied. An evolved prompt that drops any of them is not a
# worse prompt, it is a broken one.
REQUIRED_TEMPLATE_FIELDS = ("{task}", "{passages}", "{criteria}")
REQUIRED_CONTRACT_TOKENS = ("results", "idx", "supplied")


def validate_coverage_prompt(system: str, template: str) -> str:
    """Return "" if the prompt is usable, else a human-readable reason."""
    if not (system or "").strip():
        return "empty system prompt"
    if not (template or "").strip():
        return "empty template"
    missing = [f for f in REQUIRED_TEMPLATE_FIELDS if f not in template]
    if missing:
        return f"template is missing required field(s): {', '.join(missing)}"
    absent = [t for t in REQUIRED_CONTRACT_TOKENS if t not in template]
    if absent:
        return (f"template no longer states the JSON output contract "
                f"(missing {', '.join(absent)}) — the parser would reject every verdict")
    try:
        # Catches stray single braces, which raise at format() time and would
        # otherwise take down the reward for the whole run.
        template.format(task="x", passages="y", criteria="z")
    except Exception as e:  # noqa: BLE001
        return f"template does not format cleanly: {type(e).__name__}: {e}"
    return ""


_PROMPT_CACHE: dict = {"raw": None, "system": COVERAGE_SYSTEM,
                       "template": COVERAGE_TEMPLATE, "version": 0}
_PROMPT_WARNED: set = set()


def current_coverage_prompt() -> tuple[str, str, int]:
    """(system, template, version). Falls back to the built-in default whenever
    the file is missing, unreadable, or invalid.

    Change detection compares the file's CONTENT, not its mtime. mtime looked
    like the obvious cache key and is what the generation prompts use, but it is
    only reliable when writes are far apart: filesystems with coarse timestamp
    granularity report the same mtime for two writes in the same second, and a
    freshly-evolved prompt would then be ignored for the rest of the run with
    nothing in the logs to say so. The file is a few KB and every read here
    precedes an LLM call, so re-reading it costs nothing worth measuring.
    """
    path = COVERAGE_PROMPT_FILE
    if not path:
        return _PROMPT_CACHE["system"], _PROMPT_CACHE["template"], 0
    try:
        with open(path) as f:
            raw = f.read()
    except OSError:
        return COVERAGE_SYSTEM, COVERAGE_TEMPLATE, 0
    if raw == _PROMPT_CACHE["raw"]:
        return _PROMPT_CACHE["system"], _PROMPT_CACHE["template"], _PROMPT_CACHE["version"]
    try:
        obj = json.loads(raw)
        system = str(obj.get("system") or "")
        template = str(obj.get("template") or "")
        version = int(obj.get("version") or 0)
    except Exception as e:  # noqa: BLE001
        if path not in _PROMPT_WARNED:
            _PROMPT_WARNED.add(path)
            logger.error("coverage prompt %s unreadable, keeping default: %s: %s",
                         path, type(e).__name__, e)
        _PROMPT_CACHE["raw"] = raw          # don't re-parse the same bad file every call
        return _PROMPT_CACHE["system"], _PROMPT_CACHE["template"], _PROMPT_CACHE["version"]
    reason = validate_coverage_prompt(system, template)
    if reason:
        # Loud: a silently-rejected evolved prompt means the reward-evolution
        # experiment is not running even though every log says it is.
        logger.error("coverage prompt v%d REJECTED (%s); keeping previous", version, reason)
        _PROMPT_CACHE["raw"] = raw
        return _PROMPT_CACHE["system"], _PROMPT_CACHE["template"], _PROMPT_CACHE["version"]
    _PROMPT_CACHE.update({"raw": raw, "system": system,
                          "template": template, "version": version})
    logger.warning("coverage prompt reloaded: v%d from %s", version, path)
    return system, template, version


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
    system, template, _version = current_coverage_prompt()
    try:
        prompt = template.format(
            task=(task_text or "")[:4000],
            passages=(context or "")[:CTX_CHARS],
            criteria=criteria,
        )
    except Exception as e:  # noqa: BLE001 — validation should make this unreachable
        logger.error("coverage template failed to format, using built-in: %s: %s",
                     type(e).__name__, e)
        system, template = COVERAGE_SYSTEM, COVERAGE_TEMPLATE
        prompt = template.format(
            task=(task_text or "")[:4000],
            passages=(context or "")[:CTX_CHARS],
            criteria=criteria,
        )
    try:
        raw = await _call_api(
            api_base, api_key, model_name, system, prompt,
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
