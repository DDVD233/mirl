"""Self-evolving reward: judge-prompt evolution + executable function reward.

Optional, default-OFF feature layered on top of the existing self-evolving medical
reward (see ``self_evolving.compute_score``). When enabled, at the end of each
training step the judge model:

  1. rewrites the JUDGING PROMPT used to score student responses, and
  2. writes / improves a Python ``function_reward(...)`` that programmatically
     inspects the response text.

Both artifacts are versioned per-step under ``{output_dir}/reward_evolution/`` and
the reward function reads the latest *valid* pair from ``current/``. When the
switch is OFF, none of this runs and the reward is byte-identical to before.

Consumers:
  - ``self_evolving.compute_score`` (evolve_enable kwarg) — applies the evolved
    judge prompt + function as the training reward.
  - ``ray_trainer._maybe_evolve_reward`` — calls ``evolve_once`` at end of step.
  - ``scripts/self_evolving/eval/test_reward_evolution.py`` — offline harness used
    to design / validate the meta-prompts on real dumps before production.

This module imports a few helpers from ``self_evolving`` (``_call_api``,
``extract_boxed_answer``, ``check_accuracy``). To avoid an import cycle,
``self_evolving`` imports *this* module lazily (inside ``compute_score``), never at
module top level.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile

import aiohttp

from verl.utils.reward_score.self_evolving import (  # noqa: E402
    _call_api,
    check_accuracy,
    extract_boxed_answer,
    extract_final_answer,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Steering + starter artifacts
# ---------------------------------------------------------------------------

# One-sentence steering injected into both evolution meta-prompts. The training
# data is mimic-rare (diagnose the primary condition from an admission); the
# downstream heldout eval is HealthBench-Professional (patient/doctor chat
# quality), so we steer toward generic diagnostic correctness + usefulness, never
# toward the heldout set itself.
STEER_SENTENCE = (
    "Optimize for medical answers that are diagnostically correct AND genuinely useful as a "
    "doctor's reply — complete, clearly communicated, appropriately calibrated, and safe — so "
    "the signal transfers from rare-disease diagnosis (MIMIC) to open-ended patient–doctor "
    "consultations (HealthBench Professional)."
)

# Shared context injected into BOTH meta-prompts. The reward is trained on MIMIC (single
# ground-truth diagnosis per item) but the artifact is also evaluated, unseen, on HealthBench
# Professional (open-ended consultations graded by physician rubrics). Left unchecked, the
# evolved judge/function drift toward "exact boxed diagnosis = 10, penalize everything else,
# prioritize terseness" — which games MIMIC accuracy but tanks HealthBench (which rewards
# thorough, well-communicated, safe consultations). This block counters that drift.
DUAL_OBJECTIVE = """\
DUAL OBJECTIVE — read carefully. The student model is TRAINED on MIMIC rare-disease cases \
(each has ONE ground-truth diagnosis) but will ALSO be evaluated, unseen, on open-ended \
patient–doctor consultations graded by physician-written rubrics (HealthBench Professional), \
which score: diagnostic/factual ACCURACY, COMPLETENESS (covers the key issues, work-up, and \
next steps), COMMUNICATION quality (clear, well-organized, patient-appropriate), CALIBRATION \
(commits when warranted, states uncertainty when warranted), and SAFETY/ESCALATION (flags red \
flags, no dangerous advice). Therefore reward GENERALIZABLE clinical-answer quality, NOT just \
exact diagnosis-string / ICD matching. Concretely:
- A correct diagnosis delivered as a clear, complete, safe, patient-appropriate explanation \
must OUTSCORE a bare correct \\boxed{code} with no reasoning.
- Do NOT reward terse, code-only outputs and do NOT penalize appropriate, well-organized \
clinical detail — that bias games MIMIC but fails HealthBench.
- Keep diagnostic accuracy a necessary, heavily-weighted axis, but make reasoning, \
completeness, communication, calibration, and safety co-count so the reward transfers.
- Reward a CLEAR, COMMITTED final answer in the format the task asks for — the MIMIC task \
requires the final diagnosis in \\boxed{{...}}, so reward responses that commit one and \
penalize wishy-washy answers that never state a single final diagnosis. (Do not zero a \
clearly-stated conclusion merely for a missing literal box, but a present, committed \\boxed{{}} \
answer is the desired behavior and should score higher than an uncommitted one.)"""

# The judge-scoring contract is fixed: the grader receives these exact bracket
# labels in the user message and must emit one integer 0-10 in \boxed{...}. The
# evolved prompt becomes the grader's *system* prompt and must honor this.
STARTER_JUDGE_PROMPT = """\
You are a senior attending physician grading another model's answer to a clinical \
case. Your score is used as a reinforcement-learning reward, so it must be \
faithful and discriminating.

The user message gives you:
- [QUESTION]: the clinical case / question posed to the model.
- [MODEL RESPONSE]: the model's full answer, including its reasoning and final diagnosis.
- [CORRECT ANSWER]: the reference correct diagnosis / answer.

Rate the MODEL RESPONSE from 0 to 10, judging holistically:
- Diagnostic accuracy: does the final diagnosis match [CORRECT ANSWER]? Exact match \
earns full credit; a closely related diagnosis (same disease family, broader/narrower \
form, same ICD-10 category) earns partial credit; an unrelated or wrong diagnosis earns \
little or none.
- Clinical reasoning: is the reasoning sound, evidence-based, and free of dangerous errors?
- Usefulness as a doctor's reply: is the answer clear, specific, well-organized, and \
genuinely helpful and safe for a patient–doctor conversation?

Guidance: 0 = unrelated, wrong, or unsafe. 5 = correct diagnosis but mediocre reasoning \
or an unclear reply, OR a strong, useful reply that only partially matches the correct \
diagnosis. 10 = correct diagnosis with excellent, safe, useful clinical reasoning.

Output ONLY your integer score inside \\boxed{...} and nothing else. Example: \\boxed{7}"""

# Exact starter function the user specified. It returns a constant 0.0, which under
# GRPO advantage normalization cancels within each group -> zero gradient effect
# until the judge evolves it into something that varies across samples.
STARTER_FUNCTION_SRC = '''\
def function_reward(input_question: str, output_answer: str, ground_truth: str) -> float:
    return 0.0
'''

FUNCTION_SIGNATURE = (
    "def function_reward(input_question: str, output_answer: str, ground_truth: str) -> float"
)

# Per-sample runtime budget (seconds) the evolved function may spend. Reward scoring runs
# samples concurrently (each function call is dispatched to a thread, see
# self_evolving._evolve_addon), so a generous per-call timeout does not serialize the step.
FUNCTION_TIME_BUDGET_S = int(os.environ.get("REWARD_FN_TIMEOUT", "25"))

# Runtime tooling contract handed to the function-evolution judge. The evolved
# function_reward runs inside the training Docker on a GPU node WITH network + a writable
# cache, so it may call the grading judge, query our medical vector DB, and download
# models/tools — but it must NEVER touch the open web (held-out eval leak surface). The
# concrete endpoints arrive as environment variables (exported by the run script; see
# run_qwen36_27b_evolve_reward.sh) so the fixed 3-arg signature stays unchanged.
FUNCTION_TOOLING_GUIDE = f"""\
RUNTIME ENVIRONMENT & TOOLS. The function runs inside the training Docker on a GPU node with \
network access and a writable cache. You MAY import any library, download models/tools, call our \
grading judge LLM, and query our medical vector database. You have a generous per-call budget \
(~{FUNCTION_TIME_BUDGET_S}s; calls are dispatched concurrently across samples, so this does not \
serialize the step) — but you MUST still wrap every external call in try/except with a fast \
deterministic fallback and pass an explicit network timeout so one hung call can't stall.

HARD RULE — NO OPEN WEB: never do web search / scraping / fetch arbitrary external URLs. That can \
leak the held-out evaluation. ONLY these are allowed: our own judge endpoint, our own vector DB, \
and model/tool downloads from trusted hubs (e.g. HuggingFace). Nothing else on the network.

Credentials/endpoints are provided as environment variables:
  REWARD_JUDGE_API_BASE, REWARD_JUDGE_API_KEY, REWARD_JUDGE_MODEL   # grading LLM (OpenAI-compatible)
  EMBED_API_BASE, EMBED_API_KEY, EMBED_MODEL                        # embeddings for vector search
  MILVUS_URI, MILVUS_TOKEN, MILVUS_COLLECTION                       # our medical knowledge vector DB
  REWARD_FN_TOOL_CACHE                                              # writable dir for model/tool downloads

Example 1 — ask the judge LLM a targeted question about the answer:
```python
import os, requests
def _ask_judge(system, user, max_tokens=64, timeout={FUNCTION_TIME_BUDGET_S}):
    base = os.environ.get("REWARD_JUDGE_API_BASE")
    if not base:
        return None
    r = requests.post(base.rstrip("/") + "/chat/completions",
        headers={{"Authorization": f"Bearer {{os.environ.get('REWARD_JUDGE_API_KEY','EMPTY')}}"}},
        json={{"model": os.environ.get("REWARD_JUDGE_MODEL", ""), "max_tokens": max_tokens,
              "temperature": 0.0, "chat_template_kwargs": {{"enable_thinking": False}},
              "messages": [{{"role": "system", "content": system}},
                           {{"role": "user", "content": user}}]}},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content", "")
```

Example 2 — search our medical vector DB (embed the query, then ANN-search Milvus):
```python
import os, requests
from pymilvus import MilvusClient
def _kb_search(query, top_k=5, timeout={FUNCTION_TIME_BUDGET_S}):
    eb = os.environ.get("EMBED_API_BASE")
    e = requests.post(eb.rstrip("/") + "/embeddings",
        headers={{"Authorization": f"Bearer {{os.environ.get('EMBED_API_KEY','EMPTY')}}"}},
        json={{"model": os.environ.get("EMBED_MODEL"), "input": [query[:2000]]}}, timeout=timeout)
    vec = e.json()["data"][0]["embedding"]
    cli = MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ.get("MILVUS_TOKEN", "root:Milvus"))
    hits = cli.search(collection_name=os.environ.get("MILVUS_COLLECTION", "medical_knowledge_v2"),
        data=[vec], limit=top_k, output_fields=["text_content", "question", "answer"])
    return hits[0] if hits else []
```

Example 3 — download & run any tool/model (download ONCE at module import; it is cached after):
```python
import os
os.environ.setdefault("HF_HOME", os.environ.get("REWARD_FN_TOOL_CACHE", "/tmp/reward_fn_cache"))
os.environ["HF_HUB_OFFLINE"] = "0"   # allow the first download; cached calls need no network
_TOOL = None
def _get_tool():
    global _TOOL
    if _TOOL is None:
        from transformers import pipeline      # or a segmentation / vision model, nnUNet, etc.
        _TOOL = pipeline("text-classification", model="some/medical-model")
    return _TOOL
```

Cache results in a module-level dict keyed on (output_answer, ground_truth) so repeated identical \
samples don't re-hit the network. Keep all of this OPTIONAL: if a tool/import/endpoint is \
unavailable, fall straight back to your deterministic scoring — never raise."""


# ---------------------------------------------------------------------------
# Error-summarization meta-prompts (per-case + aggregate)
# ---------------------------------------------------------------------------

PER_CASE_SUMMARY_SYSTEM = f"""\
You are diagnosing failures in a medical reinforcement-learning reward loop. You are given ONE \
case: the clinical QUESTION, the student model's full RESPONSE, the CORRECT answer, and the \
reward signals the CURRENT reward setting produced for it — namely the CURRENT judge prompt and \
the CURRENT Python function_reward (shown to you), plus the sub-reward scores they assigned.

{STEER_SENTENCE}

In about 500 tokens, do TWO things for THIS case:
1. STUDENT ERROR — precisely what did the student get wrong (or right)? Name the specific \
knowledge or capability gap, e.g.: wrong ICD subtype / right disease family but wrong code; \
confused two related-but-distinct entities; never committed a final \\boxed diagnosis or ran out \
of tokens mid-reasoning; hedged with a differential instead of committing; over-reached for a \
rare diagnosis when the true answer was common; couldn't map raw EHR procedure/med codes to \
meaning.
2. REWARD CRITIQUE — for THIS case, what did the current judge prompt and the current \
function_reward MISS or mis-score? Did they over-credit or under-credit it relative to how a \
careful attending would grade it? What concrete signal (an ICD-subtype check, a commitment/box \
check, a knowledge-base lookup, a confusion-pair penalty, …) would have graded this case \
correctly?

Be concrete and specific to this case; do not give generic advice. Output prose."""


AGGREGATE_SUMMARY_SYSTEM = """\
You are given a set of per-case failure analyses from a medical reinforcement-learning reward \
loop. Each describes what a student model got wrong on one case AND what the current reward \
setting (the LLM-judge prompt + the Python function_reward) missed or mis-scored on that case. \
Base your synthesis only on the cases actually provided below, however many there are.

Synthesize them into about 1000 tokens with three clearly labeled sections:
1. COMMON STUDENT ERROR PATTERNS — the recurring knowledge / capability gaps, ranked by how \
often and how badly they hurt (e.g. "never commits a boxed diagnosis / truncates", "right \
disease family but wrong ICD subtype", "rare-diagnosis over-reach on common cases", specific \
confusion pairs).
2. REWARD GAPS — where the current judge prompt and the current function_reward systematically \
fail to reward or penalize the right thing (over-crediting un-committed rambles, missing \
ICD-subtype partial credit, not using available tools / knowledge base, etc.).
3. IMPROVEMENT SUGGESTIONS — concrete, actionable changes, given SEPARATELY for (a) the JUDGE \
PROMPT and (b) the FUNCTION_REWARD, that would better separate strong from weak student answers \
and push the student to fix the common errors. Be specific and implementable."""


# ---------------------------------------------------------------------------
# Example formatting (shared by both meta-prompts and the offline harness)
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n - 20].rstrip() + f"\n... [truncated {len(s) - n + 20} chars]"


def format_examples(examples: list[dict], q_chars: int = 1600, r_chars: int = 1400) -> str:
    """Render worked examples (question, response, GT, sub-rewards) for a meta-prompt."""
    blocks = []
    for i, ex in enumerate(examples, 1):
        sub = ex.get("sub_rewards", {}) or {}
        sub_str = ", ".join(f"{k}={float(v):.3f}" for k, v in sub.items()) if sub else "(none)"
        blocks.append(
            f"### Example {i}\n"
            f"[QUESTION]\n{_truncate(ex.get('question', ''), q_chars)}\n\n"
            f"[MODEL RESPONSE]\n{_truncate(ex.get('response', ''), r_chars)}\n\n"
            f"[CORRECT ANSWER]\n{ex.get('ground_truth', '')}\n\n"
            f"[EXTRACTED ANSWER] {ex.get('extracted_answer', '')}\n"
            f"[COMBINED REWARD] {float(ex.get('combined_reward', 0.0)):.3f}\n"
            f"[SUB-REWARDS] {sub_str}"
        )
    return "\n\n".join(blocks)


def format_one_case(ex: dict, current_prompt: str, current_fn_src: str,
                    q_chars: int = 1600, r_chars: int = 1800, fn_chars: int = 6000) -> str:
    """Render a single case + the current reward setting, for per-case error summarization."""
    sub = ex.get("sub_rewards", {}) or {}
    sub_str = ", ".join(f"{k}={float(v):.3f}" for k, v in sub.items()) if sub else "(none)"
    return (
        "## THE CASE\n"
        f"[QUESTION]\n{_truncate(ex.get('question', ''), q_chars)}\n\n"
        f"[STUDENT RESPONSE]\n{_truncate(ex.get('response', ''), r_chars)}\n\n"
        f"[CORRECT ANSWER]\n{ex.get('ground_truth', '')}\n\n"
        f"[STUDENT EXTRACTED ANSWER] {ex.get('extracted_answer', '')}\n"
        f"[CURRENT REWARD ASSIGNED] combined={float(ex.get('combined_reward', 0.0)):.3f}; "
        f"sub-rewards: {sub_str}\n\n"
        "## CURRENT JUDGE PROMPT (being critiqued)\n"
        f"{_truncate(current_prompt, 4000)}\n\n"
        "## CURRENT function_reward (being critiqued)\n"
        "```python\n"
        f"{_truncate(current_fn_src, fn_chars)}\n"
        "```\n"
    )


# ---------------------------------------------------------------------------
# Meta-prompt builders
# ---------------------------------------------------------------------------

def build_prompt_evolution_messages(current_prompt: str, error_summary: str) -> tuple[str, str]:
    """Return (system, user) messages asking the judge to improve the judging prompt.

    ``error_summary`` is the aggregated error-analysis + improvement-suggestions text distilled
    from this step's per-case student-failure summaries (see ``aggregate_error_summary``). It
    replaces the raw worked-examples that earlier versions pasted in.
    """
    system = f"""\
You are improving the JUDGING PROMPT used to score a medical model's answers inside a \
reinforcement-learning loop. That judging prompt is handed to a grader model as its \
system prompt; the grader then reads a [QUESTION], [MODEL RESPONSE], and [CORRECT ANSWER] \
and must output a single integer score from 0 to 10 inside \\boxed{{...}}.

{STEER_SENTENCE}

{DUAL_OBJECTIVE}

You will be shown the CURRENT judging prompt and an ERROR ANALYSIS + IMPROVEMENT SUGGESTIONS \
distilled from the student model's recent failures (recurring knowledge/capability gaps and the \
specific places the current grading mis-scored them). Use that analysis to decide what to fix. \
Watch for: too lenient on wrong diagnoses, too harsh on correct-but-differently-worded answers, \
ignoring clarity / completeness / safety, missing ICD-subtype partial credit — and especially the \
common DRIFT toward "boxed answer absent => 0, deduct for length, prioritize terseness", which \
over-fits MIMIC and would hurt HealthBench. Fix what the analysis flags.

Then write an IMPROVED judging prompt: a clear, self-contained, MULTI-AXIS rubric covering \
diagnostic accuracy, clinical reasoning, completeness, communication clarity, calibration, \
and safety (per the dual objective above). Identify the model's committed final answer \
(often in \\boxed{{...}}) and weight diagnostic correctness heavily, but score the FULL reply \
across all axes — do NOT collapse the rubric to boxed-exact-match alone, and do NOT make a \
missing box an automatic zero if the conclusion is clearly stated.

HARD REQUIREMENTS for the prompt you write:
- It must tell the grader it will receive [QUESTION], [MODEL RESPONSE], and [CORRECT ANSWER] \
(these exact bracket labels are what the grader is given).
- It must require the grader to output exactly one integer from 0 to 10 inside \\boxed{{...}} \
and nothing else.
- Keep it self-contained and under ~400 words.

First think through the failure modes (reason freely). Then, OUTSIDE of and AFTER any \
reasoning, output the final improved judging prompt and NOTHING else between these tags:
<prompt>
...your improved judging prompt...
</prompt>"""

    user = (
        "## CURRENT JUDGING PROMPT\n"
        f"{current_prompt}\n\n"
        "## ERROR ANALYSIS & IMPROVEMENT SUGGESTIONS (from this step's student failures)\n"
        f"{error_summary or '(no analysis available this round)'}\n\n"
        "Now produce the improved judging prompt inside <prompt>...</prompt>."
    )
    return system, user


def build_function_evolution_messages(current_fn_src: str, error_summary: str) -> tuple[str, str]:
    """Return (system, user) messages asking the judge to improve function_reward.

    ``error_summary`` is the aggregated error-analysis + improvement-suggestions text (see
    ``aggregate_error_summary``), used in place of raw worked examples.
    """
    system = f"""\
You are writing an executable Python reward function that AUGMENTS an LLM judge in a \
medical reinforcement-learning loop. It deterministically inspects the model's answer \
text and returns a score that will be CLAMPED to [0, 1] and combined with the judge's score.

Use exactly this signature (do not change it):
{FUNCTION_SIGNATURE}

- input_question: the clinical case / question.
- output_answer: the model's full response text (reasoning + final diagnosis).
- ground_truth: the reference correct answer (often "ICD-10 CODE: Diagnosis name").
- return: a float; higher = better. It will be clamped to [0, 1].

{STEER_SENTENCE}

{DUAL_OBJECTIVE}

{FUNCTION_TOOLING_GUIDE}

Make it ROBUST:
- Never raise on weird input — wrap risky work in try/except and fall back to a sane default.
- Avoid unbounded loops; bound every external call with an explicit timeout.
- Do NOT read or write arbitrary files or call os.system on the host; model/tool downloads to \
the provided cache dir are fine.

You will be shown the CURRENT function and an ERROR ANALYSIS + IMPROVEMENT SUGGESTIONS distilled \
from the student model's recent failures (the recurring knowledge/capability gaps and where the \
current function mis-scored them). Use that analysis to decide what signal to add. Improve the \
function so its output is a USEFUL, WELL-SPREAD signal:
- a correct or clinically-equivalent final answer scores HIGH (≈0.8–1.0),
- a clearly wrong answer scores LOW (≈0.0–0.2),
- a near-miss (same ICD-10 3-character category, or a closely related diagnosis) lands in between.

Base the score on the model's COMMITTED final answer, not on incidental mentions. Prefer the \
\\boxed{{...}} content (the student is trained to end with e.g. \\boxed{{E70.0: Classical \
phenylketonuria}}), but FALL BACK GRACEFULLY when there is no clean box: use the last \
explicit "final / primary diagnosis:" line, else the most prominent diagnosis the response \
commits to. Avoid handing full credit to an ICD code or disease name that only appears in \
passing in the middle of the reasoning, but do not become so strict that you fail to credit \
a genuinely correct answer.

Useful signals: normalize and compare ICD-10 codes (exact = full credit, same 3-char \
category = partial credit) and weight the CODE match more heavily than the name match; match \
the diagnosis name via a real synonym or substring overlap, and AVOID loose character-\
similarity ratios (e.g. difflib on whole strings) that hand false credit to unrelated \
diseases which merely share letters (a wrong "melanoma" must not look like a correct \
"myeloma"). Lightly penalize empty, hedging, or unsafe replies. CRITICAL: do NOT collapse to \
a constant or near-zero output — on the examples shown, the correct answers must end up \
clearly higher than the wrong ones.

KEEP IT FOCUSED AND MAINTAINABLE (aim for under ~250 lines). The current function is given to \
you to IMPROVE, not just to extend — once it already separates correct from wrong reliably, \
PREFER refining, consolidating, or removing redundant/dead branches over piling on more \
heuristics. Do not let it grow unboundedly.

The LLM judge already scores overall correctness and quality, so make the function COMPLEMENTARY: \
add cheap, deterministic signal the judge is weak or inconsistent at, e.g. did the model commit \
to exactly ONE final answer (penalize multiple/contradictory final diagnoses), is the ICD-10 code \
well-formed and consistent with the stated diagnosis name, and does the reply actually read as a \
usable clinical answer. Do NOT give full credit to a bare \\boxed{{code}} with no diagnosis name or \
reasoning, and do NOT reward extreme terseness (a code-only answer would fail an open-ended \
HealthBench consultation) — but likewise do not just reward length. Start simple and correct; \
refine it over future rounds.

First reason about what signal to add (reason freely). Then, OUTSIDE of and AFTER any \
reasoning, output the COMPLETE function (with any imports and helpers it needs) as the \
final artifact in a single fenced block:
```python
# imports + def function_reward(...) -> float
```
Output only that one ```python code block as the final artifact."""

    user = (
        "## CURRENT FUNCTION\n"
        "```python\n"
        f"{current_fn_src}\n"
        "```\n\n"
        "## ERROR ANALYSIS & IMPROVEMENT SUGGESTIONS (from this step's student failures)\n"
        f"{error_summary or '(no analysis available this round)'}\n\n"
        "Now produce the improved function as a single ```python code block."
    )
    return system, user


# ---------------------------------------------------------------------------
# Error summarization: per-case + aggregate (judge calls)
# ---------------------------------------------------------------------------

async def summarize_case_error(
    api_base: str,
    api_key: str,
    model_name: str,
    current_prompt: str,
    current_fn_src: str,
    example: dict,
    max_tokens: int = 900,
) -> str:
    """Ask the judge to summarize, for ONE case, what the student got wrong AND what the current
    judge prompt + function_reward missed (~500 tokens of content). Returns "" on failure."""
    user = format_one_case(example, current_prompt, current_fn_src)
    try:
        return await _call_judge_design(
            api_base, api_key, model_name, PER_CASE_SUMMARY_SYSTEM, user, max_tokens=max_tokens
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"summarize_case_error failed: {type(e).__name__}: {e}")
        return ""


async def aggregate_error_summary(
    api_base: str,
    api_key: str,
    model_name: str,
    case_summaries: list[str],
    max_tokens: int = 1400,
) -> str:
    """Collapse the per-case summaries into one ~1000-token error summary + improvement
    suggestions (separately for judge prompt and function). Returns "" on failure."""
    if not case_summaries:
        return ""
    blocks = "\n\n".join(
        f"### Case {i} analysis\n{s}" for i, s in enumerate(case_summaries, 1) if s
    )
    user = (
        "Here are the per-case failure analyses. Synthesize them per your instructions.\n\n"
        f"{blocks}"
    )
    try:
        return await _call_judge_design(
            api_base, api_key, model_name, AGGREGATE_SUMMARY_SYSTEM, user, max_tokens=max_tokens
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"aggregate_error_summary failed: {type(e).__name__}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_PROMPT_RE = re.compile(r"<prompt>\s*(.*?)\s*</prompt>", re.DOTALL | re.IGNORECASE)
_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_evolved_prompt(text: str) -> str | None:
    """Extract the last <prompt>...</prompt> block; None if absent / empty."""
    if not text:
        return None
    matches = _PROMPT_RE.findall(text)
    if not matches:
        return None
    cand = matches[-1].strip()
    return cand or None


def parse_evolved_function(text: str) -> str | None:
    """Extract the function source: last ```python block containing function_reward.

    Falls back to a raw ``def function_reward`` span if the model omitted fences.
    """
    if not text:
        return None
    # Prefer fenced blocks that actually define function_reward.
    blocks = _PY_FENCE_RE.findall(text)
    cands = [b.strip() for b in blocks if "def function_reward" in b]
    if cands:
        return cands[-1].strip()
    if blocks:
        # A fenced block without the exact name (rare) — take the last fence.
        return blocks[-1].strip()
    # Unfenced fallback: grab from the last 'def function_reward'/leading import to EOF.
    idx = text.rfind("def function_reward")
    if idx == -1:
        return None
    # include any import lines immediately preceding the def
    head = text[:idx].rstrip().splitlines()
    lead = []
    for line in reversed(head):
        if re.match(r"\s*(import |from )", line):
            lead.insert(0, line)
        elif line.strip() == "":
            continue
        else:
            break
    src = ("\n".join(lead) + "\n" if lead else "") + text[idx:]
    return src.strip() or None


# ---------------------------------------------------------------------------
# Function validation (sandboxed in a fresh subprocess so we can kill loops)
# ---------------------------------------------------------------------------

# Child program: reads {"src","calls","out_path"} as JSON on stdin, execs the
# candidate, runs function_reward on each call, and writes the result JSON to
# out_path. Run via `python -c` (a fresh interpreter) so it never re-execs the
# parent's __main__ (which for `python -m verl.trainer...` would restart training),
# and so a CUDA-initialized parent is not forked. subprocess timeout SIGKILLs it,
# which terminates unbounded loops.
_CHILD_PROGRAM = r"""
import sys, json
data = json.load(sys.stdin)
src, calls, out_path = data["src"], data["calls"], data["out_path"]
res = {}
try:
    ns = {}
    exec(compile(src, "<evolved_function>", "exec"), ns)
    fn = ns.get("function_reward")
    if not callable(fn):
        res = {"status": "err", "err": "no callable 'function_reward' defined"}
    else:
        outs = [float(fn(c[0], c[1], c[2])) for c in calls]
        res = {"status": "ok", "outs": outs}
except Exception as e:
    res = {"status": "err", "err": f"{type(e).__name__}: {e}"}
with open(out_path, "w") as f:
    json.dump(res, f)
"""


def validate_function_src(
    src: str,
    examples: list[dict],
    timeout_s: float | None = None,
    max_calls: int = 3,
) -> tuple[bool, str, list[float] | None]:
    """Compile + run the candidate function on a few examples in a fresh subprocess.

    Returns (ok, error_message, outputs_on_examples). Rejects on syntax error,
    runtime exception, missing/uncallable function_reward, or timeout (unbounded
    loops are SIGKILLed by the subprocess timeout).

    The evolved function may now call the judge / query Milvus / download a tool, so the
    validation timeout is generous: ``REWARD_FN_VALIDATE_TIMEOUT`` (default = a few per-sample
    budgets, to allow a one-time model download on the first call) and we run fewer example
    calls (``max_calls``). The child subprocess inherits this process's environment, so the
    REWARD_JUDGE_* / EMBED_* / MILVUS_* credentials reach the candidate.
    """
    if timeout_s is None:
        timeout_s = float(os.environ.get("REWARD_FN_VALIDATE_TIMEOUT", str(max(90, FUNCTION_TIME_BUDGET_S * 3))))
    if not src or "def function_reward" not in src:
        return False, "no function_reward definition", None
    try:
        compile(src, "<evolved_function>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}", None

    calls = [
        [ex.get("question", ""), ex.get("response", ""), ex.get("ground_truth", "")]
        for ex in examples[:max_calls]
    ] or [["", "", ""]]
    out_path = tempfile.mktemp(suffix="_evolve_validate.json")
    payload = json.dumps({"src": src, "calls": calls, "out_path": out_path})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_PROGRAM],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        _safe_remove(out_path)
        return False, f"timeout after {timeout_s}s (likely an unbounded loop)", None
    except Exception as e:  # noqa: BLE001
        _safe_remove(out_path)
        return False, f"subprocess failed: {type(e).__name__}: {e}", None
    try:
        with open(out_path) as f:
            res = json.load(f)
    except Exception:  # noqa: BLE001
        stderr_tail = (proc.stderr or "")[-200:]
        return False, f"child produced no result (stderr: {stderr_tail})", None
    finally:
        _safe_remove(out_path)
    if res.get("status") == "ok":
        return True, "", res.get("outs")
    return False, str(res.get("err", "unknown error")), None


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Runtime: load + safely call the current function (used by compute_score)
# ---------------------------------------------------------------------------

_FN_CACHE: dict[str, object] = {}


def load_function(src: str):
    """Compile+exec function source, returning the callable. Cached by source hash."""
    key = str(hash(src))
    fn = _FN_CACHE.get(key)
    if fn is not None:
        return fn
    ns: dict = {}
    exec(compile(src, "<evolved_function>", "exec"), ns)  # noqa: S102 (docker, intended)
    fn = ns["function_reward"]
    _FN_CACHE[key] = fn
    return fn


def clamp01(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def safe_call_function(fn, question: str, response: str, ground_truth: str) -> float:
    """Call a validated function in-process, clamping to [0,1]; 0.0 on any failure.

    The function became 'current' only after passing ``validate_function_src`` (which
    rejected crashing / looping candidates in a child process), so an in-process call
    here is acceptable and fast for the per-sample reward path.
    """
    try:
        return clamp01(fn(question, response, ground_truth))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"function_reward raised at runtime: {type(e).__name__}: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Judge scoring with the (evolvable) judge prompt
# ---------------------------------------------------------------------------

def _extract_score_0_10(text: str, default: float = 0.0) -> float:
    """Pull a 0-10 score from a judge response (boxed preferred); return [0,1]."""
    boxed = extract_boxed_answer(text)
    for src in (boxed, text):
        if not src:
            continue
        m = re.search(r"(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)", src)
        if m:
            val = max(0.0, min(10.0, float(m.group(1))))
            return val / 10.0
    return default


def build_judge_user_message(question: str, response: str, ground_truth: str) -> str:
    return (
        f"[QUESTION]\n{question}\n\n"
        f"[MODEL RESPONSE]\n{response}\n\n"
        f"[CORRECT ANSWER]\n{ground_truth}\n\n"
        "Score the MODEL RESPONSE now. Output only \\boxed{N} with an integer 0-10."
    )


async def score_with_judge_prompt(
    api_base: str,
    api_key: str,
    model_name: str,
    judge_prompt: str,
    question: str,
    response: str,
    ground_truth: str,
    max_tokens: int = 512,
) -> float:
    """Run the evolvable judge prompt to score one response; returns [0,1] (0.0 on failure)."""
    if not api_base:
        return 0.0
    user = build_judge_user_message(question, response, ground_truth)
    try:
        content = await _call_api(api_base, api_key, model_name, judge_prompt, user, max_tokens=max_tokens)
        return _extract_score_0_10(content, default=0.0)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"score_with_judge_prompt failed: {type(e).__name__}: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Design-time judge call (reasoning ENABLED, long output) — used for evolution
# ---------------------------------------------------------------------------

async def _call_judge_design(
    api_base: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8000,
) -> str:
    """Chat call for evolution: reasoning is left ON (unlike _call_api) so the model
    can think, and we read reasoning_content + content combined so parsing works
    whether or not the server splits the reasoning block.
    """
    url = f"{api_base}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    provider = os.environ.get("CHAT_PROVIDER", "vllm").lower()
    payload: dict = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if provider == "trapi":
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.7  # a little diversity helps design exploration
    timeout_total = float(os.environ.get("REWARD_EVOLVE_TIMEOUT", "600"))
    timeout = aiohttp.ClientTimeout(total=timeout_total)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    msg = data["choices"][0]["message"]
                    reasoning = msg.get("reasoning_content") or ""
                    content = msg.get("content") or ""
                    return (reasoning + "\n" + content).strip()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"unreachable, last_err={last_err}")


# ---------------------------------------------------------------------------
# On-disk versioned store
# ---------------------------------------------------------------------------

class EvolutionStore:
    """Versioned store of (judge_prompt.txt, function_reward.py) under evolve_dir.

    Layout:
        {evolve_dir}/step_000/{judge_prompt.txt, function_reward.py}  # starters
        {evolve_dir}/step_001/{...}                                   # per-step
        {evolve_dir}/current/{judge_prompt.txt, function_reward.py}   # latest valid
    """

    PROMPT_NAME = "judge_prompt.txt"
    FUNC_NAME = "function_reward.py"

    def __init__(self, evolve_dir: str):
        self.root = evolve_dir
        self.current_dir = os.path.join(self.root, "current")

    def step_dir(self, step: int) -> str:
        return os.path.join(self.root, f"step_{step:03d}")

    def is_initialized(self) -> bool:
        return os.path.exists(os.path.join(self.current_dir, self.PROMPT_NAME))

    def _write_pair(self, dest_dir: str, prompt: str, fn_src: str) -> None:
        os.makedirs(dest_dir, exist_ok=True)
        self._atomic_write(os.path.join(dest_dir, self.PROMPT_NAME), prompt)
        self._atomic_write(os.path.join(dest_dir, self.FUNC_NAME), fn_src)

    @staticmethod
    def _atomic_write(path: str, content: str) -> None:
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def init_if_needed(self) -> None:
        """Seed step_000 + current with the starter artifacts (preserves the original)."""
        if self.is_initialized():
            return
        self._write_pair(self.step_dir(0), STARTER_JUDGE_PROMPT, STARTER_FUNCTION_SRC)
        self._write_pair(self.current_dir, STARTER_JUDGE_PROMPT, STARTER_FUNCTION_SRC)
        logger.info(f"[reward_evolution] initialized store at {self.root}")

    def read_current(self) -> tuple[str, str]:
        prompt_path = os.path.join(self.current_dir, self.PROMPT_NAME)
        fn_path = os.path.join(self.current_dir, self.FUNC_NAME)
        if not os.path.exists(prompt_path):
            return STARTER_JUDGE_PROMPT, STARTER_FUNCTION_SRC
        with open(prompt_path) as f:
            prompt = f.read()
        with open(fn_path) as f:
            fn_src = f.read()
        return prompt, fn_src

    def commit_step(self, step: int, prompt: str, fn_src: str) -> None:
        """Write step_{step} and atomically update current with the effective pair."""
        self._write_pair(self.step_dir(step), prompt, fn_src)
        self._write_pair(self.current_dir, prompt, fn_src)

    def write_step_aux(self, step: int, name: str, content: str) -> None:
        """Persist an auxiliary artifact (e.g. error_summary.txt) under step_{step}."""
        if not content:
            return
        self._atomic_write(os.path.join(self.step_dir(step), name), content)


# Runtime read-through cache for compute_score: re-read current/ only when files change.
_CURRENT_CACHE: dict[str, tuple[float, str, object, str]] = {}


def get_current_artifacts(evolve_dir: str):
    """Return (judge_prompt, function_callable, function_src) from current/, cached by mtime.

    Used in the per-sample reward path. Falls back to the starters if current/ is
    missing or the function fails to load.
    """
    store = EvolutionStore(evolve_dir)
    prompt_path = os.path.join(store.current_dir, store.PROMPT_NAME)
    fn_path = os.path.join(store.current_dir, store.FUNC_NAME)
    try:
        mtime = max(os.path.getmtime(prompt_path), os.path.getmtime(fn_path))
    except OSError:
        # Not initialized yet: use starters.
        return STARTER_JUDGE_PROMPT, load_function(STARTER_FUNCTION_SRC), STARTER_FUNCTION_SRC
    cached = _CURRENT_CACHE.get(evolve_dir)
    if cached and cached[0] == mtime:
        return cached[1], cached[2], cached[3]
    prompt, fn_src = store.read_current()
    try:
        fn = load_function(fn_src)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"current function failed to load ({e!r}); using starter")
        fn, fn_src = load_function(STARTER_FUNCTION_SRC), STARTER_FUNCTION_SRC
    _CURRENT_CACHE[evolve_dir] = (mtime, prompt, fn, fn_src)
    return prompt, fn, fn_src


# ---------------------------------------------------------------------------
# End-of-step evolution
# ---------------------------------------------------------------------------

async def evolve_once(
    api_base: str,
    api_key: str,
    model_name: str,
    evolve_dir: str,
    examples: list[dict],
    step: int,
    design_max_tokens: int = 8000,
    summary_max_tokens: int = 900,
    aggregate_max_tokens: int = 1400,
) -> dict:
    """Run one evolution round.

    Flow (new):
      1. For each of the (~20) sampled cases, ask the judge — IN THE CONTEXT of the current
         judge prompt + function — to summarize what the student got wrong and what the current
         reward setting missed (``summarize_case_error``, ~500 tokens each, run concurrently).
      2. Collapse those per-case summaries into one error summary + improvement suggestions
         (``aggregate_error_summary``, ~1000 tokens).
      3. Evolve BOTH the judge prompt and the function, feeding that error summary + suggestions
         (instead of raw student answers) as the design context.
    Then validate, apply the degeneracy gate, and commit step_{step} + update current. On
    parse/validation failure for either artifact, the previous version is carried forward.

    Returns a small metrics dict for logging.
    """
    store = EvolutionStore(evolve_dir)
    store.init_if_needed()
    cur_prompt, cur_fn_src = store.read_current()

    # --- 1. per-case error summaries (concurrent), each judging the current setting too ---
    case_summaries = await asyncio.gather(
        *[
            summarize_case_error(
                api_base, api_key, model_name, cur_prompt, cur_fn_src, ex, summary_max_tokens
            )
            for ex in examples
        ]
    )
    case_summaries = [s for s in case_summaries if s]

    # --- 2. aggregate into one error summary + improvement suggestions ---
    error_summary = await aggregate_error_summary(
        api_base, api_key, model_name, case_summaries, aggregate_max_tokens
    )
    store.write_step_aux(step, "error_summary.txt", error_summary)
    store.write_step_aux(
        step, "case_summaries.txt",
        "\n\n".join(f"### Case {i}\n{s}" for i, s in enumerate(case_summaries, 1)),
    )

    # --- 3. evolve prompt + function from the error summary (not raw student answers) ---
    p_sys, p_user = build_prompt_evolution_messages(cur_prompt, error_summary)
    f_sys, f_user = build_function_evolution_messages(cur_fn_src, error_summary)

    async def _gen(sys_p, usr_p):
        try:
            return await _call_judge_design(api_base, api_key, model_name, sys_p, usr_p, design_max_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[reward_evolution] design call failed: {type(e).__name__}: {e}")
            return ""

    prompt_raw, fn_raw = await asyncio.gather(_gen(p_sys, p_user), _gen(f_sys, f_user))

    # --- judge prompt ---
    new_prompt = parse_evolved_prompt(prompt_raw)
    prompt_valid = bool(new_prompt) and "\\boxed" in new_prompt
    eff_prompt = new_prompt if prompt_valid else cur_prompt
    prompt_changed = prompt_valid and (new_prompt.strip() != cur_prompt.strip())

    # --- function ---
    new_fn = parse_evolved_function(fn_raw)
    fn_valid, fn_err, fn_outs = (False, "no function parsed", None)
    if new_fn:
        fn_valid, fn_err, fn_outs = await asyncio.get_event_loop().run_in_executor(
            None, validate_function_src, new_fn, examples
        )
    # Degeneracy gate: a function that returns a near-constant value across examples that
    # DO vary in combined reward carries no gradient signal (it collapsed to a constant).
    # Reject it and keep the previous version rather than wasting the function component.
    fn_degenerate = False
    if fn_valid and fn_outs and len(fn_outs) >= 2:
        ex_rewards = [float(e.get("combined_reward", 0.0)) for e in examples[: len(fn_outs)]]
        ex_spread = max(ex_rewards) - min(ex_rewards)
        out_spread = max(fn_outs) - min(fn_outs)
        if ex_spread > 0.1 and out_spread < 0.05:
            fn_degenerate = True
            fn_valid = False
            fn_err = (
                f"degenerate: output spread {out_spread:.3f} on examples whose reward "
                f"spread is {ex_spread:.3f} — keeping previous function"
            )
    eff_fn = new_fn if fn_valid else cur_fn_src
    fn_changed = fn_valid and (new_fn.strip() != cur_fn_src.strip())

    store.commit_step(step, eff_prompt, eff_fn)

    fn_mean = (sum(fn_outs) / len(fn_outs)) if fn_outs else 0.0
    fn_spread = (max(fn_outs) - min(fn_outs)) if fn_outs else 0.0
    metrics = {
        "reward_evolution/step": step,
        "reward_evolution/prompt_valid": float(prompt_valid),
        "reward_evolution/prompt_changed": float(prompt_changed),
        "reward_evolution/fn_valid": float(fn_valid),
        "reward_evolution/fn_changed": float(fn_changed),
        "reward_evolution/fn_degenerate": float(fn_degenerate),
        "reward_evolution/fn_mean_on_examples": float(fn_mean),
        "reward_evolution/fn_spread_on_examples": float(fn_spread),
        "reward_evolution/num_case_summaries": float(len(case_summaries)),
        "reward_evolution/error_summary_chars": float(len(error_summary or "")),
    }
    if not fn_valid and new_fn:
        logger.warning(f"[reward_evolution] step {step}: function rejected: {fn_err}")
    logger.info(
        f"[reward_evolution] step {step}: prompt_valid={prompt_valid} changed={prompt_changed} "
        f"fn_valid={fn_valid} changed={fn_changed} fn_mean={fn_mean:.3f} spread={fn_spread:.3f}"
    )
    return metrics


def evolve_once_sync(**kwargs) -> dict:
    """Blocking wrapper around ``evolve_once`` for the synchronous trainer fit loop."""
    return asyncio.run(evolve_once(**kwargs))


# ---------------------------------------------------------------------------
# Helper for building example dicts from the existing reward sub-scores
# ---------------------------------------------------------------------------

_SUB_REWARD_KEYS = (
    "acc",
    "dynamic_judge",
    "dynamic_function",
    "judge_acc_lenient",
    "judge_acc_strict",
    "answer_quality",
    "reasoning_quality",
    "format_ok",
    "embed_sim",
    "char_bleu",
)


def make_example(question: str, response: str, ground_truth: str, score_row: dict) -> dict:
    """Build one evolution example from a question/response/GT + a row of sub-rewards.

    `score_row` is a dict like the per-sample reward dict (keys: score, acc, ...).
    """
    sub = {k: score_row[k] for k in _SUB_REWARD_KEYS if k in score_row}
    extracted = score_row.get("extracted_answer") or (extract_final_answer(response) or "")
    return {
        "question": question,
        "response": response,
        "ground_truth": ground_truth,
        "extracted_answer": extracted,
        "combined_reward": score_row.get("score", score_row.get("reward", 0.0)),
        "sub_rewards": sub,
    }


def select_contrastive(examples: list[dict], k: int = 6) -> list[dict]:
    """Pick a contrastive subset: lowest, highest, and spread-out middle by combined reward."""
    if len(examples) <= k:
        return list(examples)
    ordered = sorted(examples, key=lambda e: float(e.get("combined_reward", 0.0)))
    # always include the extremes; sample the rest evenly across the range
    idxs = sorted(set(round(i * (len(ordered) - 1) / (k - 1)) for i in range(k)))
    return [ordered[i] for i in idxs]


__all__ = [
    "STEER_SENTENCE",
    "STARTER_JUDGE_PROMPT",
    "STARTER_FUNCTION_SRC",
    "build_prompt_evolution_messages",
    "build_function_evolution_messages",
    "format_one_case",
    "summarize_case_error",
    "aggregate_error_summary",
    "FUNCTION_TOOLING_GUIDE",
    "PER_CASE_SUMMARY_SYSTEM",
    "AGGREGATE_SUMMARY_SYSTEM",
    "parse_evolved_prompt",
    "parse_evolved_function",
    "validate_function_src",
    "load_function",
    "safe_call_function",
    "clamp01",
    "score_with_judge_prompt",
    "build_judge_user_message",
    "EvolutionStore",
    "get_current_artifacts",
    "evolve_once",
    "evolve_once_sync",
    "make_example",
    "select_contrastive",
    "format_examples",
    "check_accuracy",
]
