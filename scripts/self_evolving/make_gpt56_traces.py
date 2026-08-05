"""Generate warm-start SFT traces from a FRONTIER teacher (GPT-5.6) that match
the distributions the RL rollout actually samples.

Why a rewrite rather than a tweak of make_selective_traces.py
------------------------------------------------------------
The previous warm start (Qwen3.6-27B teacher) *hurt*: the SFT model's thinking
collapsed to ~110 chars (stock ~10k) and step-0 val fell to 0.488 vs the 0.559
no-tool baseline. Three causes, all fixed here:

1. FORMAT MISMATCH. Those traces trained a rendering the rollout never produced.
   The prompt strings are therefore imported from the agent-loop source, so the two
   cannot drift apart again.

   NOTE (2026-08-05): the loop is now a SINGLE CONTINUOUS TRAJECTORY — it no longer
   rebuilds a tool-free prompt for the answer, because that reset response_mask and
   deleted the query tokens from the gradient. So there is now exactly ONE shape to
   imitate, emitted below: system + question + [search turn + evidence + close
   instruction] + answer, with the tool schema present throughout.

2. DEGENERATE <think>. When the teacher returned no reasoning_content the old
   script substituted a canned one-liner, teaching "think one boilerplate
   sentence". GPT-5.6 does not expose its internal CoT, so we require it to WRITE
   substantive visible reasoning and gate on think length.

3. ANSWER-KEY LEAKAGE. --direct_hint fed the rubric to the teacher but stored a
   user turn without it, so the student imitated content it cannot derive. Here
   the teacher never sees the rubric; the rubric is used ONLY to grade afterwards,
   with the verbatim official HealthBench grader (the same one the RL reward uses,
   not an ad-hoc lenient prompt), plus anti-leak regexes ported from
   make_distill_traces.py.

Trace types emitted (one per example, both with tools=[TOOL_SCHEMA])
-------------------------------------------------------------------
  direct     [RETRIEVE_INSTRUCTION, question] -> <think>+answer
  retrieval  [RETRIEVE_INSTRUCTION, question] -> <think>+<tool_call>{queries:[...]}</tool_call>
             -> tool(evidence) -> HARD_ANSWER_INSTRUCTION -> <think>+answer

Both the query turn and the answer turn are trained (RL now trains both too), so the
warm start teaches the retrieve-or-not choice AND the query wording AND the answer.

GPT -> Qwen tool-call conversion
--------------------------------
GPT returns tool calls as structured JSON (`tool_calls[].function.arguments`).
The rollout parses Qwen's qwen3_coder XML dialect. `_toolcall_block` renders one
into the other; nothing downstream ever sees the OpenAI shape.

Usage (run where /retrieve and the gen server are reachable)
-----------------------------------------------------------
    python scripts/self_evolving/make_gpt56_traces.py \
      --gen_server http://localhost:8006 --retrieval_url http://localhost:8006/retrieve \
      --n_answer 700 --n_decision 300 --concurrency 4 \
      --out /scratch/sheng/self_evolving/gpt56_sft_traces.jsonl

TRAPI note: the deployment enforces a GLOBAL ~2000 req/60s cap shared with the
live training run's judges. Keep --concurrency small (<=6).
"""

import argparse
import ast
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

import httpx

TOOL_NAME = os.environ.get("RETRIEVAL_TOOL_NAME", "search_medical_kb")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AGENT_LOOP_SRC = os.path.join(
    REPO_ROOT, "verl", "experimental", "agent_loop", "retrieval_tool_agent_loop.py")
REWARD_SRC = os.path.join(REPO_ROOT, "verl", "utils", "reward_score", "healthbench_pro.py")


def _const_from_source(path: str, name: str) -> str:
    """Read a module-level string constant without importing the module.

    Importing the agent loop would drag in all of verl (torch, ray, vLLM); parsing
    keeps this script runnable anywhere while still making the RL source the single
    definition of these prompts."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        # f-string (e.g. RETRIEVE_FIRST_INSTRUCTION interpolates the
                        # tool name): join the literal parts and fill the tool name.
                        if isinstance(node.value, ast.JoinedStr):
                            out = []
                            for v in node.value.values:
                                if isinstance(v, ast.Constant):
                                    out.append(str(v.value))
                                else:
                                    out.append(TOOL_NAME)
                            return "".join(out)
                        raise
    raise KeyError(f"{name} not found in {path}")


HARD_ANSWER_INSTRUCTION = _const_from_source(AGENT_LOOP_SRC, "HARD_ANSWER_INSTRUCTION")
# The loop's system prompt carries a {max_searches} placeholder it fills at runtime.
RETRIEVE_INSTRUCTION = _const_from_source(AGENT_LOOP_SRC, "RETRIEVE_INSTRUCTION").format(
    max_searches=int(os.environ.get("VERL_MAX_SEARCHES", "2"))
)
GRADER_TEMPLATE = _const_from_source(REWARD_SRC, "GRADER_TEMPLATE")

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search the medical knowledge base (clinical review articles, FDA drug "
            "labels, patient-information topics, code tables, textbooks, PubMed) for "
            "evidence relevant to the request. Returns the most relevant passages."
        ),
        # Mirrors scripts/self_evolving/train/config/medical_retrieval_tool.yaml: the
        # tool takes a PLAN of 1-4 sub-queries. A single query supplies the graded
        # fact only 10.8% of the time vs 35.9% for two, so the warm start has to
        # teach the plan, not one query.
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "description": (
                        "1-4 search queries as a JSON array of strings. They must be "
                        "DIFFERENT FACETS of the request (core question / the exact "
                        "number / safety / the implied comparison), never rewordings "
                        "of each other. English only."
                    ),
                }
            },
            "required": ["queries"],
        },
    },
}

# --- teacher instructions --------------------------------------------------
# The think block must be substantive: the previous warm start's fatal flaw was
# teaching a one-sentence <think>. We ask for >=350 words and gate on length.
_THINK_SPEC = (
    "<think>\n"
    "Your genuine, thorough clinical reasoning in the first person, AT LEAST 350 words. "
    "Work through: what is actually being asked and what context is missing; the "
    "differential or the key decision; the specific evidence, thresholds, doses and "
    "guideline positions you are relying on; contraindications, interactions and "
    "red-flag/safety issues; and what a complete answer must cover. Reason as if "
    "solving the problem fresh. NEVER mention rubrics, criteria, grading, points, "
    "an answer key, or that anything was provided to you.\n"
    "</think>\n"
)

ANSWER_SYSTEM = (
    "You are an expert physician responding to a clinician colleague. Respond in "
    "EXACTLY this format: first a <think> block, then your final answer.\n\n"
    + _THINK_SPEC +
    "After </think>, write your COMPLETE final answer as plain prose for the "
    "colleague. Be as comprehensive as a thorough expert answer requires: state "
    "specific doses, thresholds and timeframes where they matter, ask for missing "
    "context when the request is ambiguous, include safety/red-flag guidance and "
    "contraindications, preserve appropriate diagnostic uncertainty, and refuse or "
    "redirect unsafe requests. Do not use placeholder text where a real clinical "
    "value belongs, and do not invent patient-specific data you were not given. "
    "No meta-commentary about formatting."
)

DECISION_SYSTEM = (
    RETRIEVE_INSTRUCTION
    + "\n\nRespond in EXACTLY this format: first a <think> block of at least 120 words "
    "in which you decide whether you genuinely need a knowledge-base lookup — say what "
    "specific fact you cannot recall with confidence, or why you can answer from your "
    "own knowledge. Then close with </think>. After that, EITHER emit exactly one call "
    f"to `{TOOL_NAME}` (if and only if a lookup is genuinely needed), OR write your "
    "complete final answer directly. When you do call the tool, pass 2-4 sub-queries "
    "covering DIFFERENT FACTS the answer needs (the core question, the exact "
    "number/threshold/dose, the safety angle) — never rewordings of one another, since "
    "each query returns only its own fact. Never mention rubrics, criteria or grading."
)


# --- format helpers --------------------------------------------------------
def _toolcall_block(queries: list[str]) -> str:
    """Render a tool call in the qwen3_coder XML dialect the rollout parses.

    The parameter is a JSON array because the tool now takes a PLAN; the parser
    routes a declared-`array` parameter through ast.literal_eval, so this is what it
    turns back into a real list at rollout time."""
    payload = json.dumps(list(queries), ensure_ascii=False)
    return (f"<tool_call>\n<function={TOOL_NAME}>\n<parameter=queries>\n{payload}\n"
            "</parameter>\n</function>\n</tool_call>")


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)

# Ported from make_distill_traces.py: prose that betrays the teacher was handed
# something the student will not have. We never show the rubric, but a teacher can
# still allude to "the criteria above" if it infers the setting — reject those.
# NOTE: these must be LEAK-specific, not generic-word matches. "criteria",
# "grading" and "points" are ordinary clinical vocabulary (inclusion criteria,
# Centor criteria, tumor grading, tender points) — matching them bare rejected a
# correct trace in testing and would have discarded most of the good data. Each
# pattern below requires evaluation-context wording, not just the word.
_LEAK_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"\brubric\b",
    r"\banswer\s+key\b",
    r"\bground[\s-]?truth\b",
    r"\bthe\s+grader\b",
    # "the criteria above", "the listed criteria", "grading criteria", "criteria
    # I need to meet" — evaluation talk, unlike "diagnostic criteria for X".
    r"\b(?:grading|scoring|evaluation)\s+criteri(?:on|a)\b",
    r"\bcriteri(?:on|a)\s+(?:above|listed|provided|given|supplied)\b",
    r"\b(?:the|these|those)\s+(?:listed|stated|provided|given)\s+criteri(?:on|a)\b",
    r"\b(?:meet|satisfy|cover|address)\s+(?:all\s+)?(?:the\s+)?(?:required\s+)?"
    r"criteri(?:on|a)\s+(?:above|listed|for\s+(?:full|maximum)\s+points)\b",
    r"\bpoints?\s+(?:are\s+)?(?:awarded|available|assigned)\b",
    r"\bfor\s+(?:full|maximum|partial)\s+(?:points|credit)\b",
    # "correct"/"known"/"confirmed" are dropped from this alternation on purpose:
    # "reaching the correct diagnosis" and "the known diagnosis" are ordinary
    # clinical prose and were rejecting good traces. Only wording that implies the
    # answer was HANDED to the writer stays.
    r"\bthe\s+(given|provided|supplied|reference|target|expected|gold)\s+"
    r"(answer|diagnosis|label)\b",
    r"\bas\s+(?:the\s+)?(?:points?|criteria|rubric)\s+(?:above\s+)?(?:indicate|require|state)\b",
    r"\bI\s+(?:am|was)\s+(?:being\s+)?(?:graded|scored|evaluated)\b",
)]

# Placeholder / fabrication tells. RL val showed a growing failure where the model
# invents "realistic example values" in note-writing tasks; traces that do it would
# teach exactly that.
_FABRICATION_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"\brealistic\s+(?:example|sample|placeholder)\s+values?\b",
    r"\b(?:populated|filled)\s+with\s+(?:realistic|example|sample|made[\s-]?up)\b",
    r"\bI\s+(?:have\s+)?(?:made\s+up|invented|fabricated)\b",
)]


def _split_think(text: str) -> tuple[str, str]:
    """Return (think, answer). Answer is everything after the LAST </think>,
    mirroring the reward's _strip_thinking so gating sees what grading sees."""
    text = text or ""
    m = _THINK_RE.search(text)
    think = m.group(1).strip() if m else ""
    low = text.lower()
    close = low.rfind("</think>")
    answer = text[close + len("</think>"):].strip() if close != -1 else text.strip()
    return think, answer


def _leaks(text: str) -> str | None:
    for rx in _LEAK_RES:
        if rx.search(text or ""):
            return rx.pattern[:40]
    for rx in _FABRICATION_RES:
        if rx.search(text or ""):
            return "fabrication:" + rx.pattern[:30]
    return None


def _user_text(prompt) -> str:
    """Extract the clinician turn. Never str() a container: a stringified
    {'messages': [...]} would be baked into the trace as the question."""
    if isinstance(prompt, dict):
        prompt = prompt.get("messages") or []
    if isinstance(prompt, list):
        for m in prompt:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return ""
    return prompt if isinstance(prompt, str) else ""


# --- teacher / judge calls -------------------------------------------------
async def _chat(client, args, messages, max_tokens=6000, tools=None, model=None):
    body = {
        "model": model or args.teacher_model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    for attempt in range(args.retries):
        try:
            r = await client.post(
                f"{args.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {args.api_key}"},
                json=body, timeout=args.timeout)
            if r.status_code == 429:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]
        except Exception as e:
            if attempt == args.retries - 1:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
            _ = e
    return {}


def _tool_queries(msg) -> list[str]:
    """Pull the query PLAN out of an OpenAI-shaped tool call (or inline XML).

    Accepts `queries` (list or JSON string) and the legacy `query`, so a teacher that
    ignores the array schema still produces a usable trace."""
    def _norm(raw) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            t = raw.strip()
            if t.startswith("["):
                try:
                    raw = json.loads(t)
                except Exception:
                    raw = t.splitlines()
            else:
                raw = t.splitlines()
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        out, seen = [], set()
        for q in raw:
            q = str(q).strip().lstrip("-*0123456789. ").strip(chr(34)).strip()
            if len(q) >= 4 and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out[:4]

    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        if fn.get("name") == TOOL_NAME:
            try:
                a = json.loads(fn.get("arguments") or "{}")
                qs = _norm(a.get("queries")) or _norm(a.get("query"))
                if qs:
                    return qs
            except Exception:
                pass
    m = re.search(r"<parameter=(?:queries|query)>\s*(.*?)\s*</parameter>",
                  msg.get("content") or "", re.DOTALL)
    return _norm(m.group(1)) if m else []


async def _retrieve(client, args, queries: list[str], question: str = ""):
    """Retrieve passages exactly as a rollout would, for a whole query PLAN.

    PREFER the /retrieve path. It is the only one that does the multi-query merge and
    the question-conditioned summarization, so it is the only one that produces the
    evidence text a rollout actually sees — and a trace built from different text
    teaches a distribution that does not exist at train time (the whole premise of
    kb/retrieval.py). --milvus_uri is an offline fallback for when no gen server is
    reachable; it merges locally but cannot summarize."""
    if args.milvus_uri:
        try:
            from kb.retrieval import RetrieveConfig, format_passages, local_search, merge_ranked
            cfg = RetrieveConfig()
            per_query = []
            for q in queries:
                passages, _ = await asyncio.to_thread(
                    local_search, q, args.top_k, args.milvus_uri, args.milvus_token,
                    args.milvus_collection, args.embed_api_base, args.embed_api_key,
                    args.embed_model, cfg)
                per_query.append(passages)
            merged, _stats = merge_ranked(per_query)
            return format_passages(merged)
        except Exception as e:
            print(f"local_search failed: {type(e).__name__}: {e}", file=sys.stderr)
            return ""
    try:
        body = {"queries": queries}
        if question:
            body["question"] = question[:4000]
        if args.top_k:
            body["top_k"] = args.top_k
        r = await client.post(args.retrieval_url, json=body, timeout=180)
        r.raise_for_status()
        return r.json().get("text") or ""
    except Exception:
        return ""


async def _grade(client, args, question, answer, rubric_items) -> float | None:
    """Score with the VERBATIM official HealthBench grader — the same prompt the RL
    reward uses, so a trace that passes here is calibrated to the training signal."""
    conversation = f"user: {question}\n\nassistant: {answer}"

    async def one(it):
        crit = it.get("criterion_text", it.get("criterion", ""))
        pts = float(it.get("points", 0))
        prompt = (GRADER_TEMPLATE
                  .replace("<<conversation>>", conversation)
                  .replace("<<rubric_item>>", f"[{pts:g}] {crit}"))
        try:
            msg = await _chat(client, args, [{"role": "user", "content": prompt}],
                              max_tokens=1200, model=args.judge_model)
            raw = (msg.get("content") or "").strip()
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw)
            try:
                met = json.loads(cleaned).get("criteria_met")
            except Exception:
                m = re.search(r'"criteria_met"\s*:\s*(true|false)', cleaned, re.I)
                met = (m.group(1).lower() == "true") if m else None
            return pts, (met if isinstance(met, bool) else None)
        except Exception:
            return pts, None

    results = await asyncio.gather(*[one(it) for it in rubric_items])
    # A judge failure is NOT "not met" (that silently deflates positives and gives
    # negative criteria a free pass) — drop the item and grade on the rest.
    usable = [(p, m) for p, m in results if m is not None]
    if len(usable) < max(2, len(results) // 2):
        return None
    achieved = sum(p for p, met in usable if met)
    total_pos = sum(p for p, _ in usable if p > 0)
    if total_pos <= 0:
        return None
    return max(0.0, min(1.0, achieved / total_pos))


# --- per-task pipeline -----------------------------------------------------
async def make_one(client, args, entry, stats, buckets):
    prompt = entry.get("prompt") or []
    ex = entry.get("extra_info") or {}
    rubric = ex.get("rubric_items") or []
    question = _user_text(prompt)
    if not question or not rubric:
        stats["skipped_no_task"] += 1
        return []

    out = []

    # ---- phase 1: the retrieve-or-not decision (tools exposed, GPT shape) ----
    decision = await _chat(
        client, args,
        [{"role": "system", "content": DECISION_SYSTEM},
         {"role": "user", "content": question}],
        max_tokens=args.decision_tokens, tools=[TOOL_SCHEMA])
    d_content = decision.get("content") or ""
    d_think, d_rest = _split_think(d_content)
    queries = _tool_queries(decision)
    searched = bool(queries)

    passages = ""
    if searched:
        passages = await _retrieve(client, args, queries, question)
        if not passages or "No relevant passages" in passages:
            stats["no_passages"] += 1
            passages = ""

    # ---- phase 2: the GRADED answer, in the exact rendering RL samples ----
    ctx = f"\n\nReference passages retrieved for this question:\n{passages}" if passages else ""
    answer_user = f"{question}{ctx}\n\n{HARD_ANSWER_INSTRUCTION}"
    ans_msg = await _chat(
        client, args,
        [{"role": "system", "content": ANSWER_SYSTEM},
         {"role": "user", "content": answer_user}],
        max_tokens=args.answer_tokens)
    a_think, answer = _split_think(ans_msg.get("content") or "")
    answer = _TOOLCALL_RE.sub("", answer).strip()

    # ---- gates ----
    if len(a_think) < args.min_think_chars:
        stats["think_too_short"] += 1
        return []
    if len(answer) < args.min_answer_chars:
        stats["answer_too_short"] += 1
        return []
    hit = _leaks(a_think) or _leaks(answer)
    if hit:
        # Record WHICH pattern fired: an over-broad gate silently discards good
        # data, and the aggregate "leak" counter alone hides that.
        stats["leak"] += 1
        stats[f"leak:{hit}"] += 1
        return []

    score = await _grade(client, args, question, answer, rubric)
    if score is None:
        stats["judge_unusable"] += 1
        return []
    stats["graded"] += 1
    if score < args.min_score:
        stats["below_thresh"] += 1
        return []

    # ---- ONE continuous trace, matching the trajectory RL actually samples ----
    # The loop no longer rebuilds a tool-free prompt for the answer: the query turn,
    # the (loss-masked) evidence and the answer are a single sequence, and the query
    # tokens are trained. So the warm start must be that same single sequence, with
    # the tool schema present throughout — emitting the old two shapes would teach a
    # rendering the rollout never produces.
    kind = "retrieval" if passages else "direct"
    if buckets[kind] >= args.n_answer_each:
        return out

    messages = [{"role": "system", "content": RETRIEVE_INSTRUCTION},
                {"role": "user", "content": question}]
    if passages:
        # The search turn must be well-formed or the trace teaches a malformed call.
        if len(d_think) < args.min_decision_think or _leaks(d_think):
            stats["decision_rejected"] += 1
            return out
        messages += [
            {"role": "assistant", "content": f"<think>\n{d_think}\n</think>\n{_toolcall_block(queries)}"},
            {"role": "tool", "content": passages},
            {"role": "user", "content": HARD_ANSWER_INSTRUCTION},
        ]
    messages.append({"role": "assistant", "content": f"<think>\n{a_think}\n</think>\n{answer}"})

    out.append({
        "messages": messages,
        # The schema IS in the prompt for the whole trajectory now, including the
        # answer turn, so the trace has to carry it: a tool-free rendering would
        # train the model on a prompt distribution the rollout never sees.
        "tools": [TOOL_SCHEMA],
        "extra_info": {"type": kind, "score": round(score, 3),
                       "question_id": ex.get("question_id"),
                       "use_case": ex.get("use_case"),
                       "queries": queries},
    })
    return out


def load_tasks_file(path: str) -> list[dict]:
    """Load tasks from a healthbench_gen.py annotation JSONL.

    Preferred over pulling /sample from the live gen server: /sample POPS entries
    from the pool the running trainer is consuming, so generating traces against
    it would starve the live run."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # Three shapes in circulation: the gen server's /sample
            # {prompt:[msg], extra_info:{rubric_items}}, healthbench_gen's
            # {conversation:{messages:[msg]}, rubric_items}, and the flat HF
            # HealthBench {prompt:[msg], rubrics}. Normalise to a message list —
            # a dict slipping through here gets str()'d into the question text.
            prompt = d.get("prompt") or d.get("conversation") or []
            if isinstance(prompt, dict):
                prompt = prompt.get("messages") or []
            if not isinstance(prompt, list):
                continue
            ex = d.get("extra_info") or {}
            rubric = (ex.get("rubric_items") or d.get("rubric_items")
                      or d.get("rubrics") or [])
            if not prompt or not rubric:
                continue
            tasks.append({
                "prompt": prompt,
                "extra_info": {
                    "rubric_items": rubric,
                    "question_id": ex.get("question_id") or d.get("prompt_id") or "",
                    "use_case": ex.get("use_case") or (d.get("tags") or [None])[0] or "",
                },
            })
    return tasks


async def _get_entry(client, args, task_iter=None):
    if task_iter is not None:
        return next(task_iter, None)
    try:
        r = await client.get(f"{args.gen_server.rstrip('/')}/sample", timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


async def main_async(args):
    stats = Counter()
    buckets = Counter()
    # One continuous trace per example now; the split is only whether it searched.
    kinds = ("direct", "retrieval")
    for k in kinds:
        buckets[k] = 0
    out_f = open(args.out, "a" if args.append else "w")
    t0 = time.time()

    def done():
        return buckets["direct"] + buckets["retrieval"] >= args.n_answer

    task_iter = None
    if args.tasks_file:
        tasks = load_tasks_file(args.tasks_file)
        print(f"loaded {len(tasks)} tasks from {args.tasks_file}")
        if not tasks:
            print("ERROR: no usable tasks (need prompt + rubric)", file=sys.stderr)
            sys.exit(2)
        task_iter = iter(tasks)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency * 4)) as client:
        async def worker(wid):
            while not done():
                async with lock:  # `next()` on a shared iterator is not task-safe
                    entry = await _get_entry(client, args, task_iter)
                if entry is None:
                    if task_iter is not None:
                        return  # task file exhausted
                    await asyncio.sleep(1)
                    continue
                async with sem:
                    try:
                        traces = await make_one(client, args, entry, stats, buckets)
                    except Exception as e:
                        stats[f"error_{type(e).__name__}"] += 1
                        print(f"[w{wid}] {type(e).__name__}: {e}", file=sys.stderr)
                        traces = []
                for t in traces:
                    kind = t["extra_info"]["type"]
                    out_f.write(json.dumps(t) + "\n")
                    out_f.flush()
                    buckets[kind] += 1
                if traces:
                    tot = sum(buckets[k] for k in kinds)
                    if tot % 10 < len(traces):
                        el = time.time() - t0
                        print(f"[{datetime.now():%H:%M:%S}] {dict(buckets)} "
                              f"({tot / max(el / 60, 0.01):.1f}/min) stats={dict(stats)}",
                              flush=True)

        await asyncio.gather(*[worker(i) for i in range(args.concurrency)])
    out_f.close()
    print(f"DONE {dict(buckets)} -> {args.out}\nstats={dict(stats)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_server", default="http://localhost:8006")
    p.add_argument("--retrieval_url", default="http://localhost:8006/retrieve")
    p.add_argument("--tasks_file", default="",
                   help="healthbench_gen.py annotation JSONL. Preferred over /sample, "
                        "which pops entries from the live trainer's task pool.")
    # Offline retrieval: query Milvus directly with the same ranking policy as
    # /retrieve, so no running gen server is needed to build traces.
    p.add_argument("--milvus_uri", default="", help="e.g. http://localhost:19531")
    p.add_argument("--milvus_token", default=os.environ.get("MILVUS_TOKEN", "root:Milvus"))
    p.add_argument("--milvus_collection", default="medical_knowledge_v2")
    p.add_argument("--embed_api_base", default=os.environ.get("EMBED_API_BASE", "http://localhost:18001/v1"))
    p.add_argument("--embed_api_key", default=os.environ.get("EMBED_API_KEY", "EMPTY"))
    p.add_argument("--embed_model", default=os.environ.get("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))
    p.add_argument("--api_base", default="http://point.dd.works:18890/v1")
    p.add_argument("--api_key", default=os.environ.get("TRAPI_API_KEY", ""))
    p.add_argument("--teacher_model", default="gpt-5.6-sol_2026-07-09")
    p.add_argument("--judge_model", default="gpt-chat-latest_2026-05-28")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_answer", type=int, default=700, help="total answer traces")
    p.add_argument("--n_answer_each", type=int, default=500, help="cap per answer subtype")
    p.add_argument("--n_decision", type=int, default=300, help="total decision traces")
    p.add_argument("--n_decision_each", type=int, default=200, help="cap per decision subtype")
    p.add_argument("--min_score", type=float, default=0.7)
    p.add_argument("--min_think_chars", type=int, default=900)
    p.add_argument("--min_decision_think", type=int, default=250)
    p.add_argument("--min_answer_chars", type=int, default=400)
    p.add_argument("--answer_tokens", type=int, default=6000)
    p.add_argument("--decision_tokens", type=int, default=2500)
    p.add_argument("--concurrency", type=int, default=4,
                   help="keep small: TRAPI has a global cap shared with the live run")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--append", action="store_true")
    p.add_argument("--out", default="/scratch/sheng/self_evolving/gpt56_sft_traces.jsonl")
    a = p.parse_args()
    if not a.api_key:
        print("ERROR: pass --api_key or set TRAPI_API_KEY", file=sys.stderr)
        sys.exit(2)
    return a


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
