"""Generate SELECTIVE retrieval SFT warm-start traces.

Root-cause fix for v7: the first warm-start SFT (make_retrieval_traces.py) put a
retrieval turn in EVERY trace, so the model learned to *always* search. Val proved
it — retrieval rate 1.00 on all 525 tasks (incl. writing/ethics/direct tasks that
need no lookup), dragging acc to 0.496 vs the 0.559 no-retrieval baseline. The
"retrieval is optional" prompt could not override the baked-in always-search habit.

This generator teaches the CHOICE. For each task the teacher first tries to answer
DIRECTLY (no tools, no rubric hint — a clean test of its parametric knowledge):

  * If the direct answer PASSES the rubric  -> keep a **direct trace**
        [user, assistant(<think>+answer)]           (tool declared but unused)
  * If the direct answer FAILS               -> the teacher then RETRIEVES and answers
        [user, assistant(<think>+search), tool(passages), assistant(<think>+answer)]
    kept as a **retrieval trace** only if the grounded answer PASSES and beats the
    direct score (i.e. retrieval genuinely rescued the failure).

Balanced targets (--n_direct / --n_retrieval) guarantee a mix, so the student learns
to retrieve *only when a direct answer would fail* — the selective behavior we want.
Both trace types declare the tool in `tools` (so it stays available at inference);
only retrieval traces actually call it.

Output jsonl of {messages, tools, extra_info{type}} for verl's MultiTurnSFTDataset.

Usage (run ON server 1, /retrieve + /sample are localhost:8006):
  python scripts/self_evolving/make_selective_traces.py \
    --gen_server http://localhost:8006 --retrieval_url http://localhost:8006/retrieve \
    --api_base http://point.dd.works:18184/v1 --model_name Qwen/Qwen3.6-27B \
    --api_key sk-xMByFeWLKB87wZ --n_direct 350 --n_retrieval 350 --concurrency 8 \
    --out /scratch/sheng/self_evolving/selective_sft_traces.jsonl
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime

import httpx

TOOL_NAME = "search_medical_kb"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search the medical knowledge base (textbooks, PubMed, clinical wikis) for "
            "evidence relevant to the clinician's request. Returns the most relevant passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused clinical search query."}
            },
            "required": ["query"],
        },
    },
}

# Direct attempt: NO tool instructions. Pure parametric-knowledge answer. This is the
# clean test of "does this task actually need a lookup?".
# NOTE: do NOT instruct "<think>" here — this endpoint returns no reasoning_content and
# a "<think>" instruction makes the model run away inside an unclosed think block and
# emit an EMPTY answer (observed: 59% empties). Natural mode -> complete answers. We
# synthesize the short <think> wrapper ourselves (as the original working traces did).
DIRECT_SYSTEM = (
    "You are a medical expert. Give a clear, clinically sound, complete answer, "
    "directly from your own knowledge."
)

# Retrieval attempt system prompt (only used when the direct answer failed).
TOOL_USE_SYSTEM = (
    "You are a medical expert with access to a knowledge-base search tool "
    f"`{TOOL_NAME}`. Always reason inside a <think>...</think> block. On your first "
    "turn, think briefly about what evidence you need, then emit EXACTLY ONE tool "
    "call in this exact format and nothing after it:\n"
    f"<tool_call>\n<function={TOOL_NAME}>\n<parameter=query>\nYOUR FOCUSED QUERY\n"
    "</parameter>\n</function>\n</tool_call>\n"
    "The query must name the key clinical entities and the specific fact you need."
)

# No "<think>" instruction (natural mode -> complete answers; we add the think wrapper).
TURN2_NUDGE = (
    "Now write your COMPLETE, clinically sound final answer. The passages AUGMENT your "
    "knowledge, they do not limit it: state all well-established facts you know even if "
    "they are not in the passages, and do NOT write disclaimers like 'the retrieved "
    "evidence does not contain...'. No meta-commentary about formatting — just the answer. "
    "Do not call any tools."
)


def _rubric_hint(rubric_items) -> str:
    cover, avoid = [], []
    for it in rubric_items:
        crit = it.get("criterion_text", it.get("criterion", "")).strip()
        if not crit:
            continue
        (avoid if float(it.get("points", 0)) < 0 else cover).append(crit)
    lines = []
    if cover:
        lines.append("A complete, correct answer addresses these clinical points "
                     "(integrate them naturally; do not quote or list them verbatim):")
        lines += [f"- {c}" for c in cover]
    if avoid:
        lines.append("Be sure to AVOID these errors:")
        lines += [f"- {c}" for c in avoid]
    return "\n".join(lines)


_TOOLCALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
_QUERY_RE = re.compile(r"<parameter=query>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)


def _toolcall_block(query: str) -> str:
    return (f"<tool_call>\n<function={TOOL_NAME}>\n<parameter=query>\n{query}\n"
            "</parameter>\n</function>\n</tool_call>")


GRADE_SYS = "You are a strict medical grader. Answer ONLY with a JSON object {\"criteria_met\": true|false}."
GRADE_TMPL = (
    "Conversation:\n{conv}\n\nResponse to grade:\n{ans}\n\nRubric criterion:\n{crit}\n\n"
    "Does the response satisfy this criterion? Reply with just {{\"criteria_met\": true}} or "
    "{{\"criteria_met\": false}}.")


def _user_text(prompt) -> str:
    if isinstance(prompt, list):
        for m in prompt:
            if m.get("role") == "user":
                c = m.get("content", "")
                return c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict))
    return str(prompt)


async def _chat(client, args, messages, max_tokens=1500, temperature=0.7, think=True, tools=None):
    body = {"model": args.model_name, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        body["tools"] = tools
    r = await client.post(
        f"{args.api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _tool_query(msg) -> str | None:
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        if fn.get("name") == TOOL_NAME:
            try:
                a = json.loads(fn.get("arguments") or "{}")
                if a.get("query"):
                    return str(a["query"])
            except Exception:
                pass
    qm = _QUERY_RE.search(msg.get("content") or "") or _QUERY_RE.search(msg.get("reasoning_content") or "")
    return qm.group(1).strip() if qm else None


async def _retrieve(client, args, query):
    try:
        r = await client.post(args.retrieval_url, json={"query": query, "top_k": args.top_k}, timeout=60)
        r.raise_for_status()
        return r.json().get("text") or ""
    except Exception:
        return ""


async def _grade(client, args, conv, ans, rubric_items) -> float:
    async def one(it):
        crit = it.get("criterion_text", it.get("criterion", ""))
        pts = float(it.get("points", 0))
        prompt = GRADE_TMPL.format(conv=conv[:5000], ans=ans[:7000], crit=f"[{pts:g}] {crit}")
        try:
            msg = await _chat(client, args, [{"role": "system", "content": GRADE_SYS},
                                             {"role": "user", "content": prompt}],
                              max_tokens=150, temperature=0.0, think=False)
            raw = msg.get("content") or msg.get("reasoning_content") or ""
            m = re.search(r'"criteria_met"\s*:\s*(true|false)', raw, re.I)
            met = bool(m) and m.group(1).lower() == "true"
        except Exception:
            met = False
        return pts, met
    results = await asyncio.gather(*[one(it) for it in rubric_items])
    achieved = sum(p for p, met in results if met)
    total_pos = sum(p for p, _ in results if p > 0) or 1.0
    return max(0.0, min(1.0, achieved / total_pos))


_DIRECT_THINK_FALLBACK = "I can answer this directly from established clinical knowledge without needing an external lookup."


async def make_one(client, args, entry, stats, buckets):
    """Return (type, trace_dict) or (None, None). type in {'direct','retrieval'}."""
    prompt = entry.get("prompt") or []
    ex = entry.get("extra_info") or {}
    rubric = ex.get("rubric_items") or []
    question = _user_text(prompt)
    if not question or not rubric:
        stats["skipped"] += 1
        return None, None
    conv = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in prompt if isinstance(m, dict))

    # ---------- DIRECT attempt ----------
    # Optionally fold the rubric hint into the direct prompt: the self-EVOLVED tasks are
    # deliberately hard/adversarial, so a hint-free direct answer clears the bar only ~4%
    # of the time (too slow to fill the bucket). With the hint the teacher produces a
    # strong direct answer -> fast, high-yield DIRECT traces that teach the model it CAN
    # answer without searching. (--direct_only skips the retrieval attempt entirely.)
    want_direct = buckets["direct"] < args.n_direct
    want_retr = (not args.direct_only) and buckets["retrieval"] < args.n_retrieval
    d_user = question
    if args.direct_hint:
        hint = _rubric_hint(rubric)
        if hint:
            d_user = f"{question}\n\n{hint}"
    md = await _chat(client, args, [{"role": "system", "content": DIRECT_SYSTEM},
                                    {"role": "user", "content": d_user}], max_tokens=2500)
    ans_d = _TOOLCALL_RE.sub("", (md.get("content") or "")).strip()
    score_d = 0.0
    if len(ans_d) >= 40:
        # --no_grade: the direct prompt already carried the rubric hint (the answer key),
        # so a hint-guided answer covers the criteria by construction. Skip the 6 grading
        # calls (7x less teacher load) and keep any substantive answer. Used for the fast
        # DIRECT-trace bucket; retrieval traces are still graded for the score>direct gate.
        if args.no_grade:
            score_d = 1.0
        else:
            score_d = await _grade(client, args, conv, ans_d, rubric)
    stats["direct_graded"] += 1

    if args.direct_only and score_d < args.min_score:
        stats["below_thresh_direct"] += 1
        return None, None

    if score_d >= args.min_score:
        # Task is answerable directly -> a DIRECT (no-search) trace.
        if not want_direct:
            stats["direct_overflow"] += 1
            return None, None
        think_d = (md.get("reasoning_content") or "").strip() or _DIRECT_THINK_FALLBACK
        t = f"<think>\n{think_d}\n</think>\n{ans_d}"
        messages = [{"role": "user", "content": question},
                    {"role": "assistant", "content": t}]
        stats["kept_direct"] += 1
        return "direct", {"messages": messages, "tools": [TOOL_SCHEMA],
                          "extra_info": {"type": "direct", "question_id": ex.get("question_id"),
                                         "use_case": ex.get("use_case"), "score": round(score_d, 3)}}

    # ---------- Direct failed -> RETRIEVAL attempt ----------
    if not want_retr:
        stats["retr_skipped_full"] += 1
        return None, None
    sys_msg = {"role": "system", "content": TOOL_USE_SYSTEM}
    m1 = await _chat(client, args, [sys_msg, {"role": "user", "content": question}],
                     max_tokens=900, tools=[TOOL_SCHEMA])
    query = _tool_query(m1) or question[:300]
    think1 = (m1.get("reasoning_content") or "").strip() or (
        f"To answer this precisely I need current clinical evidence on: {query[:160]}. "
        "Let me search the knowledge base before responding.")
    t1 = f"<think>\n{think1}\n</think>\n{_toolcall_block(query)}"
    passages = await _retrieve(client, args, query)
    if not passages or "No relevant passages" in passages:
        stats["no_passages"] += 1
    nudge = TURN2_NUDGE
    if args.rubric_hint:
        hint = _rubric_hint(rubric)
        if hint:
            nudge = f"{TURN2_NUDGE}\n\n{hint}"
    m2 = await _chat(client, args, [
        sys_msg,
        {"role": "user", "content": question},
        {"role": "assistant", "content": t1},
        {"role": "tool", "content": passages},
        {"role": "user", "content": nudge},
    ], max_tokens=2500)
    answer = _TOOLCALL_RE.sub("", (m2.get("content") or "")).strip()
    if len(answer) < 40:
        stats["empty_answer"] += 1
        return None, None
    score_r = await _grade(client, args, conv, answer, rubric)
    stats["retr_graded"] += 1
    # Keep only if retrieval PASSED and genuinely beat the direct attempt.
    if score_r < args.min_score or score_r <= score_d:
        stats["retr_no_gain"] += 1
        return None, None
    think2 = (m2.get("reasoning_content") or "").strip() or (
        "I'll ground the answer in the retrieved evidence, prioritizing safety and guideline "
        "concordance, and note where the evidence is incomplete.")
    t2 = f"<think>\n{think2}\n</think>\n{answer}"
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": t1},
        {"role": "tool", "content": passages},
        {"role": "assistant", "content": t2},
    ]
    stats["kept_retrieval"] += 1
    return "retrieval", {"messages": messages, "tools": [TOOL_SCHEMA],
                         "extra_info": {"type": "retrieval", "question_id": ex.get("question_id"),
                                        "use_case": ex.get("use_case"),
                                        "score": round(score_r, 3), "direct_score": round(score_d, 3),
                                        "query": query}}


async def _get_entry(client, args):
    try:
        r = await client.get(f"{args.gen_server.rstrip('/')}/sample", timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


async def main_async(args):
    stats = {k: 0 for k in ("skipped", "direct_graded", "kept_direct", "direct_overflow",
                            "below_thresh_direct", "retr_skipped_full", "no_passages",
                            "empty_answer", "retr_graded", "retr_no_gain", "kept_retrieval")}
    buckets = {"direct": 0, "retrieval": 0}
    sem = asyncio.Semaphore(args.concurrency)
    out_f = open(args.out, "w")

    def done():
        if args.direct_only:
            return buckets["direct"] >= args.n_direct
        return buckets["direct"] >= args.n_direct and buckets["retrieval"] >= args.n_retrieval

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency * 4)) as client:
        async def worker():
            while not done():
                entry = await _get_entry(client, args)
                if entry is None:
                    await asyncio.sleep(1)
                    continue
                async with sem:
                    try:
                        typ, trace = await make_one(client, args, entry, stats, buckets)
                    except Exception as e:
                        print(f"trace error: {type(e).__name__}: {e}", file=sys.stderr)
                        typ, trace = None, None
                if trace is not None and buckets[typ] < (args.n_direct if typ == "direct" else args.n_retrieval):
                    out_f.write(json.dumps(trace) + "\n")
                    out_f.flush()
                    buckets[typ] += 1
                    tot = buckets["direct"] + buckets["retrieval"]
                    if tot % 10 == 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] direct={buckets['direct']}/{args.n_direct} "
                              f"retrieval={buckets['retrieval']}/{args.n_retrieval}  stats={stats}", flush=True)

        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
    out_f.close()
    print(f"DONE: direct={buckets['direct']} retrieval={buckets['retrieval']} -> {args.out}\nstats={stats}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_server", default="http://localhost:8006")
    p.add_argument("--retrieval_url", default="http://localhost:8006/retrieve")
    p.add_argument("--api_base", default="http://point.dd.works:18184/v1")
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--model_name", default="Qwen/Qwen3.6-27B")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_direct", type=int, default=350)
    p.add_argument("--n_retrieval", type=int, default=350)
    p.add_argument("--min_score", type=float, default=0.7)
    p.add_argument("--rubric_hint", action="store_true", default=True)
    p.add_argument("--no_rubric_hint", dest="rubric_hint", action="store_false")
    p.add_argument("--direct_only", action="store_true", default=False,
                   help="Only produce direct traces (skip the retrieval attempt). Fast.")
    p.add_argument("--direct_hint", action="store_true", default=False,
                   help="Fold the rubric hint into the DIRECT prompt (raises direct yield on hard tasks).")
    p.add_argument("--no_grade", action="store_true", default=False,
                   help="Skip grading DIRECT answers (keep any non-empty hint-guided answer). 7x faster.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="/scratch/sheng/self_evolving/selective_sft_traces.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
