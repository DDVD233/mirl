"""Generate retrieval-augmented SFT warm-start traces.

Teaches the model the concise search->answer format used by RetrievalToolAgentLoop
BEFORE RL, so RL starts above (not below) the no-retrieval baseline and rollouts are
less verbose (faster). The server-5 Qwen3.6-27B teacher replays the exact 2-turn
rollout on the gen server's co-generated tasks:

    turn 1 (assistant): <think>brief</think> + <tool_call>search_medical_kb(query)</tool_call>
    tool  turn         : retrieved passages (masked in SFT loss)
    turn 2 (assistant): <think>reasoning</think> + grounded answer

Each trace is graded against its task's rubric (server-5 judge, same protocol as the
RL reward); only traces scoring >= --min_score are kept. Output is a jsonl of
{messages, tools, extra_info} consumable by verl's MultiTurnSFTDataset (loss only on
the assistant turns; the tool turn is masked).

Usage (run ON server 1, where /retrieve is localhost:8006):
  python scripts/self_evolving/make_retrieval_traces.py \
    --gen_server http://localhost:8006 --retrieval_url http://localhost:8006/retrieve \
    --api_base http://point.dd.works:18184/v1 --model_name Qwen/Qwen3.6-27B \
    --api_key $(cat /scratch/sheng/self_evolving/.climb_teacher_key) \
    --n_traces 800 --min_score 0.7 --concurrency 8 \
    --out /scratch/sheng/self_evolving/retrieval_sft_traces.jsonl
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

TOOL_USE_SYSTEM = (
    "You are a medical expert with access to a knowledge-base search tool "
    f"`{TOOL_NAME}`. Always reason inside a <think>...</think> block. On your first "
    "turn, think briefly about what evidence you need, then emit EXACTLY ONE tool "
    "call in this exact format and nothing after it:\n"
    f"<tool_call>\n<function={TOOL_NAME}>\n<parameter=query>\nYOUR FOCUSED QUERY\n"
    "</parameter>\n</function>\n</tool_call>\n"
    "The query must name the key clinical entities and the specific fact you need."
)

TURN2_NUDGE = (
    "Now write your COMPLETE final answer. Reason briefly in a <think>...</think> block, "
    "then give a clear, clinically sound answer. The passages AUGMENT your knowledge, they "
    "do not limit it: state all well-established facts you know even if they are not in the "
    "passages, and do NOT write disclaimers like 'the retrieved evidence does not contain...'. "
    "No meta-commentary about formatting — just the answer. Do not call any tools."
)


def _rubric_hint(rubric_items) -> str:
    """The rubric IS the answer key. Fold it into the answer-turn prompt as guidance
    so the teacher's answer covers the graded points (raises trace quality + yield).
    Phrased as clinical guidance, not verbatim, so the student learns behavior not
    checklist-parroting (and the student never sees the rubric at RL/eval time)."""
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


def _toolcall_block(query: str) -> str:
    return (f"<tool_call>\n<function={TOOL_NAME}>\n<parameter=query>\n{query}\n"
            "</parameter>\n</function>\n</tool_call>")

GRADE_SYS = "You are a strict medical grader. Answer ONLY with a JSON object {\"criteria_met\": true|false}."
GRADE_TMPL = (
    "Conversation:\n{conv}\n\nResponse to grade:\n{ans}\n\nRubric criterion:\n{crit}\n\n"
    "Does the response satisfy this criterion? Reply with just {{\"criteria_met\": true}} or "
    "{{\"criteria_met\": false}}.")

_QUERY_RE = re.compile(r"<parameter=query>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)


def _user_text(prompt) -> str:
    if isinstance(prompt, list):
        for m in prompt:
            if m.get("role") == "user":
                c = m.get("content", "")
                return c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict))
    return str(prompt)


async def _chat(client, args, messages, max_tokens=1500, temperature=0.7, think=True, tools=None):
    """Return the full assistant message dict. The teacher runs with a reasoning
    parser, so real reasoning is in `reasoning_content` and the answer in `content`
    (and a tool call, if any, in structured `tool_calls`) — callers reconstruct the
    <think>...</think> trace from these."""
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
    """Extract a search query from a structured tool_call or inline <parameter=query>."""
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
    except Exception as e:
        return ""


async def _grade(client, args, conv, ans, rubric_items) -> float:
    """Grade against each rubric criterion (parallel per-criterion binary calls)."""
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


async def make_one(client, args, entry, stats) -> dict | None:
    prompt = entry.get("prompt") or []
    ex = entry.get("extra_info") or {}
    rubric = ex.get("rubric_items") or []
    question = _user_text(prompt)
    if not question or not rubric:
        stats["skipped"] += 1
        return None
    sys_msg = {"role": "system", "content": TOOL_USE_SYSTEM}
    # Turn 1: teacher reasons + emits the search tool call (tools exposed so it
    # returns a structured tool_call; reasoning comes back in reasoning_content).
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
    # Turn 2: continue the conversation; light user nudge to answer. The rubric (the
    # answer key) is folded in as clinical guidance when --rubric_hint is on.
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
    ], max_tokens=1500)
    answer = _TOOLCALL_RE.sub("", (m2.get("content") or "")).strip()
    if len(answer) < 40:
        stats["empty_answer"] += 1
        return None
    think2 = (m2.get("reasoning_content") or "").strip()
    if not think2:
        variants = [
            "The retrieved passages give me the key evidence. Let me integrate them with clinical "
            "reasoning and answer the request directly, flagging any gaps or wrong premises.",
            "I'll ground the answer in the retrieved evidence, prioritizing safety and guideline "
            "concordance, and note where the evidence is incomplete.",
            "Synthesizing the retrieved evidence with the clinical context to give a precise, "
            "actionable answer, correcting any inaccurate assumptions in the request.",
        ]
        think2 = variants[len(answer) % len(variants)]
    t2 = f"<think>\n{think2}\n</think>\n{answer}"
    # Grade the final answer against the rubric.
    conv = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in prompt if isinstance(m, dict))
    score = await _grade(client, args, conv, answer, rubric)
    stats["graded"] += 1
    if score < args.min_score:
        stats["below_thresh"] += 1
        return None
    stats["kept"] += 1
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": t1},
        {"role": "tool", "content": passages},
        {"role": "assistant", "content": t2},
    ]
    return {"messages": messages, "tools": [TOOL_SCHEMA],
            "extra_info": {"question_id": ex.get("question_id"), "use_case": ex.get("use_case"),
                           "score": round(score, 3), "query": query}}


async def _get_entry(client, args):
    try:
        r = await client.get(f"{args.gen_server.rstrip('/')}/sample", timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


async def main_async(args):
    stats = {k: 0 for k in ("skipped", "no_passages", "empty_answer", "graded", "below_thresh", "kept")}
    sem = asyncio.Semaphore(args.concurrency)
    out_f = open(args.out, "w")
    n_written = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency * 4)) as client:
        async def worker():
            nonlocal n_written
            while n_written < args.n_traces:
                entry = await _get_entry(client, args)
                if entry is None:
                    await asyncio.sleep(1)
                    continue
                async with sem:
                    try:
                        trace = await make_one(client, args, entry, stats)
                    except Exception as e:
                        print(f"trace error: {type(e).__name__}: {e}", file=sys.stderr)
                        trace = None
                if trace is not None and n_written < args.n_traces:
                    out_f.write(json.dumps(trace) + "\n")
                    out_f.flush()
                    n_written += 1
                    if n_written % 10 == 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] kept {n_written}/{args.n_traces}  stats={stats}")

        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
    out_f.close()
    print(f"DONE: wrote {n_written} traces to {args.out}\nstats={stats}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_server", default="http://localhost:8006")
    p.add_argument("--retrieval_url", default="http://localhost:8006/retrieve")
    p.add_argument("--api_base", default="http://point.dd.works:18184/v1")
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--model_name", default="Qwen/Qwen3.6-27B")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_traces", type=int, default=800)
    p.add_argument("--min_score", type=float, default=0.7)
    p.add_argument("--rubric_hint", action="store_true", default=True,
                   help="Feed the rubric (answer key) into the answer turn as guidance.")
    p.add_argument("--no_rubric_hint", dest="rubric_hint", action="store_false")
    p.add_argument("--best_of", type=int, default=1,
                   help="Generate N answer turns per task, keep the highest-scoring.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="/scratch/sheng/self_evolving/retrieval_sft_traces.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
