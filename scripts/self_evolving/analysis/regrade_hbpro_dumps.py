#!/usr/bin/env python3
"""Re-grade HealthBench Professional validation dumps with a chosen grader.

Why: every number in the stage-2 tables was graded in-loop by the training-time
validation grader (gpt-chat-latest, or the frozen 9B in the self-judge settings). This
script re-grades the SAME answers under one common grader (default: the benchmark's
official GPT-5.4 grader at reasoning effort low) so that (a) rows graded by different
instruments become comparable and (b) the gain is shown not to be an artifact of the
grader the policy trained against. It can also apply the STRICT note used during
adversarial probes, which credits only content the answer itself states; the gap
between strict and standard grading of the same answers is a hollow-credit measure.

Protocol (identical to verl/utils/reward_score/healthbench_pro.py, itself the verbatim
simple-evals template):
  - answer = what the in-loop reward graded: for retrieval-agent rollouts the LAST
    assistant turn's text after its last </think> with tool spans removed (the loop's
    `graded_answer`, rebuilt from the dump), otherwise the think-stripped response;
    the dump's 512-char `extracted_answer` prefix decides which, per row. (Before
    2026-09-14 this script stripped tool spans over the WHOLE response first, and a
    budget-exhausted rollout's open <tool_call> then swallowed the answer: 0.0 rows.)
  - conversation = clinician turns from the val parquet (row i of the dump is row i of
    the parquet; the join is verified per file and the run aborts below --join-min);
  - one grader call per rubric criterion, "[points] criterion", majority of --votes;
  - raw = sum(points met) / sum(positive points); length-adjusted signed =
    raw - 0.0147 * (chars - 2000) / 500; clipped variants as in the benchmark.

Resumable: per-(row, item, vote) verdicts are appended to <out>.verdicts.jsonl and
reused on rerun, so a rate-limit crash costs nothing.

Example (on an MSR pod, TRAPI proxy):
  python3 regrade_hbpro_dumps.py --dump .../val_generations/<run>/<step>.jsonl \
      --val-parquet /scratch/sheng/self_evolving/healthbench_pro_val.parquet \
      --api-base http://point.dd.works:18890/v1 --api-key $(cat .trapi_key) \
      --model gpt-5.4_2026-03-05 --effort low --out out/<run>_<step>.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import aiohttp

GRADER_SYSTEM = "You are a careful grader. Return only the requested JSON object."

# Verbatim copy of the template in healthbench_pro.py (which copies simple-evals).
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

# Verbatim _GRADER_STRICT_NOTE from scripts/self_evolving/generation_server.py (the note
# appended during adversarial probes and patch validation), so a strict re-grade uses the
# exact wording the repair loop used.
STRICT_NOTE = (
    "\nNote: a criterion may describe something UNDESIRABLE and carry negative points. "
    "Report whether the criterion is MET, not whether the response is good. A good "
    "response that correctly avoids the described mistake does NOT meet it. "
    "Judge only the content the response itself states. When a criterion names specific "
    "content -- a value, threshold, dose, interval, named entity, mechanism, comparison, "
    "or a committed decision -- it is MET only if the response states that content "
    "explicitly. Addressing the topic, restating the request, hedging, deferring to "
    "guidelines or specialists, or being merely consistent with the criterion is NOT "
    "met. If a criterion has several parts, every part must be met. "
    'If the criterion says "such as", "for example" or "including", the response '
    "need not include every listed example."
)

LENGTH_CENTER = 2000.0
LENGTH_PENALTY_PER_500 = 0.0147

_TOOL_SPAN_RE = re.compile(
    r"<tool_call>.*?(?:</tool_call>|\Z)|<tool_response>.*?(?:</tool_response>|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    text = _TOOL_SPAN_RE.sub("", text)
    low = text.lower()
    close = low.rfind("</think>")
    if close != -1:
        return text[close + len("</think>"):].strip()
    open_i = low.find("<think>")
    if open_i != -1:
        return text[:open_i].strip()
    return text.strip()


def final_turn_answer(text: str) -> str:
    """The retrieval agent loop's `graded_answer`, rebuilt from the dump's decoded
    response: the LAST assistant turn (after the final tool response's "assistant"
    marker), minus the chat template's leading <think> that the loop's own decode does
    not contain, then tool spans removed and the text after the last </think> -- the
    same steps as _final_answer_of in verl/experimental/agent_loop/retrieval_tool_agent_loop.py.
    Applying the span regex to the WHOLE response instead is wrong: a rollout whose
    search budget ran out leaves a bare <tool_call> before the injected error, and the
    regex's end-of-text fallback then deletes everything after it, answer included."""
    if not text:
        return ""
    seg = text
    lt = text.rfind("<tool_response>")
    if lt != -1:
        la = text.rfind("\nassistant\n", lt)
        seg = text[la + len("\nassistant\n"):] if la != -1 else text[lt:]
    seg = seg.lstrip()
    if seg.startswith("<think>"):
        seg = seg[len("<think>"):]
    seg = _TOOL_SPAN_RE.sub("", seg)
    seg = seg.rsplit("</think>", 1)[-1] if "</think>" in seg else seg
    return seg.strip()


def graded_answer(row: dict) -> tuple:
    """(answer, source). The in-loop reward graded either the agent loop's final-turn
    answer or the plain think-stripped response, and the dump keeps the first 512
    chars of whichever it was as `extracted_answer`; take the candidate that
    reproduces that prefix. Dumps without the field (written by other scripts) get
    the plain strip, this script's original behaviour."""
    out = row.get("output", "")
    plain = strip_thinking(out)
    ex = row.get("extracted_answer")
    if not isinstance(ex, str):
        return plain, "no_field"
    for cand in (final_turn_answer(out), plain):
        if cand[:512] == ex[:512]:
            return cand, "verified"
    return plain, "unverified"


def conversation_text(conv, answer: str) -> str:
    lines = []
    for m in conv or []:
        if isinstance(m, dict) and m.get("content"):
            lines.append(f"{m.get('role', 'user')}: {m['content']}")
    lines.append(f"assistant: {answer}")
    return "\n\n".join(lines)


def parse_met(text: str):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", (text or "").strip())
    try:
        d = json.loads(cleaned)
        val = d.get("criteria_met")
        return val if isinstance(val, bool) else None
    except Exception:
        m = re.search(r'"criteria_met"\s*:\s*(true|false)', cleaned, re.I)
        return (m.group(1).lower() == "true") if m else None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def load_parquet(path):
    import pandas as pd
    d = pd.read_parquet(path)
    rows = []
    for _, r in d.iterrows():
        ei = r["extra_info"]
        conv = ei.get("conversation")
        if conv is None or (hasattr(conv, "__len__") and len(conv) == 0):
            conv = r["prompt"]
        conv = [dict(m) for m in conv]
        items = [{"criterion": str(it.get("criterion_text", it.get("criterion"))), "points": float(it["points"])}
                 for it in ei["rubric_items"]]
        rows.append({"conversation": conv, "items": items, "use_case": ei.get("use_case"),
                     "difficulty": ei.get("difficulty"), "specialty": ei.get("specialty"),
                     "question_id": ei.get("question_id")})
    return rows


def verify_join(dump_rows, val_rows) -> float:
    ok = 0
    for dr, vr in zip(dump_rows, val_rows):
        last_user = next((m["content"] for m in reversed(vr["conversation"]) if m.get("role") == "user"), "")
        probe = norm(last_user)[:80]
        if probe and probe in norm(dr.get("input", "")):
            ok += 1
    return ok / max(1, len(dump_rows))


async def call_grader(session, sem, args, prompt, attempt_limit=6):
    body = {"model": args.model,
            "messages": [{"role": "system", "content": GRADER_SYSTEM}, {"role": "user", "content": prompt}],
            "max_completion_tokens": args.max_completion_tokens}
    if args.effort and args.effort != "omit":
        body["reasoning_effort"] = args.effort
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    url = args.api_base.rstrip("/") + "/chat/completions"
    delay = 2.0
    for attempt in range(attempt_limit):
        try:
            async with sem:
                async with session.post(url, json=body, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=args.timeout)) as resp:
                    txt = await resp.text()
                    if resp.status == 200:
                        d = json.loads(txt)
                        return (d["choices"][0]["message"].get("content") or "")
                    if resp.status in (400, 401, 404):
                        raise RuntimeError(f"non-retriable {resp.status}: {txt[:200]}")
        except RuntimeError:
            raise
        except Exception as e:  # network / 429 / 5xx
            txt = f"{type(e).__name__}: {e}"
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)
    raise RuntimeError(f"grader failed after {attempt_limit} attempts: {txt[:200]}")


async def main_async(args):
    dump_rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    val_rows = load_parquet(args.val_parquet)
    if len(dump_rows) != len(val_rows):
        sys.exit(f"dump has {len(dump_rows)} rows, parquet {len(val_rows)}; refusing to join")
    join = verify_join(dump_rows, val_rows)
    print(f"join verified at {join:.3f}", flush=True)
    if join < args.join_min:
        sys.exit(f"join {join:.3f} below --join-min {args.join_min}")
    if args.limit:
        dump_rows, val_rows = dump_rows[:args.limit], val_rows[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    graded = [graded_answer(dr) for dr in dump_rows]
    answers = [a for a, _ in graded]
    sources = Counter(src for _, src in graded)
    print(f"answer source: {dict(sources)}", flush=True)
    hashes = [hashlib.sha1(a.encode()).hexdigest()[:12] for a in answers]
    # What earlier versions of this script graded (no hash in their records): their
    # verdicts stay valid exactly where that text equals today's answer.
    legacy = [strip_thinking(dr.get("output", "")) for dr in dump_rows]

    vpath = args.out + ".verdicts.jsonl"
    cache = {}
    invalid = 0
    if os.path.exists(vpath):
        for l in open(vpath):
            if l.strip():
                v = json.loads(l)
                i = v["row"]
                if i >= len(answers):
                    continue
                h = v.get("h")
                ok = (legacy[i] == answers[i]) if h is None else (h == hashes[i])
                if ok:
                    cache[(i, v["item"], v["vote"])] = v
                else:
                    invalid += 1
    print(f"cached verdicts: {len(cache)} (invalidated {invalid}: graded text changed)", flush=True)
    vfile = open(vpath, "a")
    lock = asyncio.Lock()
    convs = [conversation_text(vr["conversation"], ans) for vr, ans in zip(val_rows, answers)]
    template = GRADER_TEMPLATE + (STRICT_NOTE if args.strict else "")

    sem = asyncio.Semaphore(args.concurrency)
    todo = [(i, j, v) for i, vr in enumerate(val_rows) for j in range(len(vr["items"]))
            for v in range(args.votes) if (i, j, v) not in cache]
    print(f"grader calls to make: {len(todo)}", flush=True)
    done = [0]
    t0 = time.time()

    async def one(session, i, j, v):
        it = val_rows[i]["items"][j]
        prompt = (template.replace("<<conversation>>", convs[i])
                  .replace("<<rubric_item>>", f"[{it['points']:g}] {it['criterion']}"))
        try:
            raw = await call_grader(session, sem, args, prompt)
            met = parse_met(strip_thinking(raw))
            err = None if met is not None else "unparseable"
        except Exception as e:
            raw, met, err = "", None, str(e)[:300]
        rec = {"row": i, "item": j, "vote": v, "met": met, "err": err, "h": hashes[i]}
        async with lock:
            cache[(i, j, v)] = rec
            vfile.write(json.dumps(rec) + "\n"); vfile.flush()
            done[0] += 1
            if done[0] % 200 == 0:
                rate = done[0] / (time.time() - t0)
                print(f"  {done[0]}/{len(todo)} calls, {rate:.1f}/s", flush=True)

    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*(one(session, i, j, v) for i, j, v in todo))
    vfile.close()

    # Aggregate.
    per_row = []
    ungradable = 0
    for i, vr in enumerate(val_rows):
        met_pts = 0.0
        pos = sum(it["points"] for it in vr["items"] if it["points"] > 0)
        met_flags = []
        for j, it in enumerate(vr["items"]):
            votes = [cache[(i, j, v)]["met"] for v in range(args.votes)]
            valid = [x for x in votes if x is not None]
            if not valid:
                ungradable += 1
                met = False
            else:
                met = sum(valid) * 2 > len(valid)  # majority; tie -> not met
            met_flags.append(met)
            if met:
                met_pts += it["points"]
        raw = met_pts / pos if pos > 0 else 0.0
        chars = len(answers[i])
        adj = raw - LENGTH_PENALTY_PER_500 * (chars - LENGTH_CENTER) / 500.0
        per_row.append({"row": i, "question_id": vr["question_id"], "acc_raw_signed": raw,
                        "acc_raw": min(1.0, max(0.0, raw)), "acc_len_adj_signed": adj,
                        "acc_len_adj": min(1.0, max(0.0, adj)), "chars": chars,
                        "use_case": vr["use_case"], "difficulty": vr["difficulty"],
                        "specialty": vr["specialty"], "met": met_flags,
                        "inloop_acc_len_adj_signed": dump_rows[i].get("acc_len_adj_signed")})

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    metrics = ["acc_len_adj_signed", "acc_raw", "acc_len_adj", "acc_raw_signed"]
    out = {"dump": args.dump, "grader": args.model, "effort": args.effort, "votes": args.votes,
           "strict": bool(args.strict), "n": len(per_row), "join_verified": join,
           "ungradable_items": ungradable,
           "answer_source": dict(sources), "verdicts_invalidated": invalid,
           "overall": {m: mean([r[m] for r in per_row]) for m in metrics},
           "inloop_overall_acc_len_adj_signed": mean([r["inloop_acc_len_adj_signed"] for r in per_row
                                                      if r["inloop_acc_len_adj_signed"] is not None])}
    for dim in ("use_case", "difficulty", "specialty"):
        groups = defaultdict(list)
        for r in per_row:
            groups[r[dim]].append(r)
        out[dim] = {k: {"n": len(v), **{m: mean([r[m] for r in v]) for m in metrics}} for k, v in groups.items()}
    out["per_row"] = per_row
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("overall", "inloop_overall_acc_len_adj_signed", "ungradable_items", "answer_source")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--val-parquet", default="/scratch/sheng/self_evolving/healthbench_pro_val.parquet")
    ap.add_argument("--api-base", default=os.environ.get("API_BASE", "http://point.dd.works:18890/v1"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--model", default="gpt-5.4_2026-03-05")
    ap.add_argument("--effort", default="low", help="reasoning_effort; 'omit' sends none (gpt-chat-latest)")
    ap.add_argument("--max-completion-tokens", type=int, default=512)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--strict", action="store_true", help="append the strict grading note")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--join-min", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("--api-key (or API_KEY) required")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
