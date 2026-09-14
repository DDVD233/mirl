#!/usr/bin/env python3
"""Frontier (API) models on the stage-2 benchmarks with OUR tools, not theirs.

dvd 2026-09-14: compare GPT-5.6 deployments against our trained solvers on
PRBench, ProfBench and MedXpertQA at several reasoning efforts -- with native tool
use disabled and the model given exactly the tools our solver has: `web_search`
(Serper through kb/serper_cache_server.py, verbatim SERP blocks) and, on the medical
benchmarks, `search_medical_kb` (the gen server's /retrieve evidence brief). The
system instruction, tool schemas, search budget, budget-exhausted error and forced
final-answer instruction are the solver's own, imported from the agent loop and the
tool YAMLs, so the information access matches; only the call format differs
(OpenAI function calling instead of the XML tool markup).

Writes a validation-dump-shaped jsonl (row i = parquet row i, `output` = the final
answer) that scripts/self_evolving/analysis/regrade_hbpro_dumps.py grades with the
same rubric grader as every other row.

Run on a pod (needs the proxy, a serper cache server and, for KB tools, a gen server):
  SE_DOMAIN=prbench python3 scripts/self_evolving/eval/frontier_tool_eval.py \
      --parquet $S/prbench_hard_val.parquet --tools web --max-searches 4 \
      --model gpt-5.6-luna_2026-07-09 --effort low --search-url http://localhost:8057/search \
      --out $S/paper_refresh/frontier/prbench_luna_low.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import random
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "self_evolving"))

CFG = REPO / "scripts" / "self_evolving" / "train" / "config"
TOOL_YAML = {"web": CFG / "web_search_tool_only.yaml", "kb+web": CFG / "medical_retrieval_web_tool.yaml",
             "kb": CFG / "medical_retrieval_tool.yaml"}


def load_instructions(tools: str, max_searches: int) -> tuple[str, str, str]:
    """(system instruction, budget-exhausted error, forced-answer instruction), rebranded
    for SE_DOMAIN exactly as the agent loop does at import."""
    import verl.experimental.agent_loop.retrieval_tool_agent_loop as al  # noqa: E402
    if tools == "web":
        return (al.WEB_ONLY_INSTRUCTION.format(max_searches=max_searches),
                al.WEB_ONLY_BUDGET_EXHAUSTED_ERROR.format(n=max_searches),
                al.WEB_ONLY_HARD_ANSWER_INSTRUCTION)
    text = al.RETRIEVE_INSTRUCTION.format(max_searches=max_searches)
    if tools == "kb+web":
        text += al.WEB_INSTRUCTION.format(max_searches=max_searches)
    return text, al.BUDGET_EXHAUSTED_ERROR.format(n=max_searches), al.HARD_ANSWER_INSTRUCTION


def load_tool_schemas(tools: str) -> list[dict]:
    spec = yaml.safe_load(open(TOOL_YAML[tools]))
    out = []
    for t in spec["tools"]:
        fn = json.loads(json.dumps(t["tool_schema"]["function"]))
        props = fn.get("parameters", {}).get("properties", {})
        for p in props.values():           # OpenAI needs `items` on array params
            if p.get("type") == "array" and "items" not in p:
                p["items"] = {"type": "string"}
        out.append({"type": "function", "function": fn})
    return out


def _data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(open(path, 'rb').read()).decode()}"


def build_conversation(row, images_root: str) -> list[dict]:
    msgs = []
    images = list(row["images"]) if "images" in row and row["images"] is not None else []
    images = [i if isinstance(i, str) else i.get("image") for i in images]
    if images_root:
        images = [i if os.path.isabs(i) else os.path.join(images_root, i) for i in images]
    it = iter(images)
    for m in list(row["prompt"]):
        role, content = m["role"], m["content"]
        if role == "user" and isinstance(content, str) and "<image>" in content and images:
            parts = []
            for k, seg in enumerate(content.split("<image>")):
                if k:
                    try:
                        parts.append({"type": "image_url", "image_url": {"url": _data_uri(next(it))}})
                    except StopIteration:
                        pass
                if seg.strip():
                    parts.append({"type": "text", "text": seg})
            msgs.append({"role": role, "content": parts})
        else:
            msgs.append({"role": role, "content": content})
    return msgs


def last_user_text(msgs: list[dict]) -> str:
    for m in reversed(msgs):
        if m["role"] != "user":
            continue
        c = m["content"]
        if isinstance(c, str):
            return c
        return " ".join(p.get("text", "") for p in c if p.get("type") == "text")
    return ""


class Runner:
    def __init__(self, a, instr, tool_schemas):
        self.a, self.instr, self.tool_schemas = a, instr, tool_schemas
        self.sem = asyncio.Semaphore(a.concurrency)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=30))
        self.headers = {"Authorization": f"Bearer {a.api_key}", "Content-Type": "application/json"}

    async def chat(self, messages, tools, tool_choice=None):
        body = {"model": self.a.model, "messages": messages, "max_completion_tokens": self.a.max_tokens}
        if self.a.effort != "omit":
            body["reasoning_effort"] = self.a.effort
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        delay = 3.0
        for attempt in range(self.a.retries):
            try:
                r = await self.client.post(f"{self.a.api_base}/chat/completions", headers=self.headers, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                if attempt == self.a.retries - 1:
                    raise
                await asyncio.sleep(delay + random.random()); delay = min(delay * 2, 60)

    async def tool(self, name: str, args: dict, question: str) -> tuple[str, dict]:
        try:
            if name == "web_search":
                q = str(args.get("query") or "").strip()
                if not q:
                    return "No search query provided. Provide one specific English query.", {"error": 1}
                r = await self.client.post(self.a.search_url, json={"query": q}, timeout=240)
                r.raise_for_status(); d = r.json()
                return d.get("text") or f'No web results for "{q}".', {"hits": d.get("hits"), "cached": d.get("cached")}
            if name == "search_medical_kb":
                qs = args.get("queries") or args.get("query") or []
                if isinstance(qs, str):
                    try:
                        qs = json.loads(qs) if qs.strip().startswith("[") else [qs]
                    except Exception:  # noqa: BLE001
                        qs = [qs]
                qs = [str(q).strip() for q in qs if str(q).strip()][:4]
                if not qs:
                    return "No search query provided. Provide 1-4 English sub-queries.", {"error": 1}
                r = await self.client.post(self.a.retrieve_url, json={"queries": qs, "question": question[:4000]}, timeout=240)
                r.raise_for_status(); d = r.json()
                return d.get("text") or "No relevant passages found in the medical knowledge base.", {"passages": len(d.get("passages") or []), "summarized": d.get("summarized")}
            return f"Unknown tool {name}.", {"error": 1}
        except Exception as e:  # noqa: BLE001
            unavailable = ("Web search is temporarily unavailable; answer from your own knowledge."
                           if name == "web_search" else
                           "Retrieval is temporarily unavailable; answer from your own knowledge.")
            return unavailable, {"error": 1, "exc": type(e).__name__}

    async def episode(self, idx: int, row) -> dict:
        system, exhausted, hard = self.instr
        conv = build_conversation(row, self.a.images_root)
        messages = [{"role": "system", "content": system}] + conv
        question = last_user_text(conv)
        n_search = n_web = 0; calls = []; usage = {"prompt": 0, "completion": 0, "reasoning": 0}
        output, finish = "", "none"
        async with self.sem:
            for _turn in range(self.a.max_searches + 3):
                open_ = n_search < self.a.max_searches
                resp = await self.chat(messages, self.tool_schemas if open_ else None,
                                       tool_choice=None if open_ else "none")
                ch = resp["choices"][0]; msg = ch["message"]
                u = resp.get("usage") or {}
                usage["prompt"] += u.get("prompt_tokens") or 0; usage["completion"] += u.get("completion_tokens") or 0
                usage["reasoning"] += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                tcs = msg.get("tool_calls") or []
                if tcs and open_:
                    messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
                    for tc in tcs:
                        fn = tc["function"]; name = fn["name"]
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except Exception:  # noqa: BLE001
                            args = {"query": fn.get("arguments")}
                        if n_search >= self.a.max_searches:
                            text, meta = exhausted, {"over_budget": 1}
                        else:
                            text, meta = await self.tool(name, args, question)
                            n_search += 1; n_web += int(name == "web_search")
                        calls.append({"name": name, "args": args, "meta": meta, "chars": len(text)})
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
                    continue
                if tcs:  # over budget: the solver's refusal, then a forced answer turn
                    messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
                    for tc in tcs:
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": exhausted})
                    messages.append({"role": "user", "content": hard})
                    continue
                output, finish = (msg.get("content") or ""), ch.get("finish_reason")
                break
        return {"index": idx, "question_id": (row["extra_info"] or {}).get("question_id"),
                "output": output, "n_search": n_search, "n_web": n_web, "tool_calls": calls,
                "usage": usage, "finish_reason": finish, "model": self.a.model, "effort": self.a.effort}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--tools", choices=["web", "kb+web", "kb", "none"], default="web")
    ap.add_argument("--max-searches", type=int, default=2)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="low", help="reasoning_effort, or 'omit'")
    ap.add_argument("--api-base", default=os.environ.get("API_BASE", "http://point.dd.works:18890/v1"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--search-url", default="http://localhost:8057/search")
    ap.add_argument("--retrieve-url", default="http://localhost:8041/retrieve")
    ap.add_argument("--images-root", default="")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.api_key:
        a.api_key = open("/scratch/sheng/self_evolving/.trapi_key").read().strip()

    df = pd.read_parquet(a.parquet)
    if a.limit:
        df = df.iloc[: a.limit]
    if a.tools == "none":
        instr, schemas = ("Answer the request directly and completely.", "", ""), []
    else:
        instr, schemas = load_instructions(a.tools, a.max_searches), load_tool_schemas(a.tools)
    partial = a.out + ".partial.jsonl"
    done = {}
    if os.path.exists(partial):
        for line in open(partial):
            try:
                r = json.loads(line); done[r["index"]] = r
            except Exception:  # noqa: BLE001
                pass
    runner = Runner(a, instr, schemas)
    t0 = time.time(); todo = [i for i in range(len(df)) if i not in done]
    print(f"{a.model} effort={a.effort} tools={a.tools} rows={len(df)} todo={len(todo)}", flush=True)
    lock = asyncio.Lock()

    async def one(i):
        try:
            r = await runner.episode(i, df.iloc[i])
        except Exception as e:  # noqa: BLE001
            r = {"index": i, "output": "", "n_search": 0, "n_web": 0, "tool_calls": [], "error": f"{type(e).__name__}: {str(e)[:200]}",
                 "model": a.model, "effort": a.effort}
        async with lock:
            done[i] = r
            with open(partial, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n = len(done)
            if n % 25 == 0 or n == len(df):
                print(f"  {n}/{len(df)} done, {time.time()-t0:.0f}s", flush=True)

    await asyncio.gather(*(one(i) for i in todo))
    with open(a.out, "w") as f:
        for i in range(len(df)):
            f.write(json.dumps(done[i], ensure_ascii=False) + "\n")
    errs = sum(1 for r in done.values() if r.get("error"))
    ns = sum(r.get("n_search", 0) for r in done.values()) / max(1, len(done))
    print(f"wrote {a.out}: {len(done)} rows, errors={errs}, mean searches={ns:.2f}", flush=True)
    await runner.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
