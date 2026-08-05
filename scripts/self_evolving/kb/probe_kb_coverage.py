"""Per-criterion knowledge-base coverage probe for the HealthBench-Pro val set.

Answers, for every rubric criterion in `healthbench_pro_val.parquet`, the question
"could retrieval have supplied the fact this criterion rewards?" and, when the
answer is no, WHY not. Categories (exclusive, mirroring the failure modes that
actually need different fixes):

    RETRIEVABLE    a passage the *natural* rollout query returns already states
                   the fact -> retrieval works today; the model simply is not
                   using it.
    LATENT         absent from the natural retrieval but present when queried
                   with the criterion itself (or found only once the source
                   filter / re-ranking is removed) -> the fact IS in the KB and
                   the retrieval policy is what loses it. `latent_cause`
                   separates query-formulation from filter/ranking.
    ABSENT         in neither -> the KB does not contain it. `suggested_source`
                   names what would have to be ingested.
    CONFLICT       a returned passage contradicts the graded criterion ->
                   retrieval would actively push the model to a wrong answer.
    NOT_KNOWLEDGE  the criterion rewards behaviour, not a fact (asks a follow-up
                   question, empathetic tone, structure/format). Retrieval
                   cannot move it either way, so it must be excluded from every
                   coverage rate or the headline numbers are meaningless.

Three retrieval variants per criterion make the categories separable:

    A  natural   LLM-proposed search query built from the task ALONE (no rubric)
                 through the shipped policy (kb/retrieval.py) - what a rollout
                 realistically sees.
    B  oracle    the criterion text itself as the query, same policy - isolates
                 query formulation: if B finds it and A does not, the retrieval
                 got the right corpus but the wrong question.
    C  raw       criterion text, dense cosine over ALL rows, no EXCLUDE_EXPR and
                 no source re-ranking - isolates the policy: a hit only here
                 means our own filter/cap threw the fact away.

Usage:
    python scripts/self_evolving/kb/probe_kb_coverage.py \
        --val   scripts/self_evolving/kb/probe_out/healthbench_pro_val.parquet \
        --out   scripts/self_evolving/kb/probe_out/kb_coverage.jsonl \
        --miss  scripts/self_evolving/kb/probe_out/val_criterion_miss.json \
        --concurrency 12

Resumable: re-running skips question_ids already present in --out.
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kb.retrieval import RetrieveConfig, rank_hits  # noqa: E402

EMBED_BASE = os.environ.get("EMBED_API_BASE", "http://localhost:18001/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19531")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "root:Milvus")
COLLECTION = os.environ.get("MILVUS_COLLECTION", "medical_knowledge_v2")
JUDGE_BASE = os.environ.get("JUDGE_API_BASE", "http://point.dd.works:18890/v1")
JUDGE_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-chat-latest_2026-05-28")

OUTPUT_FIELDS = ["source_dataset", "content_type", "text_content",
                 "question", "answer", "entry_id"]

QUERY_PROPOSER_PROMPT = """\
You are the retrieval front-end of a clinical assistant. You are given a clinician's \
request. Write the ONE search query you would send to a medical knowledge base to \
retrieve the reference material needed to answer it well.

Rules:
- Write it in ENGLISH even if the request is in another language (the knowledge base \
is English).
- A specific, information-dense sentence or noun phrase - not keywords, not a \
restatement of the whole case.
- Name the clinical entities (condition, drug, procedure, guideline) explicitly.
- Output ONLY the query, nothing else."""

JUDGE_SYSTEM = """\
You are auditing whether a medical knowledge base contains the information needed to \
satisfy a grading rubric. You are precise, sceptical, and you never credit a passage \
that is merely on-topic: it must actually STATE the fact the criterion rewards.

Return ONLY valid JSON, no prose, no markdown fence."""

JUDGE_TEMPLATE = """\
# Clinician request
{task}

# Retrieval variant A - what a real rollout sees
Query the assistant proposed from the request alone: "{query_a}"
{passages_a}

# Per-criterion oracle retrieval
For each criterion below you are also shown variant B (the criterion text used as the \
query, through the same ranking policy) and variant C (the same query, raw dense search \
over the WHOLE knowledge base with no source filter and no re-ranking).

{criteria_blocks}

# Your task
For EACH criterion, decide whether the knowledge base can supply what it rewards.

Categories (choose exactly one):
- "NOT_KNOWLEDGE": the criterion rewards a behaviour, not a fact - asking a \
clarifying/follow-up question, tone/empathy, safety-netting, hedging, formatting, \
structure, length, language choice, "does not fabricate", "avoids overconfidence". \
Judge this FIRST, but apply it STRICTLY: many criteria are compound ("does at least \
one of the following: ...", "does all of the following: ..."). If ANY branch names a \
specific clinical fact - a drug, dose, timeframe ("within 72 hours"), threshold, \
score cutoff, code, red-flag list, or named guideline - the criterion IS \
knowledge-dependent and must be categorised as one of the four below. Use \
NOT_KNOWLEDGE only when NO branch requires any external fact.
- "CONFLICT": some shown passage directly CONTRADICTS what the criterion rewards \
(different dose, opposite recommendation, superseded guideline). Reserve this for a \
real contradiction, not for silence or a different emphasis.
- "RETRIEVABLE": a variant-A passage states the fact (or enough of it that a competent \
model reading only A would satisfy the criterion).
- "LATENT": not in A, but a B or C passage states it.
- "ABSENT": no shown passage states it.

For LATENT, set "latent_cause":
- "query": found in B (and possibly C) - the ranking policy is fine, the natural query \
just did not ask for it.
- "policy": found ONLY in C - our own source filter / re-ranking / per-source cap \
discarded it.

For ABSENT, fill "missing_fact" (one sentence naming the specific fact needed) and \
"suggested_source" (concretely what to ingest: e.g. "StatPearls: Adenoidectomy", \
"FDA label - apixaban", "ACC/AHA 2025 dyslipidemia guideline", "ICD-10-CM tabular \
E11.42", "WHO malaria treatment guidelines 4th ed"). Prefer a named, freely \
downloadable clinical reference.

For CONFLICT, fill "conflict_detail" (what the passage says vs what the criterion \
requires) and "conflict_passage".

Output JSON:
{{"criteria": [{{"index": 0, "category": "...", "evidence_passage": "A2"|"B1"|"C3"|null,
  "latent_cause": "query"|"policy"|null, "missing_fact": null|"...",
  "suggested_source": null|"...", "conflict_detail": null|"...",
  "conflict_passage": null|"...", "confidence": 0.0-1.0}}]}}"""


# ---------------------------------------------------------------- retrieval --
def _fmt(passages: list[dict], tag: str, cap: int = 700) -> str:
    if not passages:
        return f"(no {tag} passages)"
    return "\n\n".join(
        f"[{tag}{i + 1} | source={p['source']}]\n{p['text'][:cap]}"
        for i, p in enumerate(passages)
    )


async def embed(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    r = await client.post(f"{EMBED_BASE}/embeddings",
                          json={"model": EMBED_MODEL,
                                "input": [t[:2000] for t in texts]},
                          headers={"Authorization": "Bearer EMPTY"}, timeout=120)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def _hits(res) -> list[dict]:
    out = []
    for hit_list in res:
        for hit in hit_list:
            e = hit["entity"]
            out.append({
                "source": e.get("source_dataset", ""),
                "content_type": e.get("content_type", ""),
                "text": (e.get("text_content") or e.get("answer")
                         or e.get("question") or ""),
                "entry_id": e.get("entry_id", ""),
                "score": hit["distance"],
            })
    return out


def search_sync(mc, vec, limit, filter_expr):
    return _hits(mc.search(collection_name=COLLECTION, data=[vec], limit=limit,
                           filter=filter_expr, output_fields=OUTPUT_FIELDS))


async def search(mc, vec, limit, filter_expr=""):
    return await asyncio.to_thread(search_sync, mc, vec, limit, filter_expr)


# ------------------------------------------------------------------- judge --
async def judge_call(client: httpx.AsyncClient, system: str, user: str,
                     max_tokens: int = 6000, retries: int = 4) -> str:
    payload = {"model": JUDGE_MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "max_completion_tokens": max_tokens}
    delay = 3.0
    for attempt in range(retries):
        try:
            r = await client.post(f"{JUDGE_BASE}/chat/completions", json=payload,
                                  headers={"Authorization": f"Bearer {JUDGE_KEY}"},
                                  timeout=300)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or ""
            if content.strip():
                return content
            raise RuntimeError("empty completion (reasoning ate the budget)")
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  judge retry {attempt + 1}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            await asyncio.sleep(delay)
            delay *= 2
    return ""


def parse_json(text: str) -> dict | None:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        t = t.lstrip("json").strip()
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                return None
    return None


# -------------------------------------------------------------- per-task ----
async def probe_task(task, mc, http, judge, cfg, top_k, sem):
    async with sem:
        qid = task["question_id"]
        text = task["task"]
        crits = task["criteria"]

        # A: the query a rollout would actually issue.
        query_a = (await judge_call(http, QUERY_PROPOSER_PROMPT, text,
                                    max_tokens=2000)).strip().strip('"')[:400]
        vecs = await embed(http, [query_a] + [c["criterion_text"] for c in crits])
        va, vcs = vecs[0], vecs[1:]

        hits_a = await search(mc, va, top_k * cfg.fetch_mult, cfg.exclude_expr)
        pa = rank_hits(hits_a, top_k, cfg)

        blocks, per_crit = [], []
        for i, (c, vc) in enumerate(zip(crits, vcs)):
            hb = await search(mc, vc, top_k * cfg.fetch_mult, cfg.exclude_expr)
            pb = rank_hits(hb, top_k, cfg)
            # C: raw dense, no filter, no re-ranking, no source cap.
            hc = await search(mc, vc, top_k * 2, "")
            pc = [{"source": h["source"], "text": h["text"][:1600],
                   "score": h["score"]} for h in hc][:top_k]
            per_crit.append((pb, pc))
            blocks.append(
                f"## Criterion {i} (points {c['points']})\n{c['criterion_text']}\n\n"
                f"{_fmt(pb, 'B')}\n\n{_fmt(pc, 'C')}"
            )

        prompt = JUDGE_TEMPLATE.format(
            task=text[:4000], query_a=query_a, passages_a=_fmt(pa, "A"),
            criteria_blocks="\n\n".join(blocks))
        raw = await judge_call(http, JUDGE_SYSTEM, prompt)
        parsed = parse_json(raw)
        if not parsed or "criteria" not in parsed:
            return {"question_id": qid, "error": "unparsed_judge",
                    "raw": raw[:1000], "query_a": query_a}

        by_idx = {int(r.get("index", -1)): r for r in parsed["criteria"]}
        rows = []
        for i, c in enumerate(crits):
            v = by_idx.get(i, {})
            pb, pc = per_crit[i]
            rows.append({
                "criterion_text": c["criterion_text"],
                "points": c["points"],
                "met_rate": c.get("met_rate"),
                "n_val_obs": c.get("n_val_obs"),
                "category": v.get("category", "UNPARSED"),
                "latent_cause": v.get("latent_cause"),
                "evidence_passage": v.get("evidence_passage"),
                "missing_fact": v.get("missing_fact"),
                "suggested_source": v.get("suggested_source"),
                "conflict_detail": v.get("conflict_detail"),
                "conflict_passage": v.get("conflict_passage"),
                "confidence": v.get("confidence"),
                "b_sources": [p["source"] for p in pb],
                "c_sources": [p["source"] for p in pc],
            })
        return {"question_id": qid, "data_source": task["data_source"],
                "task": text[:1200], "query_a": query_a,
                "a_sources": [p["source"] for p in pa],
                "a_top_score": pa[0]["score"] if pa else None,
                "criteria": rows}


# ------------------------------------------------------------------- main ---
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--miss", default="", help="val_criterion_miss.json (optional)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    global JUDGE_KEY
    if not JUDGE_KEY:
        sys.exit("set JUDGE_API_KEY")

    import pandas as pd
    from pymilvus import MilvusClient

    df = pd.read_parquet(args.val)
    miss = {}
    if args.miss and os.path.exists(args.miss):
        miss = json.load(open(args.miss)).get("data", {})

    tasks = []
    for _, row in df.iterrows():
        xi = row["extra_info"]
        crits = []
        for c in xi["rubric_items"]:
            m = miss.get(c["criterion_text"])
            crits.append({
                "criterion_text": c["criterion_text"],
                "points": float(c["points"]),
                "met_rate": (m["met"] / m["n"]) if m and m["n"] else None,
                "n_val_obs": m["n"] if m else 0,
            })
        tasks.append({"question_id": xi["question_id"],
                      "data_source": row["data_source"],
                      "task": row["prompt"][0]["content"], "criteria": crits})
    if args.limit:
        tasks = tasks[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                pass
    todo = [t for t in tasks if t["question_id"] not in done]
    print(f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to probe", flush=True)

    cfg = RetrieveConfig()
    mc = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    sem = asyncio.Semaphore(args.concurrency)
    counts, n_done = Counter(), 0

    async with httpx.AsyncClient() as http:
        fh = open(args.out, "a")
        coros = [probe_task(t, mc, http, JUDGE_MODEL, cfg, args.top_k, sem)
                 for t in todo]
        for fut in asyncio.as_completed(coros):
            try:
                res = await fut
            except Exception as e:
                print(f"  task failed: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            for c in res.get("criteria", []):
                counts[c["category"]] += 1
            if n_done % 20 == 0:
                print(f"[{n_done}/{len(todo)}] {dict(counts)}", flush=True)
        fh.close()
    print(f"DONE {n_done} tasks; categories: {dict(counts)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
