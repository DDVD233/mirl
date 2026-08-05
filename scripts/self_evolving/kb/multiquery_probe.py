"""Does a TASK-ONLY multi-query decomposition recover the LATENT facts?

The coverage probe established LATENT = "variant A (one natural query) missed it,
but variant B (the criterion text itself, an ORACLE the rollout never has) finds it".
That does not tell us whether a rollout could recover the fact, because the rollout
never sees the rubric. This script measures the thing that matters:

    given the TASK ALONE, ask the front-end model for N diverse sub-queries,
    run each through the SHIPPED policy, merge + dedup, and judge (same strict
    rule as the coverage probe) whether the merged pool STATES each LATENT fact.

Arms are configurable so query count / k / per-source cap can be varied
independently and the recovery curve read off directly.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
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
LLM_BASE = os.environ.get("JUDGE_API_BASE", "http://point.dd.works:18890/v1")
LLM_KEY = os.environ.get("JUDGE_API_KEY", "sk-xMByFeWLKB87wZ")
LLM_MODEL = os.environ.get("JUDGE_MODEL", "gpt-chat-latest_2026-05-28")

OUTPUT_FIELDS = ["source_dataset", "content_type", "text_content",
                 "question", "answer", "entry_id"]

PLAN_PROMPT = """\
You are the retrieval front-end of a clinical assistant. You are given a clinician's \
request. Before answering, you may look things up in an English medical knowledge base \
(clinical reviews, FDA labels, consumer-health topics, textbooks, PubMed abstracts).

Write EXACTLY {n} search queries that TOGETHER cover everything a thorough expert answer \
to this request would need to get right.

Rules:
- ENGLISH only, even if the request is in another language.
- The queries must be DIFFERENT FACETS of the request, not rewordings of each other. \
Typical facets: (a) the core diagnosis/management question, (b) exact numbers - dose, \
threshold, timing, staging cutoff, code, (c) safety - contraindications, adverse effects, \
monitoring, interactions, special populations (pregnancy, pediatrics, renal/hepatic), \
(d) the alternative or comparison the request implies, (e) patient-facing explanation or \
follow-up/red-flag advice.
- Each query is a specific, information-dense sentence or noun phrase naming the clinical \
entities explicitly. Expand every abbreviation and acronym. If the request states a \
premise that may be wrong, ALSO query the correct entity, not only the user's wording.
- Output ONLY a JSON array of {n} strings. No prose, no markdown fence."""

JUDGE_SYSTEM = """\
You are auditing whether a set of retrieved passages supplies what a grading rubric \
criterion rewards. You are precise and sceptical: a passage that is merely on-topic does \
NOT count; it must actually STATE the fact the criterion rewards (or enough of it that a \
competent physician reading only these passages would satisfy the criterion).

Return ONLY valid JSON, no prose, no markdown fence."""

JUDGE_TEMPLATE = """\
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


# ---------------------------------------------------------------- infra
async def embed(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    r = await client.post(f"{EMBED_BASE.rstrip('/')}/embeddings",
                          json={"model": EMBED_MODEL, "input": [t[:2000] for t in texts]},
                          headers={"Authorization": "Bearer EMPTY"}, timeout=120)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def milvus_search(mc, vecs: list[list[float]], limit: int, expr: str | None):
    kw = dict(collection_name=COLLECTION, data=vecs, limit=limit, output_fields=OUTPUT_FIELDS)
    if expr:
        kw["filter"] = expr
    res = mc.search(**kw)
    out = []
    for hl in res:
        rows = []
        for h in hl:
            e = h["entity"]
            rows.append({"source": e.get("source_dataset", ""),
                         "content_type": e.get("content_type", ""),
                         "text": e.get("text_content", "") or e.get("answer", "") or e.get("question", ""),
                         "entry_id": e.get("entry_id", ""),
                         "score": h["distance"]})
        out.append(rows)
    return out


PROPOSER_BASE = os.environ.get("PROPOSER_API_BASE", LLM_BASE)
PROPOSER_MODEL = os.environ.get("PROPOSER_MODEL", LLM_MODEL)


async def llm(client: httpx.AsyncClient, system: str, user: str, max_tok: int = 1200,
              proposer: bool = False) -> str:
    base = PROPOSER_BASE if proposer else LLM_BASE
    model = PROPOSER_MODEL if proposer else LLM_MODEL
    tokkey = "max_tokens" if proposer and base != LLM_BASE else "max_completion_tokens"
    for attempt in range(5):
        try:
            r = await client.post(f"{base.rstrip('/')}/chat/completions",
                                  headers={"Authorization": f"Bearer {LLM_KEY}"},
                                  json={"model": model,
                                        "messages": [{"role": "system", "content": system},
                                                     {"role": "user", "content": user}],
                                        tokkey: max_tok},
                                  timeout=180)
            if r.status_code == 429:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 4:
                raise
            await asyncio.sleep(4 * (attempt + 1))
    return ""


def parse_json(txt: str):
    t = (txt or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# ---------------------------------------------------------------- merge
_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return _WS.sub(" ", (t or "").lower()).strip()


def shingles(t: str, n: int = 8) -> set:
    w = norm(t).split()
    if len(w) < n:
        return {" ".join(w)}
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


STUB_RE = re.compile(r"^[^.!?]{0,4000}$", re.DOTALL)


def is_stub(text: str) -> bool:
    """Heading/TOC row: many short colon-terminated lines, no sentences."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return True
    colonish = sum(1 for ln in lines if ln.endswith(":") or len(ln.split()) <= 6)
    body = " ".join(lines)
    n_sent = len(re.findall(r"[a-z]{3,}[.!?](?:\s|$)", body))
    return (colonish >= max(2, 0.7 * len(lines))) and n_sent <= 1


def merge(per_query: list[list[dict]], total: int, per_source_cap: int,
          dedup_jaccard: float = 0.6, drop_stubs: bool = True) -> tuple[list[dict], dict]:
    """Round-robin across sub-queries, global near-dup removal + global source cap."""
    out, seen_sh, seen_id, per_src = [], [], set(), Counter()
    stats = Counter()
    depth = max((len(x) for x in per_query), default=0)
    for d in range(depth):
        for q in per_query:
            if d >= len(q) or len(out) >= total:
                continue
            p = q[d]
            txt = p["text"]
            if p.get("entry_id") and p["entry_id"] in seen_id:
                stats["dup_exact"] += 1
                continue
            if drop_stubs and is_stub(txt):
                stats["stub"] += 1
                continue
            sh = shingles(txt)
            dup = False
            for prev in seen_sh:
                inter = len(sh & prev)
                if inter / max(1, min(len(sh), len(prev))) >= dedup_jaccard:
                    dup = True
                    break
            if dup:
                stats["dup_near"] += 1
                continue
            if per_src[p["source"]] >= per_source_cap:
                stats["src_cap"] += 1
                continue
            per_src[p["source"]] += 1
            seen_sh.append(sh)
            if p.get("entry_id"):
                seen_id.add(p["entry_id"])
            out.append(p)
    stats["kept"] = len(out)
    return out, dict(stats)


def fmt(passages: list[dict]) -> str:
    return "\n\n".join(f"[passage {i+1} | source={p['source']}]\n{p['text']}"
                       for i, p in enumerate(passages))


# ---------------------------------------------------------------- arms
async def run_task(client, mc, sem, task, records, arms, n_plan):
    async with sem:
        raw = await llm(client, "You output only JSON.",
                        PLAN_PROMPT.format(n=n_plan) + "\n\n# Request\n" + task, 900, proposer=True)
    qs = parse_json(raw)
    if not isinstance(qs, list):
        return None
    qs = [str(q) for q in qs][:n_plan]
    if not qs:
        return None
    # index 0 of `all_q` is the PROBE's own variant-A query (true control),
    # indices 1.. are this run's task-only decomposition.
    all_q = [records[0].get("query_a") or task[:500]] + qs

    async with sem:
        vecs_all = await embed(client, all_q)
    vecs = vecs_all[1:]

    results = {"queries": qs, "arms": {}}
    for name, cfg in arms.items():
        k = cfg["k"]
        rc = RetrieveConfig()
        rc.max_per_source = cfg.get("policy_per_source", rc.max_per_source)
        rc.fetch_mult = cfg.get("fetch_mult", rc.fetch_mult)
        if cfg.get("no_exclude"):
            rc.exclude_expr = None
        expr = rc.exclude_expr
        if cfg.get("use_query_a"):
            use_vecs = vecs_all[:1]
        else:
            use_vecs = vecs[: min(cfg["n_queries"], len(qs))]
        hitsets = await asyncio.to_thread(milvus_search, mc, use_vecs, k * rc.fetch_mult, expr)
        ranked = [rank_hits(h, k, rc) for h in hitsets]
        # rank_hits drops entry_id; re-attach by text match for dedup
        for rl, hl in zip(ranked, hitsets):
            bytext = {norm(h["text"])[:200]: h.get("entry_id", "") for h in hl}
            for p in rl:
                p["entry_id"] = bytext.get(norm(p["text"])[:200], "")
        merged, mstats = merge(ranked, cfg["total"], cfg.get("merge_per_source", 4),
                               drop_stubs=cfg.get("drop_stubs", True))
        results["arms"][name] = {"passages": merged, "stats": mstats,
                                 "chars": sum(len(p["text"]) for p in merged),
                                 "per_query_returned": [len(r) for r in ranked]}

    # judge every arm
    crit_block = "\n".join(f"{i}. [{r['points']:+g} pts] {r['criterion_text']}"
                           for i, r in enumerate(records))
    for name, a in results["arms"].items():
        if not a["passages"]:
            a["judged"] = [{"idx": i, "supplied": False, "why": "no passages"} for i in range(len(records))]
            continue
        async with sem:
            out = await llm(client, JUDGE_SYSTEM,
                            JUDGE_TEMPLATE.format(task=task[:4000], passages=fmt(a["passages"]),
                                                  criteria=crit_block), 1500)
        j = parse_json(out) or {}
        a["judged"] = j.get("results", []) if isinstance(j, dict) else []
    return results


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default="scripts/self_evolving/kb/probe_out/cat_LATENT.json")
    ap.add_argument("--n_tasks", type=int, default=60)
    ap.add_argument("--n_plan", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="scripts/self_evolving/kb/probe_out/multiquery_probe.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    from pymilvus import MilvusClient

    recs = json.load(open(args.latent))
    bytask = {}
    for r in recs:
        bytask.setdefault(r["question_id"], []).append(r)
    qids = sorted(bytask)
    random.Random(args.seed).shuffle(qids)
    qids = qids[: args.n_tasks]

    arms = {
        # TRUE control: the probe's own variant-A query, shipped policy, shipped k.
        "A0_probeq_k5": dict(n_queries=1, k=5, total=5, merge_per_source=3, use_query_a=True),
        "A0_probeq_k12": dict(n_queries=1, k=12, total=12, merge_per_source=4, use_query_a=True),
        # control: one query from this run's proposer, shipped k
        "A1_k5": dict(n_queries=1, k=5, total=5, merge_per_source=3),
        # more passages, still one query
        "A1_k12": dict(n_queries=1, k=12, total=12, merge_per_source=4),
        # the design: N sub-queries, small k each, merged
        "A2_k5": dict(n_queries=2, k=5, total=8, merge_per_source=4),
        "A3_k4": dict(n_queries=3, k=4, total=10, merge_per_source=4),
        "A4_k4": dict(n_queries=4, k=4, total=12, merge_per_source=4),
        # same but with the per-source cap relaxed inside each sub-query
        "A3_k4_cap6": dict(n_queries=3, k=4, total=10, merge_per_source=6, policy_per_source=6),
        # policy ablation: no EXCLUDE_EXPR
        "A3_k4_noexcl": dict(n_queries=3, k=4, total=10, merge_per_source=4, no_exclude=True),
        # tuned region: relax the merge cap and the fetch budget so the merge is
        # not throwing away what the sub-queries found
        "A4_k5_t12": dict(n_queries=4, k=5, total=12, merge_per_source=5, policy_per_source=5, fetch_mult=16),
        "A4_k5_t16": dict(n_queries=4, k=5, total=16, merge_per_source=6, policy_per_source=5, fetch_mult=16),
        "A6_k4_t16": dict(n_queries=6, k=4, total=16, merge_per_source=6, policy_per_source=4, fetch_mult=16),
        "A6_k4_t24": dict(n_queries=6, k=4, total=24, merge_per_source=8, policy_per_source=4, fetch_mult=16),
    }

    mc = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        tasks = [run_task(client, mc, sem, bytask[q][0]["task"], bytask[q], arms, args.n_plan) for q in qids]
        done = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for qid, res in zip(qids, done):
        if isinstance(res, Exception) or res is None:
            print("FAIL", qid, res)
            continue
        out.append({"question_id": qid, "task": bytask[qid][0]["task"],
                    "records": [{"criterion_text": r["criterion_text"], "points": r["points"],
                                 "met_rate": r["met_rate"], "latent_cause": r["latent_cause"],
                                 "query_a": r["query_a"]} for r in bytask[qid]],
                    **res})
    json.dump(out, open(args.out, "w"), indent=1)

    # ---- report
    print(f"\ntasks ok: {len(out)}/{len(qids)}   criteria: {sum(len(o['records']) for o in out)}")
    for name in arms:
        sup = tot = 0
        pts_sup = pts_tot = 0.0
        chars = []
        nps = []
        st = Counter()
        for o in out:
            a = o["arms"][name]
            chars.append(a["chars"])
            nps.append(len(a["passages"]))
            st.update(a["stats"])
            jm = {j.get("idx"): j.get("supplied") for j in a["judged"] if isinstance(j, dict)}
            for i, r in enumerate(o["records"]):
                tot += 1
                pts_tot += abs(r["points"])
                if jm.get(i):
                    sup += 1
                    pts_sup += abs(r["points"])
        print(f"{name:16s} recovered {sup:3d}/{tot} = {sup/max(1,tot):.3f}  "
              f"pts {pts_sup/max(1,pts_tot):.3f}  passages {sum(nps)/len(nps):.1f}  "
              f"chars {sum(chars)/len(chars):.0f}  ~tok {sum(chars)/len(chars)/3.6:.0f}  {dict(st)}")


if __name__ == "__main__":
    asyncio.run(main())
