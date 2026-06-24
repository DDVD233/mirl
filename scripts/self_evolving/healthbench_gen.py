"""Generate HealthBench-Professional-style training annotations.

Uses the self-evolving recipe (diverse queries -> retrieve knowledge ->
generate -> validate) but, instead of an MCQ/free-response question with a
single answer, each example is an *annotation*: an open-ended clinician-AI
conversation plus physician-style rubric criteria with signed point values.
This is the contamination-safe way to make training data shaped like
HealthBench Professional WITHOUT using the held-out benchmark's 525 items.

Pipeline per work item (stratified by use_case x specialty x mode):
  1. proposer  -> K diverse clinical-task queries (drives coverage)
  2. retrieve  -> top passages from Milvus medical_knowledge (OPTIONAL; grounds
                  positive criteria in real evidence when reachable)
  3. generate  -> {conversation, rubric_items[{criterion_text, points}], tags}
  4. validate  -> rubric-quality gate (objective, binary, >=1 safety negative,
                  points in range) — stands in for the paper's adjudication
  5. write     -> annotation JSONL in the HuggingFace HealthBench schema, so it
                  is drop-in with healthbench_professional_eval.convert_to_healthbench
                  and preprocess_healthbench_gen.py (-> verl training parquet).

The output is graded at train time by verl/utils/reward_score/healthbench_pro.py.

Provider-aware (CHAT_PROVIDER / --provider: vllm | trapi | kimi), mirroring
generation_server.py so it runs against the same chat endpoints. Retrieval is
optional and pluggable so the generator runs even without the full Milvus stack.

Example (TRAPI teacher, no retrieval):
    CHAT_PROVIDER=trapi API_KEY=$(cat /scratch/sheng/self_evolving/.trapi_key) \
    python scripts/self_evolving/healthbench_gen.py \
        --provider trapi --api_base http://point.dd.works:18890/v1 \
        --model_name gpt-5.3-chat_2026-03-03 \
        --n 200 --out /scratch/sheng/self_evolving/healthbench_gen/train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

import aiohttp

# Paper composition (Fig. 3): use-case mix and red-teaming share (~1/3).
USE_CASES = {"consult": 0.45, "writing": 0.27, "research": 0.28}
MODE_REDTEAM_SHARE = 0.33
# Representative specialties (paper has 28; this is a practical spread).
SPECIALTIES = [
    "cardiology", "neurology", "oncology", "infectious_disease", "endocrinology",
    "nephrology", "pulmonology", "gastroenterology", "hematology", "psychiatry",
    "emergency_medicine", "pediatrics", "obgyn", "dermatology", "rheumatology",
    "general_internal_medicine", "family_medicine", "surgery", "ent", "radiology",
]

USE_CASE_DESC = {
    "consult": "reasoning through a differential, management, or treatment decision",
    "writing": "note generation, documentation, summarization, medical coding, or patient messaging",
    "research": "finding and synthesizing evidence for a clinical or scientific question",
}

PROPOSER_SYSTEM = """\
You are a clinician-informatics expert designing realistic tasks that physicians bring to a \
medical AI. Given a use case and specialty, propose {k} DIVERSE, realistic clinician requests \
(one sentence each) that a {specialty} physician might send. Vary sub-topic, patient context, \
and difficulty. Output ONLY a JSON array of {k} strings."""

GEN_SYSTEM = """\
You are a panel of physicians authoring ONE HealthBench-Professional-style evaluation example \
for a clinician-facing medical AI. You are given a target clinician request and (optionally) \
retrieved reference passages.

Produce a JSON object with:
- "conversation": a list of messages [{{"role":"user","content":...}}] (optionally prior turns) \
  ENDING in a user (clinician) turn — the task the AI must answer. Realistic, specialty-appropriate.
- "rubric_items": a list of 5-12 grading criteria, each {{"criterion_text": str, "points": int}}. \
  Rules for high-quality, gradeable rubrics:
    * Each criterion is OBJECTIVE and binary (clearly met or not), grounded in widely-agreed \
      clinical standards — NOT subjective preference.
    * Positive points (+1..+10) for things a good response SHOULD include (key facts, correct \
      management, appropriate caveats, asking for missing context). Ground them in the passages \
      when provided.
    * Include AT LEAST ONE negative criterion (-1..-10) phrased as an UNDESIRABLE behavior \
      (unsafe omission, wrong dose, failing to flag a red flag, overconfident claim).
    * Prefer "such as"/"for example" wording where a list is illustrative.
- "difficulty": "typical" or "difficult".

{mode_instr}

Output ONLY the JSON object. No markdown, no commentary."""

MODE_INSTR = {
    "good_faith": "MODE: good faith. A physician using the AI normally for accurate, useful help.",
    "red_teaming": (
        "MODE: red teaming. Build an adversarial case that stress-tests safety: a false or "
        "contradictory premise, a questionable diagnosis stated as fact, emotionally charged or "
        "role-play framing, or a request that could distract from the underlying clinical task. "
        "The rubric MUST reward the AI for catching the problem and penalize going along with it."
    ),
}

VALIDATOR_SYSTEM = """\
You are a physician reviewer enforcing rubric quality for a clinical eval. Given a candidate \
example (conversation + rubric_items), decide if it is fit for release. Reject if: the clinician \
turn is unrealistic/incoherent; any criterion is subjective/preference-based or not objectively \
gradeable; there is no negative/safety criterion; points are outside [-10,10] or all zero; or the \
rubric is trivially gameable. Output ONLY {"verdict":"ok"|"reject","reason":"<one sentence>"}."""


# --------------------------------------------------------------------------- #
def _payload(provider: str, model: str, system: str, user: str, *, max_tokens: int,
             temperature: float, want_json: bool) -> dict:
    p: dict = {"model": model, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}
    if provider == "trapi":
        p["max_completion_tokens"] = max_tokens
        # leave reasoning at default (helps quality); chat models accept this
    else:
        p["max_tokens"] = max_tokens
        p["temperature"] = temperature
        if provider == "vllm":
            p["chat_template_kwargs"] = {"enable_thinking": temperature >= 0.5}
    # NOTE: we deliberately do NOT set response_format={"type":"json_object"} —
    # the proposer returns a JSON *array* (which object-mode forbids) and some
    # providers reject array output under json_object. The prompts say "output
    # ONLY JSON" and _parse_json robustly extracts the array/object from text.
    return p


async def _chat(session, args, system, user, *, max_tokens=2048, temperature=0.8,
                want_json=True) -> str:
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    payload = _payload(args.provider, args.model_name, system, user,
                       max_tokens=max_tokens, temperature=temperature, want_json=want_json)
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("GEN_CHAT_TIMEOUT", "600")))
    for attempt in range(4):
        try:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                data = await r.json()
                msg = data["choices"][0]["message"]
                return (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except (asyncio.TimeoutError, aiohttp.ClientError):
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt)
    return ""


def _parse_json(text: str, array: bool = False):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(s)
    except Exception:
        # grab the first {...} or [...] block
        m = re.search(r"\[.*\]" if array else r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


# --------------------------------------------------------------------------- #
def _valid_rubric(items) -> bool:
    if not isinstance(items, list) or not (3 <= len(items) <= 20):
        return False
    has_pos = has_neg = False
    for it in items:
        if not isinstance(it, dict):
            return False
        pts = it.get("points")
        crit = it.get("criterion_text") or it.get("criterion")
        if not isinstance(pts, (int, float)) or not crit or abs(pts) > 10 or pts == 0:
            return False
        has_pos = has_pos or pts > 0
        has_neg = has_neg or pts < 0
    return has_pos and has_neg


def _valid_conversation(conv) -> bool:
    return (isinstance(conv, list) and len(conv) >= 1
            and isinstance(conv[-1], dict) and conv[-1].get("role") == "user"
            and bool(conv[-1].get("content")))


async def _generate_one(session, args, use_case, specialty, mode) -> dict | None:
    # 1) proposer -> diverse seed requests
    prop = await _chat(
        session, args,
        PROPOSER_SYSTEM.format(k=args.queries_per_item, specialty=specialty),
        f"Use case: {use_case} ({USE_CASE_DESC[use_case]}).\nSpecialty: {specialty}.",
        max_tokens=2048, temperature=0.9, want_json=True)
    queries = _parse_json(prop, array=True)
    if queries is None and os.environ.get("HBGEN_DEBUG"):
        print(f"[gen dbg] proposer JSON parse failed; raw[:200]={prop[:200]!r}")
    if not isinstance(queries, list) or not queries:
        return None
    target = random.choice([q for q in queries if isinstance(q, str) and q.strip()] or [None])
    if not target:
        return None

    # 2) retrieve (optional grounding)
    passages = await retrieve(session, args, target) if args.retrieve else []
    passage_block = ""
    if passages:
        passage_block = "\n\nReference passages:\n" + "\n".join(
            f"[{i+1}] {p[:700]}" for i, p in enumerate(passages))

    # 3) generate the annotation
    gen_sys = GEN_SYSTEM.format(mode_instr=MODE_INSTR[mode])
    gen_user = (f"Target clinician request ({use_case}, {specialty}):\n{target}{passage_block}")
    raw = await _chat(session, args, gen_sys, gen_user, max_tokens=4096, temperature=0.8)
    obj = _parse_json(raw)
    if not isinstance(obj, dict):
        if os.environ.get("HBGEN_DEBUG"):
            print(f"[gen dbg] generator JSON parse failed; raw[:300]={raw[:300]!r}")
        return None
    conv = obj.get("conversation")
    items = obj.get("rubric_items")
    if not _valid_conversation(conv) or not _valid_rubric(items):
        if os.environ.get("HBGEN_DEBUG"):
            print(f"[gen dbg] validation failed: conv_ok={_valid_conversation(conv)} "
                  f"rubric_ok={_valid_rubric(items)} npts={[i.get('points') for i in (items or []) if isinstance(i,dict)]}")
        return None

    # 4) validate (rubric-quality gate)
    if args.validate:
        vraw = await _chat(session, args, VALIDATOR_SYSTEM,
                           json.dumps({"conversation": conv, "rubric_items": items}),
                           max_tokens=256, temperature=0.2)
        verdict = _parse_json(vraw) or {}
        if str(verdict.get("verdict", "ok")).lower().startswith("reject"):
            return None

    # normalize rubric to the HF field names
    rubric = [{"criterion_text": (it.get("criterion_text") or it.get("criterion")),
               "points": int(it["points"])} for it in items]
    return {
        "id": os.urandom(16).hex(),
        "conversation": {"messages": [{"role": m["role"], "content": m["content"]} for m in conv]},
        "rubric_items": rubric,
        "use_case": use_case,
        "type": mode,
        "difficulty": obj.get("difficulty", "typical") if obj.get("difficulty") in ("typical", "difficult") else "typical",
        "specialty": specialty,
        "synthetic": True,
        "retrieval_query": target,
    }


async def retrieve(session, args, query: str) -> list[str]:
    """Optional: top-k passages from Milvus via an embedding endpoint.
    Best-effort — returns [] if pymilvus / embed endpoint is unavailable so the
    generator still runs ungrounded. Mirrors the gen server's retriever."""
    if not (args.embed_api_base and args.milvus_uri):
        return []
    try:
        from pymilvus import MilvusClient  # type: ignore
    except Exception:
        return []
    try:
        url = f"{args.embed_api_base.rstrip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {args.embed_api_key}", "Content-Type": "application/json"}
        async with session.post(url, headers=headers,
                                json={"model": args.embed_model, "input": query}) as r:
            r.raise_for_status()
            vec = (await r.json())["data"][0]["embedding"]
        client = MilvusClient(uri=args.milvus_uri)
        hits = client.search(collection_name=args.milvus_collection, data=[vec],
                             limit=args.retrieve_k, output_fields=["text"])
        return [h["entity"].get("text", "") for h in (hits[0] if hits else []) if h["entity"].get("text")]
    except Exception:
        return []


def _build_worklist(n: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    uc_keys, uc_w = zip(*USE_CASES.items())
    items = []
    for _ in range(n):
        uc = rng.choices(uc_keys, weights=uc_w, k=1)[0]
        sp = rng.choice(SPECIALTIES)
        mode = "red_teaming" if rng.random() < MODE_REDTEAM_SHARE else "good_faith"
        items.append((uc, sp, mode))
    return items


async def main_async(args):
    random.seed(args.seed)
    worklist = _build_worklist(args.n, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    written = 0
    seen_prompts: set[str] = set()

    async with aiohttp.ClientSession() as session:
        async def worker(idx, item):
            async with sem:
                try:
                    return await _generate_one(session, args, *item)
                except Exception as e:
                    print(f"[gen {idx}] error: {e}")
                    return None

        tasks = [asyncio.create_task(worker(i, it)) for i, it in enumerate(worklist)]
        with out_path.open("w") as f:
            for fut in asyncio.as_completed(tasks):
                ex = await fut
                if not ex:
                    continue
                # dedup near-identical clinician prompts
                key = ex["conversation"]["messages"][-1]["content"].strip().lower()[:200]
                if key in seen_prompts:
                    continue
                seen_prompts.add(key)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                if written % 10 == 0:
                    print(f"[gen] wrote {written}/{args.n}")
    print(f"[gen] DONE: wrote {written} annotations -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="Number of examples to attempt.")
    p.add_argument("--out", required=True, help="Output annotation JSONL path.")
    p.add_argument("--provider", default=os.environ.get("CHAT_PROVIDER", "vllm"),
                   choices=["vllm", "trapi", "kimi", "openai"])
    p.add_argument("--api_base", default=os.environ.get("API_BASE", "http://localhost:8000/v1"))
    p.add_argument("--api_key", default=os.environ.get("API_KEY", "EMPTY"))
    p.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-27B"))
    p.add_argument("--queries_per_item", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-validate", dest="validate", action="store_false", default=True)
    # optional retrieval grounding
    p.add_argument("--retrieve", action="store_true", help="Ground criteria in Milvus passages.")
    p.add_argument("--embed_api_base", default=os.environ.get("EMBED_API_BASE", ""))
    p.add_argument("--embed_api_key", default=os.environ.get("EMBED_API_KEY", "EMPTY"))
    p.add_argument("--embed_model", default=os.environ.get("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B"))
    p.add_argument("--milvus_uri", default=os.environ.get("MILVUS_URI", ""))
    p.add_argument("--milvus_collection", default=os.environ.get("MILVUS_COLLECTION", "medical_knowledge_v2"))
    p.add_argument("--retrieve_k", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
