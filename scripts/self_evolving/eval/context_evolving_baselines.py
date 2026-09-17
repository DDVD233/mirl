#!/usr/bin/env python3
"""Training-free self-evolving baselines: GEPA and ACE, reimplemented from the papers.

Both methods improve a FROZEN backbone by rewriting its context instead of its weights:

  GEPA  (reflective prompt evolution, Agrawal et al. 2025). A pool of candidate system
        prompts. Each iteration picks a parent by Pareto-front sampling over per-instance
        scores on a held-out "pareto" set, runs it on a small minibatch, shows the prompt,
        the rollouts and their textual feedback to a reflection LM that writes a new
        prompt, and accepts the child only if it beats the parent on that minibatch; an
        accepted child is scored on the whole pareto set and joins the pool. The best
        pool member by mean pareto score is returned. (No merge/crossover.)
  ACE   (Agentic Context Engineering, Zhang et al. 2025). A playbook of itemized bullets
        with helpful/harmful counters. For each training sample a Generator answers with
        the playbook, a Reflector turns the outcome and feedback into lessons and tags
        the bullets it used, and a Curator converts lessons into incremental ADD
        operations that are merged deterministically with de-duplication (grow and
        refine). The playbook is frozen for evaluation (offline adaptation).

Every role (solver, reflector, curator) is the same frozen backbone, as in the other
frozen-backbone method baselines; only the scorer is the benchmark's standard judge.

WHERE THE LEARNING SIGNAL COMES FROM (this is the point of the protocol):
  --task hbpro   HealthBench Professional has NO training data here. The learners see only
                 tasks written by the fixed-prompt generator from the HealthBench Pro
                 description (the accepted pool of the fixed-prompt training run) and the
                 rubric judge's verdicts on those generated rubrics, scored with the same
                 length-adjusted rule the RL arms train on. The 525 validation tasks are
                 never generated on, scored, or reflected on during learning.
  --task mimic   The MIMIC-IV rare-diagnosis TRAIN split (labels allowed, as for the
                 train-set RL row of that paper); evaluation on the 2,452-case test split.

  learn : writes <out>/context.txt (the evolved prompt or playbook) plus a resumable state.
  eval  : answers the evaluation split once with the frozen context.
          hbpro -> $S/logs_hb9b/val_generations/<out_prefix>/0.jsonl (val-dump format, graded
                   afterwards by regrade_hbpro_dumps.py like every other row of the paper)
          mimic -> <out>/eval_cases.jsonl + eval_summary.json (stage-1 judge, lenient accuracy)

Examples (on a pod with the backbone served by vLLM at :8210):
  python3 context_evolving_baselines.py learn --method gepa --task hbpro --model qwen35_9b \
      --base-url http://localhost:8210/v1 --out $S/paper_refresh/context_baselines/gepa_hbpro_qwen35_9b
  python3 context_evolving_baselines.py eval --task hbpro --model qwen35_9b --base-url http://localhost:8210/v1 \
      --context-file .../context.txt --out-prefix hbpro_methods_qwen35_9b_gepa --out .../gepa_hbpro_qwen35_9b
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import importlib.util
import json
import os
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
S = os.environ.get("SE_ROOT", "/scratch/sheng/self_evolving")
LENGTH_CENTER, LENGTH_PENALTY_PER_500 = 2000.0, 0.0147   # HealthBench Pro length rule (as in the reward)

SEED_PROMPT = {
    "hbpro": "You are a medical AI assistant helping a clinician. Answer the request accurately and specifically.",
    "mimic": "Read the case carefully and commit to the single most likely diagnosis in the required format.",
}


async def patient(call, what, waits=40, pause=60):
    """The judge sits behind a shared tunnel host that stalls for minutes at a time. An outage
    must never be read as "criterion not met" / "wrong diagnosis": wait and ask again."""
    for attempt in range(waits):
        try:
            return await call()
        except Exception as e:
            print(f"[judge] {what} failed ({type(e).__name__}: {str(e)[:120]}); retry {attempt + 1}/{waits} "
                  f"in {pause}s", flush=True)
            await asyncio.sleep(pause)
    raise RuntimeError(f"judge unreachable for {waits * pause}s ({what})")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def with_context(messages, context):
    """Put the evolved context in the system turn (merged with an existing one)."""
    if not context.strip():
        return messages
    out = [dict(m) for m in messages]
    if out and out[0]["role"] == "system":      # one system turn only: Qwen templates reject a second one
        if isinstance(out[0]["content"], str):
            out[0]["content"] = context + "\n\n" + out[0]["content"]
        else:
            out[0]["content"] = [{"type": "text", "text": context + "\n\n"}] + list(out[0]["content"])
        return out
    return [{"role": "system", "content": context}] + out


def split_used(text):
    """ACE's generator ends with a `USED: b001, ...` bookkeeping line. It is not part of the
    answer: strip it before grading (and before the length rule) and return the ids."""
    m = re.search(r"\n?\s*USED:([^\n]*)\s*$", text)
    if not m:
        return text, []
    return text[: m.start()].rstrip(), re.findall(r"b\d{3}", m.group(1))


# ----------------------------------------------------------------------------------------
# Tasks: rollout + score + textual feedback for one instance under a given context
# ----------------------------------------------------------------------------------------
class HBProGenerated:
    """Tasks written by the fixed-prompt generator from the HealthBench Pro description."""

    def __init__(self, args, stage1, session):
        self.args, self.session = args, session
        self.chat = stage1.Chat(session, args)
        self.rg = load("regrade", HERE.parent / "analysis" / "regrade_hbpro_dumps.py")
        self.judge_args = SimpleNamespace(api_base=args.judge_base, api_key=os.environ[args.judge_key_env],
                                          model=args.judge_model, effort="omit", max_completion_tokens=512,
                                          timeout=180.0)
        self.judge_sem = asyncio.Semaphore(args.judge_concurrency)

    def load_pool(self):
        rows, seen = [], set()
        for f in sorted(glob.glob(os.path.join(self.args.pool_dir, "server_accepted_*.jsonl"))):
            for line in open(f):
                if not line.strip():
                    continue
                e = json.loads(line).get("entry") or {}
                ei = e.get("extra_info") or {}
                qid, items, conv = ei.get("question_id"), ei.get("rubric_items") or [], ei.get("conversation") or []
                if not qid or qid in seen or not items or not conv or e.get("images") or ei.get("images"):
                    continue
                seen.add(qid)
                rows.append({"id": qid, "messages": [{"role": m["role"], "content": m["content"]} for m in conv],
                             "items": [{"criterion": it.get("criterion_text") or it.get("criterion"),
                                        "points": float(it["points"])} for it in items]})
        return rows

    def describe(self, inst):
        return "\n".join(f"{m['role']}: {m['content']}" for m in inst["messages"])[-3000:]

    async def run(self, inst, context):
        trace = []
        try:
            answer = await self.chat.call(with_context(inst["messages"], context), trace, "answer")
        except Exception as e:  # an empty/failed generation scores zero, with the reason as feedback
            return {"answer": "", "score": 0.0, "feedback": f"No answer was produced ({type(e).__name__})."}
        answer, used = split_used(answer)
        conv = self.rg.conversation_text(inst["messages"], answer)

        async def one(it):
            prompt = (self.rg.GRADER_TEMPLATE.replace("<<conversation>>", conv)
                      .replace("<<rubric_item>>", f"[{it['points']:g}] {it['criterion']}"))
            async def ask():
                for _ in range(3):        # an unparseable verdict is re-asked, then counts as not met
                    raw = await self.rg.call_grader(self.session, self.judge_sem, self.judge_args, prompt)
                    met = self.rg.parse_met(self.rg.strip_thinking(raw))
                    if met is not None:
                        return met
                return False
            return await patient(ask, "rubric criterion")
        met = await asyncio.gather(*(one(it) for it in inst["items"]))
        pos = sum(it["points"] for it in inst["items"] if it["points"] > 0)
        raw_score = sum(it["points"] for it, m in zip(inst["items"], met) if m) / pos if pos > 0 else 0.0
        score = raw_score - LENGTH_PENALTY_PER_500 * (len(answer) - LENGTH_CENTER) / 500.0
        lines = []
        for it, m in zip(inst["items"], met):
            if it["points"] > 0:
                lines.append(("MET" if m else "NOT MET") + f" (+{it['points']:g}): {it['criterion']}")
            elif m:
                lines.append(f"PENALTY TRIGGERED ({it['points']:g}): {it['criterion']}")
        lines.append(f"Answer length: {len(answer)} characters (answers beyond {int(LENGTH_CENTER)} characters lose "
                     f"{LENGTH_PENALTY_PER_500:.4f} per extra 500). Rubric score {raw_score:.2f}, "
                     f"length-adjusted {score:.2f}.")
        return {"answer": answer, "used": used, "score": float(score), "feedback": "\n".join(lines)}


class MimicTrain:
    """MIMIC-IV rare-diagnosis cases with the stage-1 request builder, extraction and judge."""

    def __init__(self, args, stage1, session):
        self.args, self.session, self.stage1 = args, session, stage1
        self.chat = stage1.Chat(session, args)
        os.environ["REWARD_EVAL_LENIENT_ONLY"] = "1"
        self.eval_module = stage1.load_module("stage1_eval_sota", args.eval_module)
        self.reward = stage1.load_module("stage1_reward", args.reward_file)

    def load_pool(self, path=None):
        """The jsonl files point at /scratch/self_evolving_datasets/..., where a pod may hold only
        part of the images; the same files live under --image-fallback. A case whose image cannot
        be found at either place is left out (never scored as a wrong answer)."""
        rows, remapped, dropped = [], 0, 0
        for e in self.stage1.read_rows(Path(path or self.args.train_file)):
            ok = True
            for im in e.get("images") or []:
                if isinstance(im, dict) and im.get("image") and not os.path.exists(im["image"]):
                    alt = os.path.join(self.args.image_fallback, os.path.basename(im["image"]))
                    if os.path.exists(alt):
                        im["image"], remapped = alt, remapped + 1
                    else:
                        ok = False
            if ok:
                rows.append({"id": str(e["extra_info"]["hadm_id"]), "entry": e})
            else:
                dropped += 1
        print(f"[mimic] {len(rows)} cases, {remapped} image paths remapped, {dropped} cases dropped "
              f"for an image found nowhere", flush=True)
        return rows

    def describe(self, inst):
        text = [p["text"] for m in inst["entry"]["prompt"] if m["role"] == "user"
                for p in (m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}])
                if p.get("type") == "text"]
        return "\n".join(text)[-3000:]

    async def run(self, inst, context, keep_response=False):
        entry = inst["entry"]
        try:
            body, _ = self.stage1.build_request(entry, self.eval_module, self.args)
            response = await self.chat.call(with_context(body["messages"], context), [], "answer")
        except Exception as e:
            return {"answer": "", "score": 0.0, "feedback": f"No answer was produced ({type(e).__name__})."}
        gt = entry["reward_model"]["ground_truth"]
        row = {"extracted_answer": self.reward.extract_final_answer(response) or "", "ground_truth": gt}
        async def ask():
            try:
                return await self.stage1.grade(row, entry, self.session, self.args, self.reward)
            except ValueError:            # the judge answered, unparseably, on every attempt
                return {"verdict": "incorrect", "judge_acc_lenient": 0.0}
        verdict = await patient(ask, "diagnosis verdict")
        out = {"answer": response[-1500:], "used": split_used(response)[1], "score": float(verdict["judge_acc_lenient"]),
               "feedback": f"The assistant's final answer was '{row['extracted_answer'] or '(none extracted)'}'. "
                           f"The correct diagnosis is '{gt}'. Judged {verdict['verdict']}."}
        if keep_response:
            out.update(response=response, extracted=row["extracted_answer"], verdict=verdict["verdict"])
        return out


# ----------------------------------------------------------------------------------------
# GEPA
# ----------------------------------------------------------------------------------------
GEPA_REFLECT = """I provided an assistant with the following instructions to perform a task for me:
```
{prompt}
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
{examples}
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""


def pareto_parent(scores, rng):
    """GEPA's candidate selection. scores[k][i] = score of candidate k on pareto instance i.
    Keep, per instance, the candidates attaining the best score; drop candidates dominated by
    another (its winning instances are a strict subset of the other's, or equal with a lower
    mean); sample a survivor with probability proportional to the instances it wins."""
    n_inst = len(scores[0])
    wins = {k: set() for k in range(len(scores))}
    for i in range(n_inst):
        best = max(s[i] for s in scores)
        for k, s in enumerate(scores):
            if s[i] >= best - 1e-12:
                wins[k].add(i)
    alive = [k for k in wins if wins[k]]
    mean = lambda k: sum(scores[k]) / n_inst  # noqa: E731
    survivors = []
    for k in alive:
        dominated = any(j != k and (wins[k] < wins[j] or (wins[k] == wins[j] and (mean(j), -j) > (mean(k), -k)))
                        for j in alive)
        if not dominated:
            survivors.append(k)
    return rng.choices(survivors, weights=[len(wins[k]) for k in survivors], k=1)[0]


def extract_block(text):
    blocks = re.findall(r"```(?:[a-zA-Z]*\n)?(.*?)```", text, re.S)
    body = max(blocks, key=len) if blocks else text
    return body.strip()


async def learn_gepa(task, pool, args, reflect, out):
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    pareto, feedback = pool[: args.pareto_size], pool[args.pareto_size: args.pareto_size + args.train_size]
    state_path = out / "gepa_state.json"
    sem = asyncio.Semaphore(args.concurrency)

    async def run_many(insts, prompt):
        async def one(inst):
            async with sem:
                return await task.run(inst, prompt)
        return await asyncio.gather(*(one(i) for i in insts))

    if state_path.exists():
        st = json.loads(state_path.read_text())
    else:
        seed_res = await run_many(pareto, SEED_PROMPT[args.task])
        st = {"candidates": [{"prompt": SEED_PROMPT[args.task], "parent": None,
                              "scores": [r["score"] for r in seed_res]}],
              "rollouts": len(pareto), "iteration": 0, "log": []}
        state_path.write_text(json.dumps(st))
    order, cursor = list(range(len(feedback))), 0
    while st["rollouts"] < args.budget:
        st["iteration"] += 1
        k = pareto_parent([c["scores"] for c in st["candidates"]], rng)
        parent = st["candidates"][k]
        if cursor + args.minibatch > len(order):
            rng.shuffle(order); cursor = 0
        batch = [feedback[i] for i in order[cursor: cursor + args.minibatch]]
        cursor += args.minibatch
        before = await run_many(batch, parent["prompt"])
        st["rollouts"] += len(batch)
        entry = {"iteration": st["iteration"], "parent": k, "parent_minibatch": sum(r["score"] for r in before)}
        if all(r["score"] >= args.perfect_score for r in before):
            entry["outcome"] = "skipped: minibatch already perfect"
        else:
            examples = "\n\n".join(f"# Example {j + 1}\n## Inputs\n{task.describe(inst)}\n\n## Generated Outputs\n"
                                   f"{r['answer'][-2500:]}\n\n## Feedback\n{r['feedback']}"
                                   for j, (inst, r) in enumerate(zip(batch, before)))
            try:
                child = extract_block(await reflect(GEPA_REFLECT.format(prompt=parent["prompt"], examples=examples)))
            except Exception as e:
                child = ""
                entry["outcome"] = f"reflection failed: {type(e).__name__}"
            if child and child != parent["prompt"]:
                after = await run_many(batch, child)
                st["rollouts"] += len(batch)
                entry["child_minibatch"] = sum(r["score"] for r in after)
                if entry["child_minibatch"] > entry["parent_minibatch"]:
                    full = await run_many(pareto, child)
                    st["rollouts"] += len(pareto)
                    st["candidates"].append({"prompt": child, "parent": k, "scores": [r["score"] for r in full]})
                    entry["outcome"] = f"accepted as candidate {len(st['candidates']) - 1}, pareto mean " \
                                       f"{sum(r['score'] for r in full) / len(full):.4f}"
                else:
                    entry["outcome"] = "rejected: no minibatch improvement"
            elif "outcome" not in entry:
                entry["outcome"] = "rejected: reflection returned no new prompt"
        st["log"].append(entry)
        state_path.write_text(json.dumps(st))
        means = [sum(c["scores"]) / len(c["scores"]) for c in st["candidates"]]
        print(f"[gepa] it {st['iteration']} rollouts {st['rollouts']}/{args.budget} pool {len(means)} "
              f"best {max(means):.4f} | {entry.get('outcome')}", flush=True)
    means = [sum(c["scores"]) / len(c["scores"]) for c in st["candidates"]]
    best = max(range(len(means)), key=lambda k: means[k])
    (out / "context.txt").write_text(st["candidates"][best]["prompt"])
    (out / "learn_summary.json").write_text(json.dumps(
        {"method": "gepa", "task": args.task, "rollouts": st["rollouts"], "candidates": len(means),
         "seed_pareto_mean": means[0], "best_pareto_mean": means[best], "best_candidate": best}, indent=1))
    print(f"[gepa] done: candidate {best} pareto mean {means[best]:.4f} (seed {means[0]:.4f})", flush=True)


# ----------------------------------------------------------------------------------------
# ACE
# ----------------------------------------------------------------------------------------
ACE_SECTIONS = ["strategies_and_hard_rules", "domain_knowledge", "common_mistakes", "output_format_and_length"]
ACE_GENERATOR = """You have a PLAYBOOK of strategies, domain facts and pitfalls distilled from earlier attempts at this kind of task. Read it, apply the bullets that are relevant to the request, and ignore the rest. At the very end of your answer add one line of the form `USED: id, id, ...` listing the ids of the bullets you relied on (or `USED: none`).

PLAYBOOK
{playbook}"""
ACE_REFLECTOR = """You are the Reflector. Diagnose the assistant's attempt below using the feedback, then distill reusable lessons for future tasks of this kind (not for this one instance).

## Task input
{inputs}

## Assistant's answer
{answer}

## Feedback from the evaluator
{feedback}

## Playbook bullets the assistant said it used
{used}

Return ONLY a JSON object with these keys:
"error_identification": what went wrong or what was missing (or "none"),
"root_cause": why it happened,
"correct_approach": what the assistant should do on similar tasks,
"key_insights": a list of 1-3 short, general, self-contained lessons,
"bullet_tags": an object mapping each used bullet id to "helpful", "harmful" or "neutral"."""
ACE_CURATOR = """You are the Curator of a playbook. Given the current playbook and new lessons from a reflection, decide which lessons add something the playbook does not already say. Do not rewrite or remove existing bullets; only add.

## Current playbook
{playbook}

## New lessons
{lessons}

Return ONLY a JSON object {{"operations": [{{"type": "ADD", "section": "<one of: """ + ", ".join(ACE_SECTIONS) + """>", "content": "<one concise, general, self-contained bullet>"}}, ...]}}. Return an empty list when nothing new is worth adding."""


def render_playbook(pb):
    lines = []
    for sec in ACE_SECTIONS:
        items = [b for b in pb["bullets"] if b["section"] == sec]
        if items:
            lines.append(f"## {sec}")
            lines += [f"[{b['id']}] {b['content']}" for b in items]
    return "\n".join(lines) if lines else "(empty)"


def norm_tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def merge_operations(pb, ops, max_bullets):
    """Deterministic merge with de-duplication: a new bullet that overlaps an existing one
    (Jaccard >= 0.7 on word sets) is folded into it as a helpful vote instead of added."""
    added = 0
    for op in ops:
        if not isinstance(op, dict) or op.get("type") != "ADD":
            continue
        content, sec = str(op.get("content") or "").strip(), op.get("section")
        if len(content) < 12 or len(content) > 600:
            continue
        if sec not in ACE_SECTIONS:
            sec = ACE_SECTIONS[0]
        toks = norm_tokens(content)
        dup = next((b for b in pb["bullets"]
                    if len(toks & norm_tokens(b["content"])) / max(1, len(toks | norm_tokens(b["content"]))) >= 0.7), None)
        if dup:
            dup["helpful"] += 1
            continue
        if len(pb["bullets"]) >= max_bullets:   # refine: make room by dropping the worst-scoring bullet
            worst = min(pb["bullets"], key=lambda b: (b["helpful"] - b["harmful"], -b["created"]))
            if worst["helpful"] - worst["harmful"] > 0:
                continue
            pb["bullets"].remove(worst)
        pb["next_id"] += 1
        pb["bullets"].append({"id": f"b{pb['next_id']:03d}", "section": sec, "content": content,
                              "helpful": 0, "harmful": 0, "created": pb["next_id"]})
        added += 1
    return added


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


async def learn_ace(task, pool, args, reflect, out):
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    train = pool[: args.train_size]
    state_path = out / "ace_state.json"
    st = json.loads(state_path.read_text()) if state_path.exists() else {
        "playbook": {"bullets": [], "next_id": 0}, "done": 0, "epoch": 0, "scores": []}
    pb = st["playbook"]
    sem = asyncio.Semaphore(args.concurrency)

    async def attempt(inst, snapshot):
        async with sem:
            res = await task.run(inst, ACE_GENERATOR.format(playbook=snapshot))
            used = res.get("used") or []
            try:
                ref = parse_json(await reflect(ACE_REFLECTOR.format(
                    inputs=task.describe(inst), answer=res["answer"][-2500:], feedback=res["feedback"],
                    used=", ".join(used) or "none")))
            except Exception:
                ref = {}
            return res, used, ref

    total = args.train_size * args.epochs
    while st["done"] < total:
        start = st["done"] % args.train_size
        group = train[start: start + args.ace_batch]
        snapshot = render_playbook(pb)
        results = await asyncio.gather(*(attempt(inst, snapshot) for inst in group))
        for res, used, ref in results:          # curation is sequential: each delta sees the latest playbook
            st["scores"].append(res["score"])
            tags = ref.get("bullet_tags") if isinstance(ref.get("bullet_tags"), dict) else {}
            for b in pb["bullets"]:
                if tags.get(b["id"]) in ("helpful", "harmful"):
                    b[tags[b["id"]]] += 1
            lessons = [ref.get(k) for k in ("correct_approach",) if ref.get(k)] + list(ref.get("key_insights") or [])
            if not lessons:
                continue
            try:
                ops = parse_json(await reflect(ACE_CURATOR.format(
                    playbook=render_playbook(pb), lessons="\n".join(f"- {x}" for x in lessons)))).get("operations", [])
            except Exception:
                ops = []
            merge_operations(pb, ops if isinstance(ops, list) else [], args.max_bullets)
        st["done"] += len(group)
        state_path.write_text(json.dumps(st))
        recent = st["scores"][-50:]
        print(f"[ace] {st['done']}/{total} samples, {len(pb['bullets'])} bullets, "
              f"mean score of last {len(recent)}: {sum(recent) / len(recent):.4f}", flush=True)
    # The evaluation context is the playbook with the generator instruction, minus the USED line request.
    context = ("You have a PLAYBOOK of strategies, domain facts and pitfalls distilled from earlier attempts at this "
               "kind of task. Apply the bullets that are relevant to the request and ignore the rest. Do not mention "
               "the playbook in your answer.\n\nPLAYBOOK\n" + render_playbook(pb))
    (out / "context.txt").write_text(context)
    (out / "learn_summary.json").write_text(json.dumps(
        {"method": "ace", "task": args.task, "samples": st["done"], "bullets": len(pb["bullets"]),
         "mean_score_first50": sum(st["scores"][:50]) / max(1, len(st["scores"][:50])),
         "mean_score_last50": sum(st["scores"][-50:]) / max(1, len(st["scores"][-50:]))}, indent=1))
    print(f"[ace] done: {len(pb['bullets'])} bullets", flush=True)


# ----------------------------------------------------------------------------------------
# Evaluation with the frozen context
# ----------------------------------------------------------------------------------------
async def eval_hbpro(args, stage1, session, context, out):
    import pandas as pd
    chat = stage1.Chat(session, args)
    val = pd.read_parquet(args.val_parquet)
    rows = [{"row": i, "messages": [{"role": m["role"], "content": m["content"]} for m in r["extra_info"]["conversation"]]}
            for i, r in val.iterrows()]
    rows = rows[: args.limit] if args.limit else rows
    out_dir = Path(f"{S}/logs_hb9b/val_generations/{args.out_prefix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path, done = out_dir / "answers.jsonl", {}
    if cache_path.exists():
        done = {json.loads(l)["row"]: json.loads(l)["answer"] for l in open(cache_path) if l.strip()}
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    cache = open(cache_path, "a")

    async def one(item):
        if item["row"] in done:
            return
        async with sem:
            try:
                answer = await chat.call(with_context(item["messages"], context), [], "answer")
            except Exception:
                return
        async with lock:
            done[item["row"]] = answer
            cache.write(json.dumps({"row": item["row"], "answer": answer}) + "\n"); cache.flush()
    await asyncio.gather(*(one(it) for it in rows))
    cache.close()
    with open(out_dir / "0.jsonl", "w") as f:
        for it in rows:
            conv_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in it["messages"])
            f.write(json.dumps({"input": conv_text, "output": done.get(it["row"], ""), "method": args.out_prefix,
                                "model": args.model, "row": it["row"]}) + "\n")
    missing = sum(1 for it in rows if it["row"] not in done)
    (out / "eval_summary.json").write_text(json.dumps({"dump": str(out_dir / "0.jsonl"), "rows": len(rows),
                                                       "missing": missing, "context_chars": len(context)}, indent=1))
    print(f"[eval hbpro] wrote {out_dir / '0.jsonl'} ({len(rows)} rows, {missing} missing)", flush=True)


async def eval_mimic(args, stage1, session, context, out):
    task = MimicTrain(args, stage1, session)
    cases = task.load_pool(args.val_file)
    cases = cases[: args.limit] if args.limit else cases
    cache_path, done = out / "eval_cases.jsonl", {}
    if cache_path.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in open(cache_path) if l.strip()}
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    cache = open(cache_path, "a")

    async def one(inst):
        if inst["id"] in done:
            return
        async with sem:
            res = await task.run(inst, context, keep_response=True)
        if "verdict" not in res:      # infrastructure failure: leave it missing and retryable
            return
        rec = {"id": inst["id"], "category": str(inst["entry"].get("data_source", "")).split("/")[-1],
               "score": res["score"], "extracted": res["extracted"], "verdict": res["verdict"]}
        async with lock:
            done[inst["id"]] = rec
            cache.write(json.dumps(rec) + "\n"); cache.flush()
    await asyncio.gather(*(one(c) for c in cases))
    cache.close()
    acc = sum(done[c["id"]]["score"] for c in cases if c["id"] in done) / len(cases)   # missing counts as wrong
    by_cat = {}
    for c in cases:                                # per ICD chapter, as in the stage-1 main table
        cat = str(c["entry"].get("data_source", "")).split("/")[-1]
        n, tot = by_cat.get(cat, (0, 0.0))
        by_cat[cat] = (n + 1, tot + (done[c["id"]]["score"] if c["id"] in done else 0.0))
    (out / "eval_summary.json").write_text(json.dumps(
        {"cases": len(cases), "answered": sum(1 for c in cases if c["id"] in done),
         "judge_acc_lenient": acc, "by_category": {k: {"n": n, "acc": t / n} for k, (n, t) in sorted(by_cat.items())},
         "context_chars": len(context)}, indent=1))
    print(f"[eval mimic] {len(cases)} cases, lenient accuracy {acc:.4f}", flush=True)


# ----------------------------------------------------------------------------------------
async def main_async(args):
    import aiohttp
    stage1 = load("mimic_method_baselines", HERE / "mimic_method_baselines.py")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        if args.command == "eval":
            context = Path(args.context_file).read_text()
            await (eval_hbpro if args.task == "hbpro" else eval_mimic)(args, stage1, session, context, out)
            return
        task = (HBProGenerated if args.task == "hbpro" else MimicTrain)(args, stage1, session)
        pool = task.load_pool()
        print(f"[learn] {args.method} on {args.task}: pool of {len(pool)} instances", flush=True)
        chat = stage1.Chat(session, args)

        async def reflect(prompt):            # every evolving role is the same frozen backbone
            return await chat.call([{"role": "user", "content": prompt}], [], "reflect", tokens=args.reflect_tokens)
        (out / "config.json").write_text(json.dumps({k: v for k, v in vars(args).items()}, indent=1, default=str))
        await (learn_gepa if args.method == "gepa" else learn_ace)(task, pool, args, reflect, out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("learn", "eval"))
    p.add_argument("--method", choices=("gepa", "ace"))
    p.add_argument("--task", choices=("hbpro", "mimic"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--provider", default="vllm")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--reasoning-effort", default="")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--reflect-tokens", type=int, default=4096)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    # learning data
    p.add_argument("--pool-dir", default=f"{S}/logs_hb9b/hb9b_specgap_simple_retrieval_websearch",
                   help="hbpro: accepted tasks of the fixed-prompt generator (HealthBench Pro description)")
    p.add_argument("--train-file", default=f"{S}/mimiciv_rare/train.jsonl")
    p.add_argument("--image-fallback", default=f"{S}/mimiciv_rare/ecg_images")
    p.add_argument("--train-size", type=int, default=300)
    # GEPA
    p.add_argument("--pareto-size", type=int, default=100)
    p.add_argument("--minibatch", type=int, default=3)
    p.add_argument("--budget", type=int, default=1500, help="rollout budget (metric calls)")
    p.add_argument("--perfect-score", type=float, default=1.0)
    # ACE
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--ace-batch", type=int, default=8)
    p.add_argument("--max-bullets", type=int, default=120)
    # evaluation
    p.add_argument("--context-file")
    p.add_argument("--out-prefix", help="hbpro eval: dump directory name under val_generations/")
    p.add_argument("--val-parquet", default=f"{S}/healthbench_pro_val.parquet")
    p.add_argument("--val-file", default=f"{S}/mimiciv_rare/test.jsonl")
    # judge + stage-1 protocol pieces
    p.add_argument("--judge-base", default="http://point.dd.works:18890/v1")
    p.add_argument("--judge-model", default="gpt-chat-latest_2026-05-28")
    p.add_argument("--judge-key-env", default="TRAPI_API_KEY")
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=65536)
    p.add_argument("--max-text-chars", type=int, default=9000)
    p.add_argument("--eval-module", default=str(HERE.parent / "eval_sota.py"))
    p.add_argument("--reward-file", default=f"{S}/verl/verl/utils/reward_score/self_evolving.py",
                   help="mimic: the STAGE-1 checkout's reward file (judge prompt + extraction)")
    a = p.parse_args()
    if a.command == "learn" and not a.method:
        p.error("learn needs --method")
    if a.command == "eval" and (not a.context_file or (a.task == "hbpro" and not a.out_prefix)):
        p.error("eval needs --context-file (and --out-prefix for hbpro)")
    return a


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
