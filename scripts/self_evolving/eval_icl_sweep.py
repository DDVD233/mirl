"""Few-shot in-context-learning sweep for a frontier model on MIMIC-IV rare disease.

Produces the second curve of the paper's compute-scaling figure. Our curve spends
compute on *training* (proposing tasks, rolling out, judging, updating); a frontier
model with no training available spends compute at *inference*, by reading more
worked examples in its context. Sweeping the number of few-shot examples therefore
sweeps inference compute, and the two curves land on one axis.

Few-shot examples are drawn fresh per query from the train split, so no prefix is
shared across queries and the context is genuinely paid for on every case (a fixed
prompt would be cached and the compute axis would be a fiction). Every case is fully
multimodal -- the worked examples and the case being diagnosed both carry their X-ray
and ECG images -- so the zero-shot point of this sweep is the same measurement as the
full-benchmark zero-shot run. Images are what bound the shot count: each case brings
one to three of them, so the sweep stops where requests stop being accepted.

Scoring reuses verl's own lenient judge prompt and answer extraction, with the
judge pointed at the same model that judged inside the training loop, so the ICL
points and the training curve are directly comparable.

    python scripts/self_evolving/eval_icl_sweep.py \
        --k 0 1 2 4 8 16 --n_eval 400 \
        --api_base http://point.dd.works:18890/v1 --api_key "$TRAPI_KEY" \
        --judge_api_base http://point.dd.works:18184/v1 \
        --out_dir /scratch/self_evolving_datasets/eval_gpt56/icl
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import os
import random
import re
from pathlib import Path

import aiohttp

REWARD_FILE = os.environ.get(
    "REWARD_FILE", "/home/dvd/mirl/verl/utils/reward_score/self_evolving.py")
_spec = importlib.util.spec_from_file_location("se", REWARD_FILE)
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)


def clean_answer(ans: str) -> str:
    r"""Strip LaTeX wrappers the model puts inside \boxed{}, e.g. \text{...}."""
    a = ans.strip()
    for wrapper in (r"\text{", r"\textbf{", r"\mathrm{"):
        if a.startswith(wrapper):
            a = a[len(wrapper):]
    return a.rstrip("}").strip()


def load_rows(path, limit=None):
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def stratified_sample(rows, n, seed=0):
    """Sample n rows keeping the per-category proportions of the full set."""
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r.get("data_source", "?")].append(r)
    rng = random.Random(seed)
    out = []
    for cat, items in sorted(by_cat.items()):
        take = max(1, round(n * len(items) / len(rows)))
        rng.shuffle(items)
        out.extend(items[:take])
    rng.shuffle(out)
    return out[:n]


def read_image_b64(path, max_pixels=256 * 256):
    """Resize to the same pixel budget the trained actor sees, then base64-JPEG it.

    Mirrors eval_sota._read_image_b64 so the zero-shot point of this sweep is the
    same measurement as the full-benchmark zero-shot run. An unreadable image is
    dropped from the prompt and counted, never silently ignored.
    """
    if not os.path.exists(path):
        IMG_STATS["missing"] += 1
        return None
    try:
        import base64
        import math
        from io import BytesIO

        from PIL import Image

        with Image.open(path) as im:
            im.load()
            im = im.convert("RGB")
            w, h = im.size
            if w * h > max_pixels:
                scale = math.sqrt(max_pixels / (w * h))
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.BILINEAR)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=85)
        IMG_STATS["ok"] += 1
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        IMG_STATS["error"] += 1
        return None


IMG_STATS = {"ok": 0, "missing": 0, "error": 0}


def image_paths_of(row):
    """Rows store images as {"image": path, "max_pixels": n} dicts, or bare paths."""
    out = []
    for img in row.get("images") or []:
        p = img.get("image") or img.get("url") if isinstance(img, dict) else img
        if p:
            out.append(p)
    return out


def multimodal_content(question, image_paths):
    """Interleave the case's images at their <image> placeholders, as verl does."""
    pieces = re.split(r"<image>", question)
    content = []
    it = iter(image_paths or [])
    for i, piece in enumerate(pieces):
        if piece:
            content.append({"type": "text", "text": piece})
        if i < len(pieces) - 1:
            b64 = read_image_b64(next(it, ""))
            if b64:
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    for path in it:
        b64 = read_image_b64(path)
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def system_prompt(row):
    for m in row.get("prompt") or []:
        if m.get("role") == "system":
            return m["content"]
    return "You are a senior physician. Identify the single most likely primary diagnosis."


MAX_IMAGES = 50  # hard API limit per request


def build_user_content(shots, row, shot_images=True, max_images=MAX_IMAGES):
    """The full user turn: the worked examples, then the case to diagnose.

    With shot_images each example is a complete multimodal case. The API accepts at
    most `max_images` per request, so the query's own images are reserved first --
    they are the case being diagnosed and are never dropped -- and the remainder is
    spent on examples in order; any example past the budget falls back to text.
    Returns (content, n_images) so each sweep point can record what it actually sent.
    """
    query_imgs = image_paths_of(row)
    budget = max(0, max_images - len(query_imgs))
    content = []
    used = 0
    for i, ex in enumerate(shots, 1):
        content.append({"type": "text", "text": f"### Worked example {i}\n"})
        q = ex["extra_info"]["question"]
        ex_imgs = image_paths_of(ex) if shot_images else []
        if ex_imgs and used + len(ex_imgs) <= budget:
            content.extend(multimodal_content(q, ex_imgs))
            used += len(ex_imgs)
        else:
            content.append({"type": "text", "text": re.sub(r"<image>", "", q)})
        gt = ex["reward_model"]["ground_truth"]
        content.append({"type": "text",
                        "text": f"\n\nCorrect diagnosis: \\boxed{{{gt}}}\n\n"})
    if shots:
        content.append({"type": "text", "text": "### Now the case to diagnose\n"})
    content.extend(multimodal_content(row["extra_info"]["question"], query_imgs))
    n_images = sum(1 for c in content if c["type"] == "image_url")
    return content, n_images


async def call_model(session, url, headers, model, messages, max_tokens, attempts=8,
                     reasoning_effort=None):
    payload = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    for attempt in range(attempts):
        try:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=900)) as resp:
                if resp.status == 200:
                    d = await resp.json()
                    msg = d["choices"][0]["message"]
                    return msg.get("content") or "", d.get("usage", {})
                if resp.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(5 * 2 ** attempt, 180))
                    continue
                return "", {"error": f"HTTP {resp.status}"}
        except Exception:  # noqa: BLE001
            await asyncio.sleep(min(5 * 2 ** attempt, 60))
    return "", {"error": "retries exhausted"}


async def run_k(args, test_rows, train_rows, k):
    tag = f"k{k}" if not args.reasoning_effort else f"effort_{args.reasoning_effort}"
    out_path = Path(args.out_dir) / f"icl_{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done[r["hadm_id"]] = r
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in test_rows if r["extra_info"]["hadm_id"] not in done]
    print(f"[k={k}] {len(done)} done, {len(todo)} to run", flush=True)

    url = args.api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    conc = max(4, int(args.concurrency / (1 + k / 16)))
    sem = asyncio.Semaphore(conc)
    print(f"[k={k}] concurrency {conc}", flush=True)
    jsem = asyncio.Semaphore(args.judge_concurrency)
    fh = out_path.open("a")
    lock = asyncio.Lock()
    counter = [len(done)]
    failures = [0]

    async def one(session, row, idx):
        ei = row["extra_info"]
        rng = random.Random(args.seed * 100003 + idx)
        shots = rng.sample(train_rows, k) if k else []
        content, n_images = build_user_content(shots, row, shot_images=args.shot_images)
        messages = [{"role": "system", "content": system_prompt(row)},
                    {"role": "user", "content": content}]
        async with sem:
            text, usage = await call_model(session, url, headers, args.model, messages,
                                           args.max_tokens,
                                           reasoning_effort=args.reasoning_effort)
        if not text.strip():
            # retries exhausted: leave this case unrecorded so a resume retries it,
            # rather than persisting an empty answer that grades as wrong
            async with lock:
                failures[0] += 1
            return None
        ans = clean_answer(se.extract_final_answer(text) or "")
        gt = row["reward_model"]["ground_truth"]
        async with jsem:
            acc = await se.judge_accuracy_lenient(
                args.judge_api_base, args.judge_api_key, args.judge_model,
                ei["question"], gt, ans, None)
        rec = {"hadm_id": ei["hadm_id"], "data_source": row.get("data_source"),
               "k": k, "extracted_answer": ans, "judge_acc_lenient": float(acc),
               "ground_truth": gt, "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"),
               "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                   "reasoning_tokens"),
               "response_chars": len(text), "n_images": n_images}
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counter[0] += 1
            if counter[0] % 25 == 0:
                seen = [json.loads(l)["judge_acc_lenient"] for l in out_path.open()]
                print(f"[k={k}] {counter[0]}/{len(test_rows)} acc={sum(seen)/len(seen):.4f}",
                      flush=True)
        return rec

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(one(session, r, i) for i, r in enumerate(todo)))
    fh.close()

    rows = [json.loads(l) for l in out_path.open()]
    acc = sum(r["judge_acc_lenient"] for r in rows) / len(rows)
    ptok = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    ctok = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    summary = {"k": k, "n": len(rows), "acc_lenient": acc,
               "mean_prompt_tokens": sum(ptok) / len(ptok) if ptok else None,
               "mean_completion_tokens": sum(ctok) / len(ctok) if ctok else None}
    summary["reasoning_effort"] = args.reasoning_effort
    (Path(args.out_dir) / f"icl_{tag}.summary.json").write_text(json.dumps(summary, indent=2))
    summary["images"] = dict(IMG_STATS)
    summary["failed_requests_this_pass"] = failures[0]
    summary["sum_prompt_tokens"] = sum(r.get("prompt_tokens") or 0 for r in rows)
    summary["sum_completion_tokens"] = sum(r.get("completion_tokens") or 0 for r in rows)
    summary["sum_reasoning_tokens"] = sum(r.get("reasoning_tokens") or 0 for r in rows)
    nimg = [r.get("n_images") for r in rows if r.get("n_images") is not None]
    summary["mean_images_per_request"] = sum(nimg) / len(nimg) if nimg else None
    if failures[0]:
        print(f"[k={k}] WARNING {failures[0]} requests failed and were left unrecorded; "
              f"re-run to fill them in", flush=True)
    print(f"[k={k}] DONE acc={acc:.4f} n={len(rows)} "
          f"prompt_tok={summary['mean_prompt_tokens']} images={IMG_STATS}", flush=True)
    return summary


def n_for_k(k, n_base):
    """How many cases to evaluate at a given shot count.

    Prompt cost per case grows linearly in k (~2.1k tokens per example), so a
    256-shot pass over the full sample would be ~200M prompt tokens on a shared,
    rate-limited proxy. The large-k conditions therefore run on a nested prefix of
    the same stratified sample -- nested so the points stay paired rather than
    independently sampled, at the cost of a wider error bar where the curve is
    flattest anyway.
    """
    if k <= 32:
        return n_base
    if k <= 64:
        return min(n_base, 200)
    if k <= 128:
        return min(n_base, 150)
    return min(n_base, 100)


async def main_async(args):
    test_rows = load_rows(args.val_file)
    train_rows = load_rows(args.train_file, limit=args.train_pool)
    sample = stratified_sample(test_rows, args.n_eval, seed=args.seed)
    print(f"test set {len(test_rows)} -> sample {len(sample)}; train pool {len(train_rows)}")
    summaries = []
    for k in args.k:
        rows = sample[:n_for_k(k, args.n_eval)]
        summaries.append(await run_k(args, rows, train_rows, k))
    (Path(args.out_dir) / "icl_sweep.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--val_file", default="/scratch/self_evolving_datasets/mimiciv_rare/test.jsonl")
    ap.add_argument("--train_file",
                    default="/scratch/self_evolving_datasets/mimiciv_rare/train.jsonl")
    ap.add_argument("--train_pool", type=int, default=2000)
    ap.add_argument("--model", default="gpt-5.6-sol_2026-07-09")
    ap.add_argument("--api_base", default="http://point.dd.works:18890/v1")
    ap.add_argument("--api_key", default=os.environ.get("TRAPI_API_KEY", ""))
    ap.add_argument("--judge_model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--judge_api_base", default="http://point.dd.works:18184/v1")
    ap.add_argument("--judge_api_key", default="EMPTY")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--judge_concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasoning_effort", default=None,
                    help="none|low|medium|high|xhigh for gpt-5.6-sol (minimal/max rejected)")
    ap.add_argument("--shot_images", action="store_true", default=True,
                    help="give the worked examples their own images (default)")
    ap.add_argument("--no_shot_images", dest="shot_images", action="store_false")
    ap.add_argument("--out_dir", default="/scratch/self_evolving_datasets/eval_gpt56/icl")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
