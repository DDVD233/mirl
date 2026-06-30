"""Re-judge verl validation-generation traces (trainer.validation_data_dir dumps)
with a single judge model (default gpt-chat-latest via the gcr TRAPI proxy),
producing overall + per-ICD-category (data_source) accuracy per trace file.

Unlike eval_sota dumps, these training-time val traces have NO hadm_id /
data_source per row — but they are written in the SAME ORDER as the val set, so
we index-join against test.jsonl (row i of the trace <-> row i of test.jsonl) to
recover the question text and ICD category. The ground truth (`gts`) and the
model's `extracted_answer` are read straight from each trace row. The judge call
is byte-identical to rejudge_dumps_dir.py (STANDARD lenient prompt from verl's
reward file) -> directly comparable to the eval_full_gpt55/eval_baselines rejudge.

We assert the gts sequence matches between trace and test.jsonl before judging,
so a misaligned dump fails loud instead of silently mislabeling categories.

    API_BASE=http://point.dd.works:18890/v1 API_KEY=$(cat .trapi_key) \\
    MODEL_NAME=gpt-chat-latest_2026-05-28 CONC=64 MAX_TOK=2048 REASONING=omit \\
    python rejudge_val_traces.py <TEST.jsonl> <OUTDIR> <trace1.jsonl> [trace2.jsonl ...]

Each <trace>.jsonl is judged into <OUTDIR>/<parentdir>__step<N>.json (skipped if
it already exists, so the sweep is resumable).
"""
import json, os, sys, asyncio, collections, importlib.util, re
import aiohttp

TEST = sys.argv[1]
OUTDIR = sys.argv[2]
TRACES = sys.argv[3:]
API_BASE = os.environ["API_BASE"]; API_KEY = os.environ["API_KEY"]; MODEL = os.environ["MODEL_NAME"]
CONC = int(os.environ.get("CONC", "64"))
MAXTOK = int(os.environ.get("MAX_TOK", "2048"))
REASONING = os.environ.get("REASONING", "omit")  # "omit" => don't send (gpt-chat-latest forces medium)

os.makedirs(OUTDIR, exist_ok=True)
_REWARD = os.environ.get("REWARD_FILE",
    "/scratch/sheng/self_evolving/verl/verl/utils/reward_score/self_evolving.py")
spec = importlib.util.spec_from_file_location("se", _REWARD)
se = importlib.util.module_from_spec(spec); spec.loader.exec_module(se)

# index-aligned reference: question, ground truth, ICD category per test row
TQ, TGT, TCAT = [], [], []
for l in open(TEST):
    e = json.loads(l); ei = e.get("extra_info", {}) or {}
    TQ.append(ei.get("question", ""))
    TGT.append((e.get("reward_model", {}) or {}).get("ground_truth", ""))
    ds = e.get("data_source", "?")
    TCAT.append(ds.split("/")[-1] if "/" in ds else ds)
N = len(TQ)

url = API_BASE.rstrip("/") + "/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def tag_for(path):
    parent = os.path.basename(os.path.dirname(path))
    parent = re.sub(r"_20\d{6}_\d{6}$", "", parent).replace("mimiciv_rare_", "")
    step = os.path.basename(path).replace(".jsonl", "")
    return f"{parent}__step{step}"


async def judge(session, q, gts, ans):
    if not ans:
        return 0.0
    up = (f"Question: {q}\nGround truth answer: {gts}\n"
          f"Model's extracted answer: {ans}\n\nIs the model's answer correct under the rubric above?")
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": se.JUDGE_ACCURACY_LENIENT_PROMPT},
                            {"role": "user", "content": up}],
               "max_completion_tokens": MAXTOK}
    if REASONING != "omit":
        payload["reasoning_effort"] = REASONING
    for attempt in range(6):
        try:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=150)) as resp:
                if resp.status == 200:
                    d = await resp.json()
                    c = d["choices"][0]["message"].get("content") or ""
                    return 1.0 if se._extract_judge_verdict(c) == "correct" else 0.0
                elif resp.status == 429:
                    await asyncio.sleep(min(2 ** attempt, 20)); continue
                else:
                    return 0.0
        except Exception:
            await asyncio.sleep(min(2 ** attempt, 10)); continue
    return 0.0


async def do_trace(session, sem, f):
    name = tag_for(f)
    out_json = f"{OUTDIR}/{name}.json"
    if os.path.exists(out_json):
        r = json.load(open(out_json))
        print(f"SKIP {name} (already judged) | overall={r.get('overall'):.3f} (n={r.get('n')})", flush=True)
        return
    rows = [json.loads(l) for l in open(f)]
    if len(rows) != N:
        print(f"SKIP {name}: {len(rows)} rows != {N} test rows (partial/smoke val)", flush=True)
        return
    # alignment guard: ground truth must match index-for-index with test.jsonl
    mism = sum(1 for i, r in enumerate(rows) if (r.get("gts") or "") != TGT[i])
    if mism > 0:
        raise SystemExit(f"{name}: {mism}/{N} gts mismatches vs test.jsonl — order not aligned, refusing to mislabel")
    done = [0]

    async def one(i, r):
        async with sem:
            ans = str(r.get("extracted_answer", "")).strip()
            v = await judge(session, TQ[i], TGT[i], ans)
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"    {name}: {done[0]}/{N}", flush=True)
            return TCAT[i], v
    res = await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
    cc = collections.Counter(); ct = collections.Counter()
    for cat, v in res:
        ct[cat] += 1; cc[cat] += (1 if v >= 0.5 else 0)
    overall = sum(1 for _, v in res if v >= 0.5) / N
    per_cat = {c: cc[c] / ct[c] for c in ct}
    print(f"RESULT {name} | overall={overall:.3f} (n={N}) | " +
          " ".join(f"{c}={per_cat[c]:.3f}" for c in sorted(per_cat)), flush=True)
    json.dump({"model": name, "overall": overall, "n": N,
               "per_cat": per_cat, "cat_n": dict(ct)}, open(out_json, "w"))


async def main():
    sem = asyncio.Semaphore(CONC)
    conn = aiohttp.TCPConnector(limit=CONC + 8)
    print(f"judging {len(TRACES)} traces with {MODEL} (CONC={CONC}, MAXTOK={MAXTOK}, REASONING={REASONING}, N={N})", flush=True)
    async with aiohttp.ClientSession(connector=conn) as session:
        for f in TRACES:
            await do_trace(session, sem, f)
    print("ALL TRACES DONE", flush=True)

asyncio.run(main())
