"""Re-judge every *.jsonl response dump in a directory with a single judge model
(default gpt-chat-latest via the gcr TRAPI proxy), producing overall + per-ICD
(data_source) accuracy per dump. Reads each row's saved `extracted_answer`, joins
the question by hadm_id from the test set, and judges with the STANDARD lenient
prompt from verl's reward file -> identical methodology across all models.

Same as eval_full_gpt55/rejudge_all_models.py but SKIPS dumps already judged
(OUTDIR/<name>.json exists), so it can be re-run incrementally as new dumps land
(e.g. baseline generation finishing model-by-model) without re-judging.

    API_BASE=http://point.dd.works:18890/v1 API_KEY=$(cat .trapi_key) \\
    MODEL_NAME=gpt-chat-latest_2026-05-28 CONC=64 MAX_TOK=2048 REASONING=omit \\
    python rejudge_dumps_dir.py <DUMPS_DIR> <TEST.jsonl> <OUTDIR>
"""
import json, os, sys, asyncio, glob, importlib.util, collections
import aiohttp

DUMPS_DIR = sys.argv[1]
TEST = sys.argv[2]
OUTDIR = sys.argv[3]
API_BASE = os.environ["API_BASE"]; API_KEY = os.environ["API_KEY"]; MODEL = os.environ["MODEL_NAME"]
CONC = int(os.environ.get("CONC", "64"))
MAXTOK = int(os.environ.get("MAX_TOK", "2048"))
REASONING = os.environ.get("REASONING", "omit")  # "omit" = don't send (gpt-chat-latest forces medium)

os.makedirs(OUTDIR, exist_ok=True)
_REWARD = os.environ.get("REWARD_FILE",
    "/scratch/sheng/self_evolving/verl/verl/utils/reward_score/self_evolving.py")
spec = importlib.util.spec_from_file_location("se", _REWARD)
se = importlib.util.module_from_spec(spec); spec.loader.exec_module(se)

qmap = {}
for l in open(TEST):
    e = json.loads(l); ei = e.get("extra_info", {}) or {}
    hid = ei.get("hadm_id", e.get("hadm_id"))
    if hid is None: hid = (e.get("reward_model", {}) or {}).get("hadm_id")
    qmap[hid] = ei.get("question", "")

url = API_BASE.rstrip("/") + "/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


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


async def do_dump(session, sem, f):
    name = os.path.basename(f).replace(".jsonl", "")
    out_json = f"{OUTDIR}/{name}.json"
    if os.path.exists(out_json):
        r = json.load(open(out_json))
        print(f"SKIP {name} (already judged) | overall={r.get('overall'):.3f} (n={r.get('n')})", flush=True)
        return
    rows = [json.loads(l) for l in open(f)]
    done = [0]

    async def one(r):
        async with sem:
            q = qmap.get(r.get("hadm_id"), "")
            gts = r.get("ground_truth") or r.get("gts") or ""
            ans = str(r.get("extracted_answer", "")).strip()
            v = await judge(session, q, gts, ans)
            ds = r.get("data_source", "?")
            cat = ds.split("/")[-1] if "/" in ds else ds
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"    {name}: {done[0]}/{len(rows)}", flush=True)
            return cat, v
    res = await asyncio.gather(*[one(r) for r in rows])
    cc = collections.Counter(); ct = collections.Counter()
    for cat, v in res:
        ct[cat] += 1; cc[cat] += (1 if v >= 0.5 else 0)
    n = len(rows); overall = sum(1 for _, v in res if v >= 0.5) / n
    per_cat = {c: cc[c] / ct[c] for c in ct}
    print(f"RESULT {name} | overall={overall:.3f} (n={n}) | " +
          " ".join(f"{c}={per_cat[c]:.3f}" for c in sorted(per_cat)), flush=True)
    json.dump({"model": name, "overall": overall, "n": n,
               "per_cat": per_cat, "cat_n": dict(ct)}, open(out_json, "w"))


async def main():
    sem = asyncio.Semaphore(CONC)
    conn = aiohttp.TCPConnector(limit=CONC + 8)
    files = sorted(f for f in glob.glob(f"{DUMPS_DIR}/*.jsonl") if ".summary" not in f)
    print(f"judging {len(files)} dumps with {MODEL} (CONC={CONC}, MAXTOK={MAXTOK}, REASONING={REASONING})", flush=True)
    async with aiohttp.ClientSession(connector=conn) as session:
        for f in files:
            await do_dump(session, sem, f)
    print("ALL DUMPS DONE", flush=True)

asyncio.run(main())
