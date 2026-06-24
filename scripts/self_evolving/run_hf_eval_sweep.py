"""Evaluate every uploaded MIMIC-rare model (ddvd233/mimiciv_rare_<run>) on the
FULL test set (2452) with the verl validation pipeline (run_val_only_hf.sh),
sequentially, in a single environment. Qwen first, Gemma last (per request).

Thinking is ON (per request): the reasoning models think, then answer. The earlier
truncation (responses cut at 4096 tokens before \\boxed{...}) was a too-small budget,
NOT thinking itself, so we run with a HIGH generation limit (max_response_length=16384,
max_model_len=32768) and enable_thinking=True for qwen. Gemma has no thinking mode
-> no flag.

Per model: run verl val_only with model.path=ddvd233/<repo>; OVERALL metrics come
from the per-sample dump (dump/<exp>/0.jsonl, 2452 rows), PER-CATEGORY metrics from
verl's console log (val-core/mimic_rare/<cat>/<metric>/mean@1). Headline `acc` ==
lenient LLM-judge disease match (gpt-5.3). Resumable (skips done), frees GPU + HF
cache between models.

    PYTHONUNBUFFERED=1 /usr/local/bin/python \
        scripts/self_evolving/run_hf_eval_sweep.py 2>&1 | tee eval_hf/sweep.log
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = "/scratch/sheng/self_evolving/verl"
SCRIPT = f"{REPO}/scripts/self_evolving/run_val_only_hf.sh"
OUT = "/scratch/sheng/self_evolving/eval_hf"
LOGS = f"{OUT}/logs"
RES = f"{OUT}/results"
DUMP = f"{OUT}/dump"
HF_HUB = "/scratch/sheng/self_evolving/hf_cache/hub"
HF_ORG = "ddvd233"
JUDGE = "gpt-5.3-chat_2026-03-03"
# Runs whose weights are NOT a ddvd233/mimiciv_rare_* repo (e.g. the untrained base).
BASE_REPOS = {"qwen36_27b_base_untrained": "Qwen/Qwen3.6-27B"}

# qwen first (headline self-improve first), gemma last.
MODELS = [
    "qwen36_27b_selfimprove",
    "qwen36_27b_selfimprove_sft",
    "qwen36_27b_full_gpt55",
    "qwen36_27b_full_kimi",
    "qwen36_27b_full_opd_qwen397b_v2",
    "qwen36_27b_deepseekv4pro",
    "qwen36_27b_sft_long",
    "qwen35_9b_selfimprove",
    "qwen35_9b_sft_long",
    "qwen35_9b_full_gpt55",
    # untrained base (after all qwen, before gemma) -> "did training help?" baseline
    "qwen36_27b_base_untrained",
    # --- gemma last (no thinking mode) ---
    "gemma4_31b_selfimprove_split",
    "gemma4_31b_rl_gpt51judge",
    "gemma4_e4b_rl_gpt51judge",
    "gemma4_e4b_sft",
    "gemma4_e4b_sft.gpt51_archived",
]

METRICS = ("acc", "judge_acc_lenient", "judge_acc_strict", "exact_acc",
           "embed_sim", "answer_quality", "reasoning_quality", "format_ok",
           "char_bleu", "score")
# verl console flat line: val-core/mimic_rare/<cat>/<metric>/mean@1:np.float64(0.34)
CAT_RE = re.compile(
    r"(val-(?:core|aux)/mimic_rare/[\w.-]+/\w+/mean@\d+):np\.float64\(([-0-9.eE]+)\)")


def log(m):
    print(f"[hfsweep {time.strftime('%H:%M:%S')}] {m}", flush=True)


def overall_from_dump(dump_path):
    """Mean of every metric over all per-sample rows (the dump has no data_source,
    so this is the exact micro-average over the full test set)."""
    if not os.path.exists(dump_path):
        return {}, 0
    rows = [json.loads(l) for l in open(dump_path) if l.strip()]
    n = len(rows)
    out = {}
    for k in METRICS:
        xs = [r[k] for r in rows
              if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        out[k] = round(sum(xs) / len(xs), 4) if xs else 0.0
    return out, n


def per_category_from_console(log_path):
    cats = {}
    if not os.path.exists(log_path):
        return cats
    txt = open(log_path, errors="ignore").read()
    for key, val in CAT_RE.findall(txt):
        m = re.match(r"val-(?:core|aux)/(mimic_rare/[\w.-]+)/(\w+)/mean@", key)
        if m:
            cats.setdefault(m.group(1), {})[m.group(2)] = round(float(val), 4)
    return cats


def free_hf_cache(run):
    for d in glob.glob(f"{HF_HUB}/models--{HF_ORG}--mimiciv_rare_{run.replace('.', '*')}"):
        shutil.rmtree(d, ignore_errors=True)
        log(f"  freed HF cache: {os.path.basename(d)}")


def main():
    for d in (LOGS, RES):
        os.makedirs(d, exist_ok=True)
    done, failed = [], []
    for i, run in enumerate(MODELS):
        repo = BASE_REPOS.get(run, f"{HF_ORG}/mimiciv_rare_{run}")
        exp = f"hfeval_{run}"
        res_path = f"{RES}/{run}.json"
        log_path = f"{LOGS}/{exp}.log"
        dump_path = f"{DUMP}/{exp}/0.jsonl"
        if os.path.exists(res_path):
            log(f"[{i+1}/{len(MODELS)}] SKIP {run} (have {res_path})")
            done.append(run)
            continue
        is_gemma = "gemma" in run.lower()
        extra = [] if is_gemma else ["+data.apply_chat_template_kwargs.enable_thinking=True"]
        log(f"[{i+1}/{len(MODELS)}] EVAL {run} -> {repo}  (thinking={'n/a-gemma' if is_gemma else 'ON'})")
        shutil.rmtree(f"{DUMP}/{exp}", ignore_errors=True)  # fresh dump
        t0 = time.time()
        env = {**os.environ, "MODEL": repo, "EXP": exp, "TP": "4", "GPUS": "4",
               "JUDGE_MODEL": JUDGE, "PYTHONUNBUFFERED": "1"}
        with open(log_path, "w") as lf:
            rc = subprocess.call(["bash", SCRIPT, *extra], cwd=REPO, env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        overall, n = overall_from_dump(dump_path)
        cats = per_category_from_console(log_path)
        ok = (rc == 0) and n > 0 and len(cats) >= 6
        result = {"run": run, "repo": repo, "judge": JUDGE, "rc": rc, "ok": ok,
                  "thinking": "on" if not is_gemma else "n/a",
                  "n_samples": n, "minutes": round(dt / 60, 1),
                  "overall": overall, "by_category": cats}
        json.dump(result, open(res_path, "w"), indent=2)
        log(f"  rc={rc} ok={ok} n={n} cats={len(cats)} {dt/60:.1f}min  "
            f"OVERALL acc(lenient)={overall.get('acc')} exact={overall.get('exact_acc')} "
            f"fmt={overall.get('format_ok')} embed={overall.get('embed_sim')}")
        for c in sorted(cats, key=lambda c: -cats[c].get("acc", 0)):
            log(f"     {c:30s} acc={cats[c].get('acc')}")
        (done if ok else failed).append(run)
        if ok:
            free_hf_cache(run)
        subprocess.call("pkill -9 -f 'VLLM::' 2>/dev/null; pkill -9 -f main_ppo 2>/dev/null; sleep 6",
                        shell=True)

    # consolidated
    allres = {r: json.load(open(f"{RES}/{r}.json")) for r in MODELS
              if os.path.exists(f"{RES}/{r}.json")}
    json.dump({"judge": JUDGE, "headline": "acc == lenient LLM-judge disease match",
               "n_models": len(allres), "runs": allres},
              open(f"{OUT}/results.json", "w"), indent=2)
    log(f"DONE ok={len(done)} failed={len(failed)} failed={failed}")
    log(f"wrote {OUT}/results.json ({len(allres)} models)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
