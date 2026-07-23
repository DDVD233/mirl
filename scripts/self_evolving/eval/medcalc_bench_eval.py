"""Evaluate a model on MedCalc-Bench-Verified (Khandekar et al., NeurIPS 2024;
verified fork: https://github.com/nikhilk7153/MedCalc-Bench-Verified).

MedCalc-Bench is a 1k-example benchmark of clinical *calculation* tasks: each
example is a patient note + a question asking for a numeric / date / tuple
score (55 calculators across 7 categories: lab test, physical, date, dosage,
risk, severity, diagnosis). The model must produce a JSON answer; correctness
is an EXACT match for integers/dates/tuples and a tolerance-band check
(lower <= x <= upper) for decimals.

Conformance
-----------
To match the upstream eval *exactly*, we do NOT re-implement the scoring. We
vendor the MedCalc-Bench-Verified repo at runtime (clone if absent) and import
its own functions verbatim:

  * prompt builders  `zero_shot` / `one_shot` / `direct_answer`   (from run.py)
  * answer parser    `extract_answer(answer, calid)`             (from run.py)
  * scorer           `check_correctness(...)`                    (from evaluate.py)

The ONLY thing we replace is `LLMInference` (upstream calls a local HF pipeline
or the legacy `openai.ChatCompletion`): we swap in an OpenAI-compatible chat
client pointed at either the TRAPI (Azure-OpenAI) proxy or one of our vLLM
servers. We reproduce upstream's post-processing exactly, including the
`re.sub(r"\\s+", " ", answer)` whitespace collapse that `LLMInference.answer`
applies before parsing.

Because run.py imports the heavy `llm_inference` module at import time (torch /
transformers / tiktoken), we stub that module in sys.modules before importing,
so vendoring stays torch-free.

Two providers, one script
-------------------------
  --provider trapi : TRAPI proxy (gpt-5.x). Sends `max_completion_tokens` and an
                     optional `reasoning_effort`; no `temperature`/`max_tokens`.
                     Use NOW while the GPU servers are busy (reference baseline).
  --provider vllm  : an OpenAI-compatible vLLM server hosting our *trained*
                     checkpoint. Sends `temperature` + `max_tokens` and an
                     optional Qwen `enable_thinking` toggle. Use this to score
                     our trained models once a server is free.

Usage (TRAPI reference, now):
    python medcalc_bench_eval.py \
        --provider trapi --model-base http://point.dd.works:18890/v1 \
        --model-name gpt-5.3-chat_2026-03-03 --model-key "$TRAPI_KEY" \
        --prompt one_shot --concurrency 8 --limit 20

Usage (our trained checkpoint via vLLM, later):
    python medcalc_bench_eval.py \
        --provider vllm --model-base http://localhost:8000/v1 \
        --model-name Qwen/Qwen3.6-27B --prompt one_shot \
        --max-tokens 4096 --no-thinking-off
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/nikhilk7153/MedCalc-Bench-Verified"
HF_DATASET = "nsk7153/MedCalc-Bench-Verified"  # mirror of the test CSV


# --------------------------------------------------------------------------- #
# Vendoring the MedCalc-Bench-Verified repo (for verbatim scoring conformance)
# --------------------------------------------------------------------------- #
def ensure_medcalc(src_dir: str):
    """Clone the repo into `src_dir` (if absent) and import its scoring code.

    Returns (medcalc_run_module, check_correctness, one_shot_json, test_csv_path).
    We stub `llm_inference` before importing run.py so no torch is pulled in.
    """
    src = Path(src_dir).expanduser()
    eval_dir = src / "evaluation"
    if not (eval_dir / "run.py").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"[medcalc] cloning {REPO_URL} -> {src}")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(src)],
            check=True,
        )

    # Stub the heavy llm_inference module so `import run` stays torch-free.
    if "llm_inference" not in sys.modules:
        stub = types.ModuleType("llm_inference")

        class _StubLLMInference:  # pragma: no cover - never instantiated
            def __init__(self, *a, **k):
                raise RuntimeError("stubbed LLMInference should not be used")

        stub.LLMInference = _StubLLMInference
        sys.modules["llm_inference"] = stub

    # evaluation/ must be importable as top-level (run.py uses flat imports).
    sys.path.insert(0, str(eval_dir))
    import run as medcalc_run  # noqa: E402  (vendored)
    from evaluate import check_correctness  # noqa: E402  (vendored)

    one_shot_path = eval_dir / "one_shot_finalized_explanation.json"
    with open(one_shot_path) as f:
        one_shot_json = json.load(f)

    # The verified repo ships the test set at datasets/test_data.csv.
    test_csv = src / "datasets" / "test_data.csv"
    if not test_csv.exists():  # tolerate the singular-dir typo in upstream run.py
        alt = src / "dataset" / "test_data.csv"
        test_csv = alt if alt.exists() else test_csv

    return medcalc_run, check_correctness, one_shot_json, str(test_csv)


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat client (TRAPI or vLLM) — replaces LLMInference
# --------------------------------------------------------------------------- #
class ChatModel:
    """Minimal OpenAI-compatible chat client mirroring the request shaping in
    healthbench_professional_eval.make_chat_sampler / eval_sota.py."""

    def __init__(self, *, base_url, api_key, model, provider="trapi",
                 temperature=0.0, max_tokens=4096, reasoning_effort=None,
                 enable_thinking=None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort or None
        self.enable_thinking = enable_thinking
        self._is_trapi = provider == "trapi"

    def answer(self, messages, max_retries=6):
        """Return the model's text with upstream's whitespace collapse applied."""
        import openai as openai_mod

        trial = 0
        while True:
            try:
                if self._is_trapi:
                    kwargs = dict(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=self.max_tokens,
                    )
                    if self.reasoning_effort:
                        kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    kwargs = dict(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    if self.enable_thinking is not None:
                        kwargs["extra_body"] = {
                            "chat_template_kwargs": {
                                "enable_thinking": self.enable_thinking
                            }
                        }
                resp = self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = msg.content
                if not content:  # reasoning models may leave content empty
                    content = getattr(msg, "reasoning_content", None)
                if not content:
                    raise ValueError("empty response")
                # Mirror LLMInference.answer(): collapse all whitespace runs.
                return re.sub(r"\s+", " ", content)
            except openai_mod.BadRequestError as e:
                print(f"[bad-request] {e}")
                return "No response (bad request)."
            except Exception as e:
                if trial >= max_retries:
                    print(f"[give-up] after {trial} retries: {e}")
                    return f"No response (error: {e})."
                backoff = min(2 ** trial, 60)
                print(f"[retry {trial}] {e} — sleeping {backoff}s")
                time.sleep(backoff)
                trial += 1


# --------------------------------------------------------------------------- #
# Prompt construction (mirrors run.py's __main__ dispatch, per prompt_style)
# --------------------------------------------------------------------------- #
def build_messages(medcalc_run, one_shot_json, prompt_style, note, question, calid):
    if prompt_style == "zero_shot":
        system, user = medcalc_run.zero_shot(note, question)
    elif prompt_style == "direct_answer":
        system, user = medcalc_run.direct_answer(note, question)
    elif prompt_style == "one_shot":
        # Replicates the one_shot branch in run.py's __main__ (non-meditron path).
        if str(calid) == "24":
            one_shot_question = ("Based on the patient's dose of Hydrocortisone IV, "
                                 "what is the equivalent dosage in mg of Dexamethasone PO?")
        else:
            one_shot_question = question
        example = one_shot_json[str(calid)]
        system, user = medcalc_run.one_shot(
            note, question, one_shot_question,
            example["Patient Note"],
            {"step_by_step_thinking": example["Response"]["step_by_step_thinking"],
             "answer": example["Response"]["answer"]},
        )
    else:
        raise ValueError(f"unknown prompt_style: {prompt_style}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Per-example evaluation (parse + score with vendored functions)
# --------------------------------------------------------------------------- #
def eval_one(model, medcalc_run, check_correctness, one_shot_json, prompt_style, row):
    note = row["Patient Note"]
    question = row["Question"]
    calid = str(row["Calculator ID"])
    note_id = str(row["Note ID"])

    messages = build_messages(medcalc_run, one_shot_json,
                              prompt_style, note, question, calid)
    raw = model.answer(messages)

    base = {
        "Row Number": int(row["Row Number"]),
        "Calculator Name": row["Calculator Name"],
        "Calculator ID": calid,
        "Category": row["Category"],
        "Note ID": note_id,
        "Question": question,
        "Ground Truth Answer": row["Ground Truth Answer"],
    }
    try:
        answer_value, explanation = medcalc_run.extract_answer(raw, int(calid))
        correctness = check_correctness(
            answer_value, row["Ground Truth Answer"], calid,
            row["Upper Limit"], row["Lower Limit"],
        )
        base.update({
            "LLM Answer": answer_value,
            "LLM Explanation": explanation if prompt_style != "direct_answer" else "N/A",
            "LLM Raw": raw,
            "Result": "Correct" if correctness else "Incorrect",
        })
    except Exception as e:  # parse / eval / scoring failure => Incorrect (upstream behavior)
        base.update({
            "LLM Answer": str(e),
            "LLM Explanation": "N/A",
            "LLM Raw": raw,
            "Result": "Incorrect",
        })
    return base


# --------------------------------------------------------------------------- #
# Aggregation (same per-example correctness as table_stats; correct N)
# --------------------------------------------------------------------------- #
def aggregate(records):
    import numpy as np

    by_cat, by_calid = {}, {}
    for r in records:
        ok = 1 if r["Result"] == "Correct" else 0
        by_cat.setdefault(r["Category"], []).append(ok)
        by_calid.setdefault(str(r["Calculator ID"]), []).append(ok)

    def stats(vals):
        a = np.array(vals, dtype=float)
        m = float(a.mean()) if len(a) else 0.0
        std = float(np.sqrt(m * (1 - m) / len(a))) if len(a) else 0.0
        return {"n": len(a), "average": round(m * 100, 2), "std": round(std, 2)}

    out = {"per_category": {c: stats(v) for c, v in sorted(by_cat.items())},
           "per_calculator": {c: stats(v) for c, v in sorted(by_calid.items())}}
    allv = [ok for v in by_cat.values() for ok in v]
    out["overall"] = stats(allv)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["trapi", "vllm"], default="trapi")
    p.add_argument("--model-base", required=True, help="OpenAI-compatible base URL (…/v1)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-key", default=os.getenv("MODEL_KEY", "EMPTY"))
    p.add_argument("--prompt", choices=["zero_shot", "one_shot", "direct_answer"],
                   default="one_shot",
                   help="MedCalc-Bench prompt style. one_shot = the paper's headline CoT setting.")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0, help="vLLM only")
    p.add_argument("--reasoning-effort", default=None,
                   help="trapi only (gpt-5.x): low/medium/high; omit for default.")
    p.add_argument("--thinking-off", dest="thinking", action="store_false", default=None,
                   help="vLLM Qwen: force enable_thinking=false (default: leave unset).")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="eval only the first N rows (smoke test)")
    p.add_argument("--vendor-dir", default=os.getenv(
        "MEDCALC_VENDOR", str(Path.home() / ".cache" / "MedCalc-Bench-Verified")))
    p.add_argument("--test-csv", default=None, help="override path to test_data.csv")
    p.add_argument("--output-dir", default="./medcalc_eval_out")
    p.add_argument("--tag", default=None, help="label for output filenames")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    import pandas as pd

    medcalc_run, check_correctness, one_shot_json, vendored_csv = ensure_medcalc(args.vendor_dir)
    test_csv = args.test_csv or vendored_csv
    df = pd.read_csv(test_csv)
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()
    print(f"[medcalc] {len(df)} examples from {test_csv} | prompt={args.prompt} "
          f"| provider={args.provider} | model={args.model_name}")

    model = ChatModel(
        base_url=args.model_base, api_key=args.model_key, model=args.model_name,
        provider=args.provider, temperature=args.temperature, max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=(args.thinking if args.provider == "vllm" else None),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.model_name.replace('/', '_')}_{args.prompt}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"medcalc_{tag}_{stamp}.jsonl"
    results_path = out_dir / f"medcalc_{tag}_{stamp}.results.json"

    rows = [df.iloc[i] for i in range(len(df))]
    records = [None] * len(rows)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex, \
            open(jsonl_path, "w") as fout:
        futs = {
            ex.submit(eval_one, model, medcalc_run, check_correctness,
                      one_shot_json, args.prompt, row): i
            for i, row in enumerate(rows)
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            rec = fut.result()
            records[i] = rec
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done += 1
            if done % 25 == 0 or done == len(rows):
                acc = 100 * sum(r["Result"] == "Correct" for r in records if r) / done
                print(f"  {done}/{len(rows)} | running acc={acc:.1f}% | {time.time()-t0:.0f}s")

    summary = aggregate(records)
    summary["_meta"] = {
        "model": args.model_name, "provider": args.provider, "prompt": args.prompt,
        "n": len(rows), "test_csv": test_csv, "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort, "timestamp": stamp,
        "outputs": str(jsonl_path),
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MedCalc-Bench results ===")
    print(f"overall: {summary['overall']['average']}%  (n={summary['overall']['n']})")
    for cat, s in summary["per_category"].items():
        print(f"  {cat:12s} {s['average']:6.2f}%  (n={s['n']})")
    print(f"\noutputs : {jsonl_path}\nresults : {results_path}")

    if args.wandb_project and not args.no_wandb:
        try:
            import wandb

            run = wandb.init(project=args.wandb_project, name=f"medcalc_{tag}",
                             config=summary["_meta"])
            log = {"medcalc/overall": summary["overall"]["average"]}
            for cat, s in summary["per_category"].items():
                log[f"medcalc/cat/{cat}"] = s["average"]
            run.log(log)
            run.finish()
        except Exception as e:
            print(f"[wandb] skipped: {e}")


if __name__ == "__main__":
    main()
