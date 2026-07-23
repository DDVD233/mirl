# MedCalc-Bench-Verified eval

Evaluate a model on [MedCalc-Bench-Verified](https://github.com/nikhilk7153/MedCalc-Bench-Verified)
(1,100 clinical-calculation examples, 55 calculators, 7 categories). We reuse
the benchmark's **own** prompt builders, answer parser, and scorer verbatim, so
numbers are directly comparable to the upstream leaderboard.

## Files
- `medcalc_bench_eval.py` — self-contained runner. Clones the benchmark repo to
  `--vendor-dir` on first use, stubs its heavy `llm_inference` module, and
  imports `zero_shot`/`one_shot`/`direct_answer`/`extract_answer` (from `run.py`)
  and `check_correctness` (from `evaluate.py`) unchanged. The only substitution
  is the model call: an OpenAI-compatible client (TRAPI or vLLM). It also
  reproduces `LLMInference.answer`'s `re.sub(r"\s+"," ")` whitespace collapse.
- `run_medcalc_eval_trapi.sh` — score a TRAPI (gpt-5.x) model. **No GPU needed** —
  use while the training servers are busy. Reference/baseline numbers.
- `run_medcalc_eval_vllm.sh` — score one of **our trained checkpoints** served by
  a vLLM OpenAI endpoint. Run once a GPU server is free.

## Scoring (upstream, unchanged)
- **date** (calid 13, 68): exact `M/D/Y` match.
- **integer tuple** (calid 69): exact `(weeks, days)` match.
- **integer** (calid 4, 15–18, …): `round(eval(answer))` exact match.
- **decimal** (calid 2, 3, 5–11, …): tolerance band `lower <= x <= upper`.
- Any parse/eval failure ⇒ `Incorrect` (upstream behavior).
- Metric: accuracy per category + overall. (We compute the binomial std with the
  correct per-N denominator instead of upstream's hard-coded 1047; the mean —
  the reported number — is identical.)

## TRAPI (now)
```bash
# smoke test
MODEL=gpt-5.3-chat_2026-03-03 LIMIT=20 bash run_medcalc_eval_trapi.sh
# full 1,100
MODEL=gpt-5.3-chat_2026-03-03 bash run_medcalc_eval_trapi.sh
```
Key is read from `/scratch/sheng/self_evolving/.trapi_key`; base
`http://point.dd.works:18890/v1`. `EFFORT=low` adds `reasoning_effort` (gpt-chat
models reject it, so it is unset by default).

## vLLM (our trained checkpoints, later)
```bash
# 1) serve the checkpoint (example)
vllm serve /scratch/sheng/.../global_step_120/hf --port 8000 \
    --served-model-name my_ckpt  # + your usual parser/dtype flags
# 2) score it
MODEL=my_ckpt BASE=http://localhost:8000/v1 bash run_medcalc_eval_vllm.sh
# reasoning model, longer budget, thinking left ON:
MODEL=my_ckpt BASE=http://localhost:8000/v1 MAXTOK=8192 bash run_medcalc_eval_vllm.sh
# force non-thinking:
MODEL=my_ckpt BASE=http://localhost:8000/v1 THINKING_OFF=1 bash run_medcalc_eval_vllm.sh
```

## Notes
- `--prompt` ∈ {`one_shot` (default, the paper's headline CoT setting),
  `zero_shot`, `direct_answer`}.
- Reasoning models need enough `--max-tokens` to emit the closing JSON after
  thinking, or `extract_answer` finds nothing → `Incorrect`. Bump `MAXTOK` if
  accuracy looks suspiciously low.
- Outputs: `<out>/medcalc_<tag>_<ts>.jsonl` (per-example, incl. raw response) and
  `…​.results.json` (aggregates + meta). Default out dir
  `/scratch/sheng/self_evolving/logs/medcalc_eval`.
