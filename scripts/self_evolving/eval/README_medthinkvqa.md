# MedThinkVQA final-answer eval (gpt-5.1 via TRAPI)

Replicates the **final-answer accuracy** of [MedThinkVQA](https://huggingface.co/datasets/bio-nlp-umass/MedThinkVQA)
(bio-nlp-umass, ICLR 2026) — a 720-case, multi-image radiology **differential-diagnosis**
benchmark. Each case = short clinical history + N case images + 5 candidate
diagnoses (A–E); the model picks the single best diagnosis letter. Accuracy =
correct letters / 720.

## Faithfulness caveat (read this)

The public repo <https://github.com/benluwang/MedThinkVQA> ships only preprocessing
utilities plus an OpenAI Responses-API wrapper (`model/gpt.py`). **The actual
final-answer eval harness — the inference prompt and the answer-letter parser — is
NOT released** (the shipped code references an `eval_rec` with
`parsed_answer_letter`/`is_correct`/`gt_letter` produced by held-out code). So this
is a faithful *reconstruction*, not their verbatim script:

- **Request construction mirrors `model/gpt.py::APIModel`**: OpenAI Responses API,
  system+user roles, images as `input_image` base64 data URLs, explicit
  `reasoning.effort`, and no sampling params under reasoning effort (gpt-5.1/5.2
  reject `temperature`/`top_p` unless effort == `none`).
- **Model input** = `CLINICAL_HISTORY` + all case images + the 5 options. The
  `IMAGING_FINDINGS` free text is withheld (answer-adjacent; the benchmark filters
  text-solvable cases, so the intended task is image-grounded).
- **Answer parsing**: model is told to end with `Answer: <LETTER>`; parser takes the
  last `Answer: X`, else the last standalone A–E token, else verbatim option-text match.

## Reference numbers (official `docs/benchmark.json`)

No `gpt-5.1` entry exists in the official leaderboard. Nearest references (all 720-case):

| Model | Mode / effort | Acc |
|---|---|---|
| claude-4.6-opus | – | 0.572 |
| gemini-3-pro | – | 0.553 |
| gpt-5.2 | Thinking xhigh | 0.549 |
| gpt-5.2 | Thinking high | 0.544 |
| gpt-5.2 | Thinking medium | 0.533 |
| gpt-5.2 | Thinking low | 0.529 |
| gpt-5.2 | Non-thinking | 0.496 |
| qwen3.5-397b | Thinking | 0.522 |

Random (5-way) = 0.20; majority-class (A, 182/720) = 0.253. A gpt-5.1 medium-effort
result near **~0.50** is consistent with gpt-5.2 sitting at 0.53 (medium) / 0.50 (non-thinking).

## How to run

```bash
# data (one-time): 5.8 GB images.zip + test.jsonl, unzipped to $DATA_DIR/images/
hf download bio-nlp-umass/MedThinkVQA --repo-type dataset \
    --include test.jsonl images.zip --local-dir /scratch/dvd/medthinkvqa
unzip -q /scratch/dvd/medthinkvqa/images.zip -d /scratch/dvd/medthinkvqa

# full eval (gpt-5.1 via TRAPI proxy, medium effort)
MODEL=gpt-5.1_2025-11-13 EFFORT=medium CONC=12 bash run_medthinkvqa_trapi.sh

# smoke test / ablations
LIMIT=5 bash run_medthinkvqa_trapi.sh
python medthinkvqa_eval.py --no-images ...        # text-only leakage ablation
```

Env knobs (see `run_medthinkvqa_trapi.sh`): `MODEL`, `EFFORT` (none|low|medium|high|xhigh),
`CONC`, `MAXTOK`, `LIMIT`, `DATA_DIR`, `TRAPI_BASE`, `TRAPI_KEY`, `PY`.

## Outputs (`results_medthinkvqa/`)

- `<tag>.jsonl` — per case: `title`, `gt`, `pred`, `correct`, `n_images`, tokens, `raw`.
- `<tag>.summary.json` — accuracy, parse/error counts, pred/gt letter distributions, config.

## Files

- `medthinkvqa_eval.py` — the reconstructed eval (async, OpenAI-compatible/TRAPI).
- `run_medthinkvqa_trapi.sh` — TRAPI runner wrapper.
