# Stage 1 MIMIC-IV method baselines

Backbones: frozen Qwen3.6-27B and Qwen3.5-9B. The runner also supports
OpenAI-compatible GPT endpoints; a GPT extension must pair every method with a
fresh direct run using the same deployment and reasoning budget.

## Matrix and references

| Method | Implementation | Reference |
| --- | --- | --- |
| Direct | Original multimodal prompt, one generation | Existing paper baseline |
| Medical RAG | One search of the case text, top 8 passages, answer | [MedRAG / MIRAGE, 2024](https://arxiv.org/abs/2402.13178) |
| RAG-Fusion | Case plus 3 generated queries, reciprocal-rank fusion (constant 60), top 8, answer | [RAG-Fusion, 2024](https://arxiv.org/abs/2402.03367) |
| i-MedRAG | 3 rounds of 2 follow-up questions, each answered by RAG; accumulated QA guides the next round and final answer | [i-MedRAG, 2024 preprint / PSB 2025](https://arxiv.org/abs/2408.00727) |
| Self-consistency | 5 independent samples, normalized diagnosis plurality, earliest sample breaks ties | [Self-consistency, ICLR 2023](https://arxiv.org/abs/2203.11171) |
| Self-refinement | Draft, evidence-based self-critique, revision | [Self-Refine, 2023](https://arxiv.org/abs/2303.17651) |
| Chain-of-verification | Draft, 3 verification questions, independent answers without the draft, revision | [CoVe, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.212/) |
| Static-pool RL | Existing 27B SFT initialization + RL on frozen 15,000-task pool; best full validation | Archived result, 0.4412724307 |

These are explicitly **shared-corpus/backbone adaptations**. MedRAG's original
retriever ensemble and full MedCorp are replaced by the project's dense index.
i-MedRAG's follow-up QA procedure follows the original algorithm; its prompts and
answer format are adapted to open-ended multimodal diagnosis. The self-consistency
vote does not merge medical synonyms, a limitation of exact normalized voting.
Self-refinement and self-consistency are older standard controls; the RAG arms
cover 2024 onward. Do not call an untrained prompt loop "Self-RAG": the original
method learns reflection tokens. RAFT also requires training. Neither is included
under the frozen-backbone methods.

## Fixed protocol

- Full 2,452-case `mimiciv_rare/test.jsonl`, unique admission IDs, original image
  inputs, 65,536 pixels/image, and original 9,000-character clinical text cap.
  Unexpected missing/unreadable images fail the case. Two audited corrupt JPEGs
  are omitted as in the original `eval_sota.py`, with their SHA-256 hashes and
  omission reasons recorded per case. All 2,452 cases remain in the denominator.
- Solver and all planning/critique/follow-up roles use the same frozen backbone.
  Qwen thinking is disabled as in the existing baseline sweep. Final/draft/sample
  budget is 8,192 tokens; auxiliary answers use 1,024. Qwen temperature is zero
  except self-consistency samples at 0.7 with seeds 42 onward. GPT omits temperature
  and seed for compatibility and records its configured reasoning effort.
  Qwen planning calls use constrained JSON arrays of the requested length.
- Raw cosine retrieval, overfetch 32, reject duplicate/short passages, retain 8,
  cap each at 1,200 characters. RAG-Fusion's final context uses the same 8-passage
  budget. i-MedRAG uses that budget per follow-up question, with its extra calls
  counted. These methods are not claimed to be equal in inference compute.
- Only `medrag_textbook`, `medrag_pubmed`, `medrag_wiki`, and `pubmedqa` text rows
  from `medical_knowledge_v2`; no generated-history index, MIMIC cases, or added
  stage-2 curated source families. This is a common source-family restriction,
  not a claim to reconstruct an immutable historical stage-1 corpus snapshot.
  Row count/filter and returned passage text, IDs, and hashes are archived.
- Generation receives only `prompt` and `images`. Labels and metadata are attached
  after generation, never used for queries, critique, voting, or candidate selection.
- Judge: `gpt-chat-latest_2026-05-28`, 2,048 completion tokens, no reasoning override,
  exact stage-1 judge prompt and extraction code. Use the remote stage-1 checkout
  `/scratch/sheng/self_evolving/verl` at `01c43aa6`, including its existing dirty
  reward file. Never substitute the stage-2 reward implementation.
- Manifest hashes runner, input data, evaluation helper, reward file, and settings.
  Resume rejects changed configuration. Generation and grading have separate
  per-case outputs; failures remain missing and retryable, never incorrect scores.
  Summary files exist only after full generation and grading.

Each generation row includes every successful model response, token usage, finish
reason, query, retrieved context, selection decision, and timing. API retry token
cost is unavailable when the provider fails before returning usage. Report that
limitation when comparing inference budgets. Long answers can still hit their
generation cap; finish reasons make truncation auditable.

## Running

Deploy these scripts to `$S/stage1_method_baselines/code/` on the shared NFS.
Launch the 9B server with `serve_mimic_9b.sh` only on an idle existing node. The
script refuses occupied GPUs and never stops existing processes. Use one tmux
session with separate server/evaluation windows. Access is SSH through the
existing FRP ports; no Kubernetes operations are required.

```bash
# 9B: existing pod reached via SSH port 2335.
bash "$S/stage1_method_baselines/code/serve_mimic_9b.sh"
TRAPI_API_KEY=... bash "$S/stage1_method_baselines/code/run_mimic_methods.sh"

# 27B: reuse the frozen service on the existing pod at SSH port 2333.
MODEL=Qwen/Qwen3.6-27B TAG=qwen36_27b BASE_URL=http://127.0.0.1:8188/v1 \
  CONCURRENCY=8 TRAPI_API_KEY=... \
  bash "$S/stage1_method_baselines/code/run_mimic_methods.sh"
```

Keep credentials in the process environment, never in tracked launch scripts or
manifests. Repeating the same launcher resumes successful cases. Use a fresh output
directory after changing code or protocol. A missing judge credential permits
generation but exits explicitly before grading.

Validation:

```bash
uv run --no-sync python -m unittest discover -s scripts/self_evolving/eval -p test_mimic_method_baselines.py
```

`smoke_mimic_methods.sh` exercises all methods and the fixed judge on one synthetic
fixture. This is a service/implementation test, not a subsample of the evaluation
set and not a reported accuracy result. Launchers read the existing `$S/.trapi_key`
when the credential is not already in the environment.

## Paper integration

The existing static-pool row is grounded in
`paper_data/stage1/fig1_scaling/regraded/staticpool/qwen36_27b_staticpool_aicr__step110.json`.
It is selected by maximum full-set fixed-judge accuracy across the archived pool
checkpoints, consistent with the paper's checkpoint selection. No new training is
needed. The manuscript reports the best checkpoint without exposing step numbers.
New inference rows must be added only from complete `*.summary.json` artifacts;
do not interpret an in-progress run as a result.

After syncing the two backbone directories and an exact copy of the remote
`mimiciv_rare/test.jsonl` into `paper_data/stage1/method_baselines/`:

```bash
uv run --no-sync python scripts/self_evolving/eval/compile_mimic_methods.py \
  --root paper_data/stage1/method_baselines \
  --output paper_data/stage1/method_baselines/results.json
```

The compiler checks all admission IDs, grades, configuration hashes, and shared
protocol fields before emitting `results.tex` and a token/call ledger. The ledger
also records total retrieval calls, empty contexts, and deeper-candidate retries.
Depth-audit coverage is explicit because original successful traces predate that
instrumentation; their returned documents still establish empty-context counts. Add
`--allow-partial` for monitoring JSON without a paper table. Incorporate the
completed table into the manuscript and describe the adaptations above when
reporting the results.

### Incremental main-table updates

`watch_mimic_main_table.py` runs locally in tmux window `main:mimic-table`,
polling existing SSH nodes every five minutes by default. It validates completed runs on the
remote machine and downloads a compact ledger. This avoids copying all generation
traces locally. Each completed method is inserted into the marked block in
`paper/tables/main_results.tex`, including all eight chapter scores. Fresh direct
runs are labeled as matched references; archived base-model rows remain visible.
Column-best bolding is recalculated, and the paper PDF is rebuilt after changes.

The watcher rejects missing or changed previously published results, writes a
backup before each table edit, retries connection/build failures, and exits when
all 14 conditions are published. It does not commit or push the paper.

```bash
uv run --no-sync python scripts/self_evolving/eval/watch_mimic_main_table.py --once
```

The one-shot command is for manual use when the watcher is not running. A lock
prevents simultaneous collectors. Current state and build/collector logs are in
`paper_data/stage1/method_baselines/{watcher_status.json,watcher.log,latexmk.log}`.

### Recovery revision (2026-09-07)

Full-set execution exposed two corrupt JPEGs and malformed unconstrained query
arrays that the synthetic test did not cover. The original run stopped short of
full coverage and produced no published scores. The active runs and collector now
use `/scratch/sheng/self_evolving/stage1_method_baselines_v2/`; the original run
directory remains intact for audit.

`recover_mimic_methods_v2.sh` copies only successful direct, single-pass medical
RAG, self-consistency, and self-refinement generations whose behavior is unchanged.
`migrate_mimic_methods_v2.py` checks configuration parity except the implementation
hash, records each copied row's original config and record hash in `reused_from`,
and leaves the original files unchanged. Only the two failed image cases need new
answers for those completed generation sets. RAG-Fusion, i-MedRAG, and CoVe restart
from scratch because their planning now uses constrained JSON.

Both backbones use the same revised implementation and image policy. Judge
concurrency is 32, independently of generation concurrency. Grading still starts
only after all 2,452 answers exist, and no failed API request counts as incorrect.

### Failure completion repair (2026-09-08 UTC)

The first failed 9B i-MedRAG query returned 32 identical eight-character Wikipedia
headings (`Symptoms`). The embedding vector was valid and normalized; all 32 hits
were rejected by the 80-character passage minimum. At depth 128, 64 passages
survived the length filter. The active repair searches deeper only after an empty
post-filter result, retaining the same corpus and final eight-passage budget.
Genuinely empty valid searches yield an explicit no-evidence context rather than
an infrastructure failure. Network and API errors still raise and retry.

The remaining malformed plans were generation-budget truncations, including a
model reasoning repeatedly about diagnosis codes inside a JSON string. Invalid
plans now retry with a 240-character-per-question JSON schema bound, concise
planning instructions, and at most 2,048 tokens. Successful first attempts are
unchanged. Both original failed cases were reproduced, diagnosed, and completed
successfully with these repairs before resuming evaluation.

The archived repair implementation lives in `code/failure_repair/` under the v2
remote root. `--resume-failure-repair` accepts only an implementation-hash change;
all experiment settings must match the original manifest. The manifest retains
its initial experiment identity and registers actual implementations in
`implementation_history`; new rows identify their implementation explicitly.
The collector rejects unregistered implementations. Existing scores and successful
case records are retained. Failed attempts now retain their full method trace.

`finish_mimic_methods.sh` runs the remaining methods to completion, retries
transient failures automatically, and grades only complete generation sets.
Remote tmux windows are `main:mimic9b-finish` and `main:mimic27b-finish`;
logs are `logs/finish_9b.log` and `logs/finish_27b.log`. The table collector continues
to publish only validated full-set results.

### Completed matrix (2026-09-08 UTC)

All 14 conditions are complete and published in the main table. Both finish
supervisors exited successfully; the collector exited after its 14/14 update.
Every row contains 2,452 unique generated and graded admissions, for 34,328
case-method evaluations. All six RAG arms together contain 53,944 retrieval calls
and zero empty contexts. The 9B i-MedRAG repair recovered 152 searches by deepening
the candidate pool. Scores below are percentages, not partial-run estimates.

| Method | Qwen3.5-9B | Qwen3.6-27B |
| --- | ---: | ---: |
| Direct (matched) | 26.18 | 33.20 |
| Medical RAG | 24.51 | 32.91 |
| RAG-Fusion | 23.78 | 31.81 |
| i-MedRAG | 22.63 | 30.14 |
| Self-consistency | 27.20 | 34.62 |
| Self-refinement | 22.06 | 27.57 |
| Chain-of-verification | 23.00 | 33.48 |

The existing 27B static-pool training control is also included, at 44.13%.
Exact scores, chapter results, implementation provenance, token counts, and
retrieval audit counts are in `paper_data/stage1/method_baselines/status.json`.
Full per-case traces remain in the remote v2 archive; the original runs are
preserved. The paper's inference-baseline appendix documents the adaptations and
failure handling. No GPT extension or new training was launched for this matrix.
