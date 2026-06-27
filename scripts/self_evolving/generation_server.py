"""
Self-evolving question generation server.

A long-lived FastAPI process that owns the entire question-generation
pipeline (QueryProposer -> Milvus retrieval -> QuestionGenerator ->
QuestionValidator) and continuously refills a pool of accepted training
samples. The trainer's dataset is a thin HTTP client over this server.

Endpoints
---------
GET  /healthz   liveness check
GET  /stats     pool size, totals, recent accuracy
GET  /sample    pop one entry from the pool (blocks up to 600s if empty)
POST /report    {question_id, accuracy} feedback for difficulty calibration
POST /replay    re-push previously-accepted entries from the log into the pool

Why this exists
---------------
The previous in-process pipeline blocked the trainer's main loop on every
batch, leaving vLLM and the actor GPUs idle while ~10 sequential LLM
calls per target ran. Externalizing it lets:
  - N workers fan out across vLLM concurrently and keep the chat server
    saturated independent of the trainer's step cadence,
  - the pool absorb generation latency (trainer never waits if the pool
    is non-empty),
  - per-question accuracy be reported back via a single HTTP POST
    instead of being inferred from a sliding rm_scores window.

Run with `start_generation_server.sh`.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import random
import re
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("gen_server")


# ======================================================================
# AGENT SYSTEM PROMPTS (verbatim from self_evolving_dataset.py)
# ======================================================================

QUERY_PROPOSER_SYSTEM_PROMPT = """\
You are a medical information retrieval expert. Given a training question (and optionally \
a list of previously-proposed similar questions with the solver's running accuracy on \
each), propose 10 diverse search queries for retrieving medical knowledge from a \
multimodal database (PubMedQA abstracts, MIRAGE MCQs, MedRAG textbooks, PubMed, Wikipedia, \
PMC-VQA, CLIMB clinical QA across chest X-ray, derm, CT, ECG, fundus, MRI, mammography, \
ultrasound, pathology).

If a list of previously-proposed similar questions is provided, FIRST briefly assess \
(internally, in your reasoning) what the solver appears to already do well (high accuracy) \
and where it struggles (low accuracy or untested), then bias your 10 queries toward the \
gaps. DO NOT generate queries whose answers would duplicate those past questions.

Output EXACTLY 10 queries, one per angle below (IN ORDER). Each query is a complete \
sentence (not keywords), specific enough to retrieve focused results, and should retrieve \
DIFFERENT content — avoid near-paraphrases.

1. MECHANISM / pathophysiology (molecular, cellular, systems level)
2. DIAGNOSTIC CRITERIA or workup (specific tests, thresholds, scoring systems)
3. COMPARATIVE effectiveness (treatment A vs B, test A vs B with outcome metric)
4. ADVERSE EFFECTS / complications / contraindications
5. PROGNOSIS / outcome / natural history (specific numbers, survival, risk factors)
6. ATYPICAL PRESENTATION or edge case (rare variant, unusual demographic)
7. DIFFERENTIAL DIAGNOSIS (distinguishing from 1–2 named mimics)
8. IMAGING / VISUAL FINDINGS (modality-specific features, if applicable — else another \
   angle not yet covered)
9. EPIDEMIOLOGY or risk-factor association (quantitative if possible)
10. RELATED CONDITION or downstream effect (comorbidity, systemic link, long-term sequela)

Keep any internal reasoning UNDER 500 WORDS, then output ONLY a JSON array of 10 strings \
in the above order. No markdown, no explanation.
["query 1", "query 2", ..., "query 10"]"""


QUESTION_GENERATOR_SYSTEM_PROMPT = """\
You are a medical educator creating training questions for a medical AI. You are given: \
(1) a reference training question, (2) several relevant passages from a medical \
knowledge base, (3) the solver's recent accuracy, (4) the REQUIRED format for this \
question.

SYNTHESIZE a NEW question that AGGREGATES information across the retrieved passages \
(not a copy of any one source). The question MUST:

- NOT be a direct copy or paraphrase of any single passage or the reference question.
- Combine facts, conditions, or mechanisms across MULTIPLE passages when possible (e.g. \
  complex clinical scenarios, rare corner cases mentioned by multiple sources, \
  differentials where passages disagree partially, treatment tradeoffs weighing \
  different sources).
- Hit one of these depths: complex clinical scenario, rare corner case, differential \
  diagnosis where multiple dx fit partially, treatment tradeoff, atypical presentation.

REQUIRED FORMAT: {required_format}

DIFFICULTY CALIBRATION:
- Solver's recent accuracy: {accuracy:.0%} over {accuracy_count} questions.
- Target ~50% accuracy. If accuracy is high, add more nuance / closer distractors. If \
  low, sharpen phrasing but keep the inferential step.

ANSWER RULES:
- Answer must be verifiable FROM THE PASSAGE + standard textbook facts.
- For MCQ: 4 plausible options. Distractors must be defensible misinterpretations (e.g. \
  adjacent condition, wrong phase of treatment, right concept but wrong threshold) — \
  NOT obvious nonsense.
- For free response: answer is a specific phrase (1-15 words, e.g. a diagnosis, drug \
  name, mechanism, threshold value).
- The correct answer must be UNAMBIGUOUS — exactly one option is defensible.

Keep any internal reasoning UNDER 500 WORDS, then output ONLY a JSON object. No markdown, \
no explanation.

For MCQ (required_format="mcq"):
{{"format": "mcq", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}

For free response (required_format="free"):
{{"format": "free", "question": "...", "answer": "short expected answer"}}"""


QUESTION_GENERATOR_MM_SYSTEM_PROMPT = """\
You are a medical educator creating MULTIMODAL training questions for a medical vision-language AI. \
You are given several retrieved clinical media items, each labeled [MEDIA k] with its clinical \
modality, the original question it came from, and its answer/label. The actual image(s) (or sampled \
video frames) are attached in order.

Create ONE NEW question that USES one or more of these media items and is answerable from them. Be \
DIVERSE across calls — pick the most fitting of these styles (vary it):
  - reuse / rephrase the original question for a single media item;
  - LOCALIZE a finding ("in <<k>>, which region / lobe / quadrant shows the abnormality?");
  - relate to KNOWLEDGE implied by the label (mechanism, next diagnostic step, complication);
  - go BROADER (the parent category of the label) or FINER (a more specific subtype);
  - COMPARE two media items ("how does the finding in <<1>> differ from <<2>>?").

Reference each media item you use with the token <<k>> (e.g. <<1>>, <<2>>) placed exactly where the \
reader must look at it; you may reference an item more than once. Every <<k>> must be a valid index, \
and you MUST reference at least one item.

REQUIRED FORMAT: {required_format}
DIFFICULTY: solver recent accuracy {accuracy:.0%} over {accuracy_count} items; target ~50%.

ANSWER RULES:
  - The answer must be unambiguously determinable from the referenced media (+ standard medical knowledge).
  - MCQ: exactly 4 options; distractors must be defensible (adjacent finding, wrong region, wrong subtype) \
    — never obvious nonsense; exactly one option is correct.
  - Free: a short specific phrase (1-8 words: a finding, region, diagnosis, threshold, or mechanism).

Keep internal reasoning UNDER 400 WORDS, then output ONLY a JSON object. No markdown, no explanation.
For MCQ: {{"format":"mcq","question":"... <<1>> ...","options":{{"A":"..","B":"..","C":"..","D":".."}},"answer":"A"}}
For free: {{"format":"free","question":"... <<1>> ...","answer":"short answer"}}"""


QUESTION_VALIDATOR_SYSTEM_PROMPT = """\
You are a medical fact-checker. You are given:
(1) A candidate training question + its proposed answer (+ options if MCQ).
(2) Retrieved passages from a medical knowledge database (re-queried using the \
candidate question).

Decide whether the question's proposed answer CONTRADICTS the retrieved knowledge.

BE LENIENT — only reject obvious contradictions:
- If the retrieved knowledge directly and unambiguously STATES something that makes \
  the proposed answer WRONG → "contradict"
- If the retrieved knowledge is silent, tangential, or only partially relevant → "ok"
- If the question is about something NOT in the retrieved passages (out-of-knowledge) \
  → "ok" (we accept new knowledge)
- If the question is well-formed but the answer seems questionable without direct \
  contradiction from the passages → "ok"
- If the question is poorly formed / ungrammatical / incoherent → "contradict"

Keep any internal reasoning UNDER 500 WORDS, then output ONLY a JSON object with a \
one-sentence reason. No markdown, no explanation.
{{"verdict": "ok" or "contradict", "reason": "..."}}"""


SOLVER_SYSTEM_PROMPT_MCQ = (
    "You are a medical expert. Read the question carefully and choose the best answer. "
    "Think through the question briefly, then give your final answer. Keep your reasoning "
    "under 500 words. Commit to your reasoning — do not waver, backtrack, or use hedging "
    "phrases like \"wait\", \"actually\", \"on second thought\", or \"hmm\". Give your "
    "best answer directly. The final answer MUST BE a single letter (A, B, C, or D) "
    "wrapped in \\boxed{}. The boxed answer is REQUIRED — do not omit it. "
    "Example: \\boxed{C}"
)

SOLVER_SYSTEM_PROMPT_FREE = (
    "You are a medical expert. Answer the question with a short specific phrase. "
    "Think through the question briefly, then give your final answer. Keep your reasoning "
    "under 500 words. Commit to your reasoning — do not waver, backtrack, or use hedging "
    "phrases like \"wait\", \"actually\", \"on second thought\", or \"hmm\". Give your "
    "best answer directly. The final answer MUST BE a short phrase (1-15 words) wrapped "
    "in \\boxed{}. The boxed answer is REQUIRED — do not omit it. "
    "Example: \\boxed{acute pancreatitis}"
)


# Teacher prompt used in SFT mode to elicit a VISIBLE reasoning trace for
# distillation. Reasoning models (e.g. TRAPI gpt-5.x) keep their chain-of-thought
# in a hidden channel that the API does not return — so we must explicitly demand
# the reasoning as visible prose in the content, otherwise we only get the final
# boxed answer (and an empty <think> block). The student is trained to imitate
# THIS trace under its own (SOLVER) system prompt; this prompt is teacher-only.
SFT_TEACHER_SYSTEM_PROMPT = (
    "You are a medical expert solving a question in order to TEACH a student. "
    "You MUST write out your full step-by-step clinical reasoning as visible prose: "
    "interpret the key findings, weigh the plausible differentials, and justify why the "
    "correct answer is right and the others are wrong. Write several sentences of reasoning "
    "— do NOT respond with only the final answer. "
    "You may be given reference medical knowledge (the same passages this question was "
    "synthesized from, plus the original training context) as background to ground yourself. "
    "USE it to reason correctly and reach the right answer, but write SELF-CONTAINED clinical "
    "reasoning — NEVER refer to \"the passage\", \"the reference\", \"the context\", or "
    "\"the document\", because the student you are teaching will NOT see this material and "
    "must learn reasoning it can reproduce from the question alone. "
    "Commit to your reasoning; do not hedge or "
    "backtrack. After the reasoning, on its own line, output the final answer wrapped in "
    "\\boxed{} — a single letter (A, B, C, or D) for a multiple-choice question, otherwise a "
    "short specific phrase (1-15 words). The boxed answer is REQUIRED. "
    "Example ending: \\boxed{C}"
)


# ======================================================================
# Sampling pool
# ======================================================================
class _RandomQueue(asyncio.Queue):
    """Bounded async queue that pops a uniform-RANDOM element on get() (not FIFO).

    Subclasses asyncio.Queue the same way the stdlib's LifoQueue / PriorityQueue
    do — overriding only _init/_put/_get — so all of Queue's backpressure and
    (importantly) cancellation-safe getter/putter handling is reused unchanged.
    This matters because /sample wraps get() in asyncio.wait_for(..., 30s).

    Why not FIFO: the workers emit entries in bursts *by mode*. ``direct`` seeds
    (long, ~4.6k-char clinical cases) are cloned instantly and flood in, while
    ``gen_train``/``gen_test`` questions (short, ~90-char) trickle in over
    50-70 s LLM calls. A FIFO pool therefore serves the trainer long contiguous
    runs of one mode. The trainer consumes this stream in order (data.shuffle
    =False -> SequentialSampler; and shuffle=True is a no-op for the
    fetch-on-first-touch SelfEvolvingDataset), so those runs alias against the
    fixed batch size into a period-2 oscillation: consecutive steps train on
    wildly different prompt-length / difficulty populations and every logged
    metric zig-zags while val flatlines.

    Drawing a uniform-random member from the bounded window (whose composition
    the deficit scheduler keeps near the target mix) makes every fetched batch a
    representative mix, so prompt-length means stay flat across steps.
    """

    def _init(self, maxsize: int) -> None:
        self._queue: list = []

    def _put(self, item) -> None:
        self._queue.append(item)

    def _get(self):
        # O(1) unordered removal: swap the chosen element to the end, then pop.
        # Order within the list is irrelevant since we always draw at random.
        i = random.randrange(len(self._queue))
        self._queue[i], self._queue[-1] = self._queue[-1], self._queue[i]
        return self._queue.pop()


# ======================================================================
# Server state
# ======================================================================
class ServerState:
    def __init__(self, args, loop: asyncio.AbstractEventLoop):
        self.args = args
        self.loop = loop
        self.train_seeds = self._load_seeds(args.seeds_path)
        self.test_seeds: list[dict] = []
        if args.test_seeds_path:
            self.test_seeds = self._load_seeds(args.test_seeds_path)
            for s in self.test_seeds:
                # Strip the label so the generator can never see it.
                s.pop("reward_model", None)
        # CLIMB multimodal seeds (real train image/video rows in verl shape with
        # "climb://" media handles). Drive the gen_mm mode and direct multimodal
        # inserts; labels are KEPT (used as the generation answer + reward GT).
        self.climb_seeds: list[dict] = []
        if getattr(args, "climb_seeds_path", ""):
            self.climb_seeds = self._load_seeds(args.climb_seeds_path)
        # Kept for /replay logging compatibility and any consumer that wants a
        # flat seed list — workers no longer iterate this in order.
        self.seeds = list(self.train_seeds) + list(self.test_seeds)
        logger.info(
            f"seeds: {len(self.seeds)} total "
            f"({len(self.train_seeds)} train, {len(self.test_seeds)} test_masked, "
            f"{len(self.climb_seeds)} climb_mm)"
        )

        # Output-mix targets and running counts. Workers pick the most-deficit
        # mode each iteration to drive the pool toward these proportions. If
        # there are no test_seeds, the gen_test target is folded into gen_train;
        # likewise gen_mm folds into gen_train when no climb seeds are loaded.
        self.mix_targets = {
            "direct": float(args.direct_target),
            "gen_train": float(args.gen_train_target),
            "gen_test": float(args.gen_test_target),
            "gen_mm": float(getattr(args, "gen_mm_target", 0.0)),
        }
        if not self.test_seeds:
            self.mix_targets["gen_train"] += self.mix_targets["gen_test"]
            self.mix_targets["gen_test"] = 0.0
        if not self.climb_seeds:
            self.mix_targets["gen_train"] += self.mix_targets["gen_mm"]
            self.mix_targets["gen_mm"] = 0.0
        # Renormalize (in case the user passed values that don't sum to 1).
        total = sum(self.mix_targets.values()) or 1.0
        for k in self.mix_targets:
            self.mix_targets[k] /= total
        self.mix_counts = {"direct": 0, "gen_train": 0, "gen_test": 0, "gen_mm": 0}

        # Random-draw (not FIFO) so each fetched batch is a representative mix
        # of the bursty per-mode output instead of a contiguous run of one mode
        # (see _RandomQueue for why FIFO caused a period-2 training oscillation).
        self.pool: asyncio.Queue = _RandomQueue(maxsize=args.max_pool_size)
        # Unbounded buffer drained by /sample BEFORE the regular pool. Used by
        # /replay so we don't drop entries when the pool is full; the trainer
        # pulls these first, then the pool's freshly-generated stream takes
        # over.
        self.replay_buffer: deque = deque()
        # Bounded ring of every entry that was ever pushed into the pool.
        # When the pool is empty and workers can't produce fast enough (e.g.
        # the chat server is saturated), /sample serves a random entry from
        # here instead of 503-ing the trainer.
        self.history: deque = deque(maxlen=args.history_size)
        self.target_idx = 0
        self.cycle = 0
        self.question_counter = 0
        self.format_counter = 0

        self.accuracy_history: deque = deque(maxlen=args.accuracy_window)
        self.accuracy_by_id: dict[str, float] = {}
        # Per-question performance counters mirrored into the Milvus history
        # collection. Keep the in-memory copy authoritative so we don't lose a
        # /report racing an in-flight insert; Milvus is the cross-restart store.
        self.history_counters: dict[str, dict] = {}
        # Lock so concurrent /report calls for the same qid don't double-count
        # before the upsert lands.
        self.history_counter_lock = asyncio.Lock()
        self.stats = {
            "total_queries": 0,
            "total_generated": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "served": 0,
            "reports": 0,
            "direct_inserted": 0,
            "served_from_history": 0,
            "served_from_seeds": 0,
            "sft_traces_skipped": 0,
            "served_incomplete_skipped": 0,
            "served_missing_image_skipped": 0,
            "served_repaired_inline": 0,
            "started_at": datetime.now().isoformat(),
        }

        os.makedirs(args.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.accepted_log = os.path.join(args.log_dir, f"server_accepted_{ts}.jsonl")
        self.rejected_log = os.path.join(args.log_dir, f"server_rejected_{ts}.jsonl")
        self.report_log = os.path.join(args.log_dir, f"server_reports_{ts}.jsonl")
        self.log_lock = asyncio.Lock()

        # threading.local for per-thread Milvus clients (sync calls go through
        # asyncio.to_thread, which uses a default ThreadPoolExecutor).
        import threading
        self._milvus_local = threading.local()

        # shared async http client
        self.http_client: Optional[httpx.AsyncClient] = None

        # per-step timing windows (rolling, last 512 samples per step)
        self.timings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=512))
        # in-flight gauges per step (incremented at start, decremented at end)
        self.inflight: dict[str, int] = defaultdict(int)

    def _load_seeds(self, path: str) -> list[dict]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"seeds_path not found: {path}")
        seeds: list[dict] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))
        if not seeds:
            raise RuntimeError(f"no seeds loaded from {path}")
        logger.info(f"loaded {len(seeds)} seeds from {path}")
        return seeds

    def accuracy_stats(self) -> dict:
        if not self.accuracy_history:
            return {"mean": 0.5, "count": 0}
        return {
            "mean": sum(self.accuracy_history) / len(self.accuracy_history),
            "count": len(self.accuracy_history),
        }


STATE: Optional[ServerState] = None


@asynccontextmanager
async def timed(state: ServerState, name: str):
    """Measure wall-clock for a pipeline step and track in-flight count.
    Both timing and concurrency feed into /stats so we can see whether each
    step is slow per-call or slow because we're queueing behind a small
    number of vLLM seats."""
    state.inflight[name] += 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        state.timings[name].append(time.perf_counter() - t0)
        state.inflight[name] -= 1


def _summarize_timings(timings: dict[str, deque[float]]) -> dict:
    out = {}
    for name, w in timings.items():
        if not w:
            continue
        arr = sorted(w)
        n = len(arr)
        out[name] = {
            "n": n,
            "avg": round(sum(arr) / n, 3),
            "p50": round(arr[n // 2], 3),
            "p95": round(arr[min(int(n * 0.95), n - 1)], 3),
            "max": round(arr[-1], 3),
        }
    return out


# ======================================================================
# LLM / Milvus helpers
# ======================================================================
async def _api_call(state: ServerState, system_prompt: str, user_prompt: str,
                    max_tokens: int = 2048, temperature: float = 0.8,
                    label: str = "chat", want_json: bool = False) -> str:
    # Provider-specific payload shape (see verl/utils/reward_score/
    # self_evolving.py for the matching judge-side code). The OpenAI HTTP
    # contract is identical; only the "control thinking" extension differs.
    # We pick thinking on/off from the caller's `temperature` so existing
    # call sites stay unchanged: creative calls (proposer/generator at
    # 0.8/0.9) get thinking enabled; the validator at 0.2 stays fast.
    provider = os.environ.get("CHAT_PROVIDER", "vllm").lower()
    want_thinking = temperature >= 0.5
    payload: dict = {
        "model": state.args.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    if provider == "vllm":
        payload["temperature"] = temperature
        payload["chat_template_kwargs"] = {"enable_thinking": want_thinking}
    elif provider == "kimi":
        # Kimi k2.6 hard-fixes temperature (1.0 thinking / 0.6 non-thinking)
        # and 400s on any explicit value, so we just signal thinking mode.
        payload["thinking"] = {"type": "enabled" if want_thinking else "disabled"}
    elif provider == "trapi":
        # TRAPI Azure-OpenAI models (Kimi-K2.6, gpt-5.5, ...) reject temperature,
        # repetition penalty, chat_template_kwargs and the Moonshot `thinking`
        # field. They also want `max_completion_tokens` (gpt-5.x/o-series 400 on
        # `max_tokens`; Kimi accepts either). Reasoning on/off switch is
        # reasoning_effort: omit for reasoning, "none" to disable.
        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = max_tokens
        if not want_thinking:
            payload["reasoning_effort"] = "none"
    elif provider == "deepseek":
        # DeepSeek-V4-Pro thinking knob differs from Qwen's enable_thinking:
        # chat_template_kwargs {"thinking": bool, "reasoning_effort": ...}.
        # Creative calls (proposer/generator, temp>=0.5) reason at the configured
        # effort; the validator (temp 0.2) stays fast with thinking off.
        payload["temperature"] = temperature
        payload["chat_template_kwargs"] = {"thinking": want_thinking}
        if want_thinking:
            payload["chat_template_kwargs"]["reasoning_effort"] = os.environ.get(
                "DEEPSEEK_REASONING_EFFORT", "high")
    else:  # openai-compatible / generic
        payload["temperature"] = temperature
    # JSON Mode (Kimi / OpenAI). When the caller will pass the response
    # through json.loads, asking the provider to constrain output to a
    # valid JSON object eliminates the "stripped fenced code block + chat
    # preamble + trailing comma" failure modes that _parse_json works
    # around heuristically. Kimi and OpenAI both honor
    # response_format={"type":"json_object"} but require the prompt itself
    # to describe the schema (which our agent system prompts already do).
    # vllm without guided_json doesn't accept this field — skip it there.
    if want_json and provider in ("kimi", "openai", "trapi"):
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {state.args.api_key}"}
    # When the chat server is saturated (e.g. heavy reward-judge traffic + 8
    # gen workers each issuing proposer/generator/validator calls with thinking
    # enabled) individual requests can queue for many minutes before they even
    # start streaming. Override via GEN_CHAT_TIMEOUT env if you keep getting
    # ReadTimeouts.
    timeout = float(os.environ.get("GEN_CHAT_TIMEOUT", "1800"))
    async with timed(state, label):
        resp = await state.http_client.post(
            f"{state.args.api_base}/chat/completions",
            json=payload, headers=headers, timeout=timeout,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        # With vLLM `--reasoning-parser qwen3`, the `<think>...</think>` block
        # is moved to `reasoning_content` (sometimes `reasoning`) and `content`
        # is only the post-thinking answer. If the model ran out of tokens
        # while still thinking, content is None — fall back to the reasoning
        # text so _parse_json can still recover an embedded JSON.
        content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        return content


async def _embed_text(state: ServerState, text: str) -> list[float]:
    payload = {"model": state.args.embed_model, "input": [text[:2000]]}
    headers = {"Authorization": f"Bearer {state.args.api_key}"}
    async with timed(state, "embed"):
        resp = await state.http_client.post(
            f"{state.args.embed_api_base}/embeddings",
            json=payload, headers=headers, timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def _milvus_search_sync(state: ServerState, embedding: list[float], top_k: int,
                        filter_expr: str = "") -> list[dict]:
    """Sync Milvus call. Each thread keeps its own MilvusClient (gRPC channels
    are not safe to share across threads, and a closed channel poisons the
    cached client). Reconnect once on closed-channel errors before giving up.

    `filter_expr` is an optional Milvus boolean expression (e.g. restricting to
    CLIMB train rows). `image_path` + `entry_id` are also returned so multimodal
    callers can resolve the media file and tell train/valid rows apart."""
    from pymilvus import MilvusClient

    def _new_client():
        return MilvusClient(uri=state.args.milvus_uri, token=state.args.milvus_token)

    client = getattr(state._milvus_local, "client", None)
    if client is None:
        client = _new_client()
        state._milvus_local.client = client

    last_err = None
    for attempt in range(2):
        try:
            search_kwargs = dict(
                collection_name=state.args.milvus_collection,
                data=[embedding],
                limit=top_k,
                output_fields=[
                    "source_dataset", "modality", "content_type",
                    "text_content", "question", "answer",
                    "image_path", "entry_id",
                ],
            )
            if filter_expr:
                search_kwargs["filter"] = filter_expr
            results = client.search(**search_kwargs)
            hits: list[dict] = []
            for hit_list in results:
                for hit in hit_list:
                    e = hit["entity"]
                    hits.append({
                        "source": e.get("source_dataset", ""),
                        "modality": e.get("modality", ""),
                        "content_type": e.get("content_type", ""),
                        "text": e.get("text_content", ""),
                        "question": e.get("question", ""),
                        "answer": e.get("answer", ""),
                        "image_path": e.get("image_path", ""),
                        "entry_id": e.get("entry_id", ""),
                        "score": hit["distance"],
                    })
            return hits
        except Exception as e:
            last_err = e
            msg = str(e)
            recoverable = (
                "closed channel" in msg or "RPC" in msg
                or "UNAVAILABLE" in msg or "Connection" in msg
            )
            if attempt == 0 and recoverable:
                try:
                    if hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass
                client = _new_client()
                state._milvus_local.client = client
                continue
            break
    logger.warning(f"milvus search failed: {last_err}")
    return []


async def _milvus_search(state: ServerState, query_text: str, top_k: int,
                         filter_expr: str = "") -> list[dict]:
    try:
        embedding = await _embed_text(state, query_text)
    except Exception as e:
        logger.warning(
            f"embed failed for '{query_text[:60]}': {type(e).__name__}: {e!r}"
        )
        return []
    async with timed(state, "milvus_search"):
        return await asyncio.to_thread(_milvus_search_sync, state, embedding, top_k, filter_expr)


# Milvus boolean filter selecting CLIMB train-split multimodal rows only. The
# build_medical_knowledge_v2 indexer keys CLIMB rows as "climb_<split>_<line>",
# so the train split is exactly entry_id LIKE "climb_train_%".
CLIMB_TRAIN_FILTER = 'source_dataset == "climb" and entry_id like "climb_train_%"'


async def _milvus_search_climb(state: ServerState, query_text: str, top_k: int,
                               images_only: bool = False) -> list[dict]:
    """Retrieve CLIMB train-split multimodal neighbors for a query. Restricts to
    image modality when `images_only` (video frame extraction is best-effort)."""
    expr = CLIMB_TRAIN_FILTER
    if images_only:
        expr += ' and modality == "image"'
    hits = await _milvus_search(state, query_text, top_k, filter_expr=expr)
    return [h for h in hits if h.get("image_path")]


# ======================================================================
# Milvus history collection: every proposed question + running solver acc.
#
# The history collection is separate from the read-only `medical_knowledge`
# RAG collection. We own write access here: insert at validator-accept time,
# upsert num_reports/num_correct counters as the trainer streams feedback.
# At propose time the worker queries top-K neighbors to (a) avoid repeating
# very similar questions and (b) surface what the solver is good at / bad at.
# ======================================================================


def _maybe_log_sample(entry: dict, mode: str, prob: float = 0.01) -> None:
    """Print full question + answer to the log with probability `prob` so we
    can spot-check what the proposer is generating without flooding the log
    (each step has ~64 inserts, so 1% ≈ a sample every step or two)."""
    if random.random() >= prob:
        return
    extra = entry.get("extra_info") or {}
    qid = extra.get("question_id", "?")
    fmt = extra.get("format", "?")
    rm = entry.get("reward_model") if isinstance(entry.get("reward_model"), dict) else {}
    gt = str(rm.get("ground_truth", "")).strip()
    ref = entry.get("reference_response")
    lines = [
        "",
        "================ gen sample (1%) ================",
        f"qid={qid}  mode={mode}  format={fmt}",
    ]
    if ref:
        # SFT mode: print the full 2-turn conversation actually used for
        # training — the user turn and the assistant turn (reasoning + answer).
        user_text = ""
        for m in entry.get("prompt", []):
            if m.get("role") == "user":
                user_text = _text_from_content(m.get("content"))
        lines.append(f"USER:\n{user_text.strip()}")
        lines.append(f"ASSISTANT:\n{ref.strip()}")
        lines.append(f"(ground_truth={gt}  teacher_answer={extra.get('teacher_answer', '')})")
    else:
        question = (extra.get("question") or "").strip()
        options = extra.get("options") if isinstance(extra.get("options"), dict) else None
        lines.append(f"Q: {question}")
        if options:
            for k, v in options.items():
                lines.append(f"   {k}. {v}")
        lines.append(f"A: {gt}")
    lines.append("=================================================")
    logger.info("\n".join(lines))


def _truncate_right_bytes(s: str, max_chars: int, max_bytes: int) -> str:
    """Trim `s` so it fits within both `max_chars` and `max_bytes` (UTF-8),
    keeping the RIGHT side. mimiciv_rare prompts put the actual question at
    the end after a long demographics/labs preamble, so the suffix is the
    semantically important part to retain. Milvus varchar fields are sized
    in bytes, so we also have to clamp the encoded byte length."""
    if not s:
        return ""
    if len(s) > max_chars:
        s = s[-max_chars:]
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    # Drop bytes from the left until we fit, then decode permissively in case
    # we sliced mid-character.
    b = b[-max_bytes:]
    return b.decode("utf-8", errors="ignore")

def _history_ensure_collection_sync(state) -> None:
    """Create the history collection if missing. Idempotent — safe to call
    on every server start."""
    from pymilvus import DataType, MilvusClient

    coll = state.args.milvus_history_collection
    client = MilvusClient(uri=state.args.milvus_uri, token=state.args.milvus_token)
    try:
        if client.has_collection(coll):
            return
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("entry_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("embedding", DataType.FLOAT_VECTOR,
                         dim=state.args.milvus_embedding_dim)
        schema.add_field("question_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("answer_text", DataType.VARCHAR, max_length=2048)
        schema.add_field("question_format", DataType.VARCHAR, max_length=16)
        schema.add_field("mode", DataType.VARCHAR, max_length=16)
        schema.add_field("num_reports", DataType.INT64)
        schema.add_field("num_correct", DataType.INT64)
        schema.add_field("created_at", DataType.INT64)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", metric_type="COSINE",
                               index_type="AUTOINDEX")
        client.create_collection(coll, schema=schema, index_params=index_params)
        logger.info(f"created milvus history collection '{coll}'")
    finally:
        try:
            client.close()
        except Exception:
            pass


def _history_insert_sync(state, entry_id: str, embedding: list[float],
                         question_text: str, answer_text: str,
                         question_format: str, mode: str) -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=state.args.milvus_uri, token=state.args.milvus_token)
    try:
        client.insert(
            collection_name=state.args.milvus_history_collection,
            data=[{
                "entry_id": entry_id,
                "embedding": embedding,
                "question_text": _truncate_right_bytes(question_text, 4096, 8000),
                "answer_text": _truncate_right_bytes(answer_text or "", 1024, 2000),
                "question_format": (question_format or "")[:16],
                "mode": (mode or "")[:16],
                "num_reports": 0,
                "num_correct": 0,
                "created_at": int(time.time()),
            }],
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


async def history_insert(state, entry: dict, mode: str) -> None:
    """Embed the question text and insert this entry into the history
    collection. Fire-and-forget: failures are logged but don't block the
    worker — Milvus being down should never wedge the pool."""
    extra = entry.get("extra_info") or {}
    entry_id = extra.get("question_id")
    question_text = (extra.get("question") or "").strip() or _extract_user_text(entry)
    if not entry_id or not question_text:
        return
    try:
        embedding = await _embed_text(state, question_text)
    except Exception as e:
        logger.warning(f"history embed failed for {entry_id}: {type(e).__name__}: {e}")
        return
    # Track in-memory counters so we never miss a /report that arrives before
    # the Milvus insert lands.
    state.history_counters.setdefault(entry_id, {"num_reports": 0, "num_correct": 0})
    answer_text = ""
    if entry.get("reward_model") and isinstance(entry["reward_model"], dict):
        answer_text = str(entry["reward_model"].get("ground_truth", ""))
    question_format = extra.get("format", "")
    try:
        await asyncio.to_thread(
            _history_insert_sync, state, entry_id, embedding,
            question_text, answer_text, question_format, mode,
        )
    except Exception as e:
        logger.warning(f"history insert failed for {entry_id}: {type(e).__name__}: {e}")


def _history_upsert_perf_sync(state, entry_id: str, num_reports: int,
                              num_correct: int) -> bool:
    """Read existing row, bump counters, upsert. Returns True if the row
    existed (we found and updated it), False if it was missing (race with
    insert — caller may retry later)."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=state.args.milvus_uri, token=state.args.milvus_token)
    try:
        rows = client.get(
            collection_name=state.args.milvus_history_collection,
            ids=[entry_id],
        )
        if not rows:
            return False
        row = rows[0]
        client.upsert(
            collection_name=state.args.milvus_history_collection,
            data=[{
                "entry_id": entry_id,
                "embedding": row["embedding"],
                "question_text": _truncate_right_bytes(
                    row.get("question_text", "") or "", 4096, 8000
                ),
                "answer_text": _truncate_right_bytes(
                    row.get("answer_text", "") or "", 1024, 2000
                ),
                "question_format": (row.get("question_format") or "")[:16],
                "mode": (row.get("mode") or "")[:16],
                "num_reports": int(num_reports),
                "num_correct": int(num_correct),
                "created_at": row.get("created_at", int(time.time())),
            }],
        )
        return True
    finally:
        try:
            client.close()
        except Exception:
            pass


def _history_search_sync(state, embedding: list[float], top_k: int) -> list[dict]:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=state.args.milvus_uri, token=state.args.milvus_token)
    try:
        results = client.search(
            collection_name=state.args.milvus_history_collection,
            data=[embedding],
            limit=top_k,
            output_fields=["question_text", "answer_text", "question_format",
                           "num_reports", "num_correct"],
        )
        hits: list[dict] = []
        for hit_list in results:
            for hit in hit_list:
                e = hit["entity"]
                hits.append({
                    "question": e.get("question_text", ""),
                    "answer": e.get("answer_text", ""),
                    "format": e.get("question_format", ""),
                    "num_reports": int(e.get("num_reports", 0) or 0),
                    "num_correct": int(e.get("num_correct", 0) or 0),
                    "score": hit.get("distance"),
                })
        return hits
    finally:
        try:
            client.close()
        except Exception:
            pass


async def history_search(state, query_text: str, top_k: Optional[int] = None) -> list[dict]:
    """Find the top-k most-similar previously-proposed questions."""
    if top_k is None:
        top_k = state.args.history_retrieve_top_k
    try:
        embedding = await _embed_text(state, query_text)
    except Exception as e:
        logger.warning(
            f"history search embed failed for '{query_text[:60]}': "
            f"{type(e).__name__}: {e!r}"
        )
        return []
    try:
        return await asyncio.to_thread(_history_search_sync, state, embedding, top_k)
    except Exception as e:
        logger.warning(f"history search failed: {type(e).__name__}: {e}")
        return []


def format_history_context(hits: list[dict]) -> str:
    """Render top-k neighbor entries as a compact context block for the
    proposer prompt. Each entry shows the question, format, and observed
    solver accuracy (correct/reports) so the proposer can reason about gaps."""
    if not hits:
        return ""
    lines = ["Recently proposed similar questions and the solver's running accuracy:"]
    for i, h in enumerate(hits, 1):
        n_rep = h.get("num_reports", 0) or 0
        n_cor = h.get("num_correct", 0) or 0
        if n_rep:
            perf = f"{n_cor}/{n_rep} correct ({n_cor / n_rep:.0%})"
        else:
            perf = "not yet scored"
        fmt = h.get("format") or ""
        q = (h.get("question") or "").replace("\n", " ").strip()
        if len(q) > 400:
            q = q[:400] + "…"
        lines.append(f"{i}. [{fmt}] {q}  [{perf}]")
    lines.append("")
    lines.append(
        "Use these to (a) NOT repeat or near-paraphrase any of the above, "
        "(b) infer what topics/skills the solver is consistently RIGHT about and "
        "should NOT be drilled further, and (c) target the gaps where the solver is "
        "getting answers wrong or has not been tested."
    )
    return "\n".join(lines)


def _parse_json(s: str, expect_array: bool = False):
    """Extract JSON from a possibly-noisy model response.

    Tries in order: (1) whole string as JSON; (2) ```json ... ``` fenced block;
    (3) scan for the first character that starts a JSON value of the expected
    type (`[` for arrays, `{` for objects) and use `JSONDecoder.raw_decode` to
    consume the first complete value, which tolerates trailing text like
    "...thought... {valid json} ...more explanation...".

    Reasoning models (Qwen3.x thinking mode) often emit JSON embedded in a
    longer prose response, especially when the chat server's reasoning_parser
    has truncated `<think>` and we fall back to `reasoning_content`.
    """
    s = s.strip()
    expect_type = list if expect_array else dict
    decoder = json.JSONDecoder()

    try:
        v = json.loads(s)
        if isinstance(v, expect_type):
            return v
    except (json.JSONDecodeError, ValueError):
        pass

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        try:
            v = json.loads(fence.group(1).strip())
            if isinstance(v, expect_type):
                return v
        except (json.JSONDecodeError, ValueError):
            pass

    open_char = "[" if expect_array else "{"
    idx = 0
    while True:
        start = s.find(open_char, idx)
        if start < 0:
            break
        try:
            v, _ = decoder.raw_decode(s[start:])
            if isinstance(v, expect_type):
                return v
        except json.JSONDecodeError:
            pass
        idx = start + 1

    raise ValueError(f"no JSON found in: {s}")


def _extract_user_text(target: dict) -> str:
    for msg in target.get("prompt", []):
        if msg.get("role") == "user":
            content = msg["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return " ".join(texts)
    return ""


# ======================================================================
# Agents
# ======================================================================
async def agent_query_proposer(state: ServerState, target: dict,
                               history_context: str = "") -> list[str]:
    target_question = (
        target.get("extra_info", {}).get("question", "")
        or _extract_user_text(target)
    )
    parts = [f"Target training question:\n{target_question}"]
    if history_context:
        parts.append(history_context)
    parts.append(f"Generate {state.args.n_queries} diverse search queries.")
    user_prompt = "\n\n".join(parts)
    # Kimi JSON Mode only outputs JSON Objects — not arrays — so we don't
    # request response_format here. The proposer's prompt already pins the
    # output to "JSON array of 10 strings" and _parse_json strips fenced
    # blocks / trailing commas heuristically.
    response = await _api_call(state, QUERY_PROPOSER_SYSTEM_PROMPT, user_prompt,
                               max_tokens=12288, temperature=0.8,
                               label="chat_query_proposer")
    queries = _parse_json(response, expect_array=True)
    if not isinstance(queries, list):
        raise ValueError("query proposer did not return a list")
    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    if not queries:
        raise ValueError("query proposer returned no valid queries")
    return queries[:state.args.n_queries]


async def agent_question_generator(state: ServerState, target_question: str,
                                   knowledge: str, accuracy_stats: dict,
                                   required_format: str) -> dict:
    sys_prompt = QUESTION_GENERATOR_SYSTEM_PROMPT.format(
        required_format=required_format,
        accuracy=accuracy_stats["mean"],
        accuracy_count=accuracy_stats["count"],
    )
    user_prompt = (
        f"Reference training question:\n{target_question}\n\n"
        f"Retrieved medical knowledge:\n{knowledge}\n\n"
        f"Synthesize one new training question in the required format ({required_format})."
    )
    response = await _api_call(state, sys_prompt, user_prompt, max_tokens=12288,
                                temperature=0.9, label="chat_generator", want_json=True)
    q = _parse_json(response)
    fmt = q.get("format", "").lower()
    question = q.get("question", "").strip()
    answer = str(q.get("answer", "")).strip()
    if not question or not answer:
        raise ValueError(f"missing question/answer: {q}")
    if fmt != required_format:
        raise ValueError(f"format mismatch: got {fmt}, want {required_format}")
    if fmt == "mcq":
        options = q.get("options", {})
        if not isinstance(options, dict) or len(options) < 2:
            raise ValueError(f"mcq missing options: {q}")
        ans_letter = answer.upper()[:1]
        if ans_letter not in options:
            raise ValueError(f"mcq answer {answer} not in options {list(options)}")
        q["format"] = "mcq"
        q["answer"] = ans_letter
        q["options"] = options
    else:
        q["format"] = "free"
        q["answer"] = answer
    return q


async def agent_validator(state: ServerState, generated: dict) -> tuple[bool, str]:
    query_text = generated["question"]
    if generated.get("format") == "mcq":
        query_text += " " + " ".join(generated.get("options", {}).values())
    hits = await _milvus_search(state, query_text, top_k=3)
    if not hits:
        return True, "no retrieval results — out-of-knowledge, accepted"

    passages = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits[:3])
    if generated.get("format") == "mcq":
        q_text = (
            f"Question: {generated['question']}\n"
            f"Options: {json.dumps(generated.get('options', {}))}\n"
            f"Proposed answer: {generated['answer']}"
        )
    else:
        q_text = (
            f"Question: {generated['question']}\n"
            f"Proposed answer: {generated['answer']}"
        )
    user_prompt = (
        f"{q_text}\n\nRetrieved passages from the database:\n{passages}\n\n"
        "Does the proposed answer CONTRADICT the retrieved knowledge?"
    )
    try:
        response = await _api_call(state, QUESTION_VALIDATOR_SYSTEM_PROMPT, user_prompt,
                                   max_tokens=12288, temperature=0.2,
                                   label="chat_validator", want_json=True)
        result = _parse_json(response)
        verdict = result.get("verdict", "").lower()
        reason = result.get("reason", "")
        if verdict == "contradict":
            return False, reason
        return True, reason
    except Exception as e:
        logger.warning(f"validator failed, accepting: {e}")
        return True, f"validator error: {e}"


def _pick_mode(state: ServerState) -> str:
    """Pick the mode whose current pool share is most below its target.

    Mode counts are number of *entries pushed*, not number of iterations,
    so a single generate-iteration that yields N entries contributes N to
    its mode's count. Modes with zero seeds available (e.g. ``gen_test``
    when ``test_seeds_path`` is unset) are skipped.
    """
    counts = state.mix_counts
    targets = state.mix_targets
    total = sum(counts.values()) + len(counts)  # +len for Laplace smoothing
    best_mode = "direct"
    best_deficit = -float("inf")
    for mode, target in targets.items():
        if target <= 0:
            continue
        if mode == "gen_test" and not state.test_seeds:
            continue
        if mode == "gen_mm" and not state.climb_seeds:
            continue
        share = (counts[mode] + 1) / total
        deficit = target - share
        if deficit > best_deficit:
            best_deficit = deficit
            best_mode = mode
    return best_mode


def _build_raw_entry(target: dict, target_idx: int, cycle: int) -> dict:
    """Pool entry built directly from a train.jsonl seed (no LLM rewriting).

    The seed already has data_source/prompt/images/reward_model in verl shape;
    we just clone it, add a question_id, and tag it as direct-insert.
    """
    entry = {
        "data_source": target.get("data_source", "self_evolving"),
        "prompt": list(target.get("prompt", [])),
        "reward_model": dict(target.get("reward_model", {})),
        "extra_info": dict(target.get("extra_info", {})),
    }
    if "images" in target:
        entry["images"] = target["images"]
    entry["extra_info"]["question_id"] = uuid.uuid4().hex
    entry["extra_info"]["split"] = "train"
    entry["extra_info"]["source"] = "direct_seed"
    entry["extra_info"]["cycle"] = cycle
    entry["extra_info"]["target_idx"] = target_idx
    return entry


def _build_entry(state: ServerState, generated: dict, target: dict,
                 passage: str, query: str, target_idx: int, cycle: int) -> dict:
    state.question_counter += 1
    target_id = (
        target.get("extra_info", {}).get("pubmed_id", "")
        or target.get("extra_info", {}).get("hadm_id", "")
        or target.get("id", "")
        or f"t{target_idx}"
    )

    if generated["format"] == "mcq":
        options = generated["options"]
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        user_content = (
            f"{generated['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            "Choose the single best answer (A, B, C, or D)."
        )
        sys_prompt = SOLVER_SYSTEM_PROMPT_MCQ
        gt = generated["answer"]
        style = "rule_mcq"
    else:
        user_content = generated["question"]
        sys_prompt = SOLVER_SYSTEM_PROMPT_FREE
        gt = generated["answer"]
        style = "rule_free"

    if state.args.no_label:
        gt = ""

    qid = uuid.uuid4().hex
    return {
        "data_source": "self_evolving",
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
        "reward_model": {"style": style, "ground_truth": gt},
        "extra_info": {
            "question_id": qid,
            "index": state.question_counter,
            "split": "train",
            "source": "self_evolving_multi_agent",
            "cycle": cycle,
            "target_idx": target_idx,
            "target_id": str(target_id),
            "format": generated["format"],
            "question": generated["question"],
            "answer": generated["answer"],
            "options": generated.get("options", {}) if generated["format"] == "mcq" else {},
            "passage": passage[:4000],
            "retrieval_query": query,
        },
    }


# ======================================================================
# Multimodal (CLIMB) generation
# ----------------------------------------------------------------------
# A climb seed (real train image/video + question + label) is combined with a
# few retrieved CLIMB train-split neighbors. The teacher SEES the images and
# synthesizes a NEW question that references them with <<k>> tokens; we map
# those back to <image>/<video> placeholders and ship the entry with portable
# "climb://<relpath>" media handles the trainer resolves from the file server.
# ======================================================================
def _climb_media_item_from_seed(seed: dict) -> Optional[dict]:
    """Turn a climb seed entry into a media item for the generation prompt."""
    rel = None
    media_modality = "image"
    for vid in (seed.get("videos") or []):
        r = _climb_relpath(vid)
        if r:
            rel, media_modality = r, "video"
            break
    if rel is None:
        for im in (seed.get("images") or []):
            r = _climb_relpath(im)
            if r:
                rel, media_modality = r, "image"
                break
    if rel is None:
        return None
    extra = seed.get("extra_info") or {}
    return {
        "rel": rel,
        "modality": media_modality,
        "clinical_modality": extra.get("modality") or "unknown",
        "question": extra.get("question") or _extract_user_text(seed),
        "answer": str((seed.get("reward_model") or {}).get("ground_truth", "")),
    }


def _climb_media_item_from_hit(hit: dict) -> Optional[dict]:
    """Turn a Milvus CLIMB hit into a media item. image_path is
    high_modality-relative (build_medical_knowledge_v2). Kept dependency-free
    (this server runs as a script, so `verl` is not importable)."""
    raw = (hit.get("image_path") or "").lstrip("/")
    rel = raw[len("high_modality/"):] if raw.startswith("high_modality/") else raw
    if not rel:
        return None
    clinical = rel.split("/")[0] or "unknown"
    return {
        "rel": rel,
        "modality": hit.get("modality") or "image",
        "clinical_modality": clinical,
        "question": hit.get("question") or "",
        "answer": hit.get("answer") or "",
    }


async def agent_question_generator_mm(state: ServerState, media_items: list,
                                      accuracy_stats: dict, required_format: str) -> dict:
    """Synthesize one multimodal question over `media_items` (the teacher sees
    the attached images / sampled video frames). Returns the validated generator
    dict ({format, question with <<k>> refs, options?, answer})."""
    sys_prompt = QUESTION_GENERATOR_MM_SYSTEM_PROMPT.format(
        required_format=required_format,
        accuracy=accuracy_stats["mean"],
        accuracy_count=accuracy_stats["count"],
    )
    n_frames = int(getattr(state.args, "mm_video_frames", 2))
    max_pixels = int(getattr(state.args, "mm_max_pixels", 1048576))
    content: list = []
    for i, item in enumerate(media_items, 1):
        content.append({"type": "text", "text": (
            f"[MEDIA {i}] clinical_modality={item['clinical_modality']} type={item['modality']}\n"
            f"original question: {(item.get('question') or '')[:600]}\n"
            f"answer/label: {(item.get('answer') or '')[:300]}"
        )})
        if item["modality"] == "video":
            uris = await asyncio.to_thread(_climb_video_frame_uris, item["rel"], n_frames)
            for uri in uris:
                content.append({"type": "image_url", "image_url": {"url": uri}})
        else:
            uri = await asyncio.to_thread(
                _image_to_data_uri, {"image": f"climb://{item['rel']}", "max_pixels": max_pixels}
            )
            if uri:
                content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": (
        f"Synthesize ONE new {required_format} question per the rules, referencing media "
        f"with <<k>> tokens (k in 1..{len(media_items)})."
    )})

    response = await _api_call(state, sys_prompt, content, max_tokens=4096,
                               temperature=0.9, label="chat_generator_mm", want_json=True)
    q = _parse_json(response)
    fmt = (q.get("format") or "").lower()
    question = (q.get("question") or "").strip()
    answer = str(q.get("answer") or "").strip()
    if not question or not answer:
        raise ValueError(f"mm missing question/answer: {q}")
    if "<<" not in question:
        raise ValueError("mm question references no media (<<k>>)")
    if fmt != required_format:
        raise ValueError(f"mm format mismatch: got {fmt}, want {required_format}")
    if fmt == "mcq":
        options = q.get("options", {})
        if not isinstance(options, dict) or len(options) < 2:
            raise ValueError(f"mm mcq missing options: {q}")
        ans_letter = answer.upper()[:1]
        if ans_letter not in options:
            raise ValueError(f"mm mcq answer {answer} not in options {list(options)}")
        q["format"], q["answer"], q["options"] = "mcq", ans_letter, options
    else:
        q["format"], q["answer"] = "free", answer
    return q


def _build_mm_entry(state: ServerState, generated: dict, media_items: list,
                    target_idx: int, cycle: int) -> Optional[dict]:
    """Build a trainer entry from a multimodal generated question. Maps each
    <<k>> reference to an <image>/<video> placeholder and an aligned
    "climb://<rel>" media handle (one media entry per placeholder occurrence)."""
    state.question_counter += 1
    images_seq: list = []
    videos_seq: list = []

    def _repl(m):
        k = int(m.group(1))
        if k < 1 or k > len(media_items):
            return ""  # drop dangling reference
        item = media_items[k - 1]
        ref = f"climb://{item['rel']}"
        if item["modality"] == "video":
            videos_seq.append(ref)
            return "<video>"
        images_seq.append(ref)
        return "<image>"

    question_text = re.sub(r"<<\s*(\d+)\s*>>", _repl, generated["question"]).strip()
    if not images_seq and not videos_seq:
        # Teacher placed no usable reference — anchor on the first media item.
        item = media_items[0]
        ref = f"climb://{item['rel']}"
        if item["modality"] == "video":
            videos_seq.append(ref)
            question_text = "<video>\n" + question_text
        else:
            images_seq.append(ref)
            question_text = "<image>\n" + question_text

    if generated["format"] == "mcq":
        options = generated["options"]
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        user_content = (
            f"{question_text}\n\nOptions:\n{options_text}\n\n"
            "Choose the single best answer (A, B, C, or D)."
        )
        sys_prompt, style = SOLVER_SYSTEM_PROMPT_MCQ, "rule_mcq"
    else:
        user_content = question_text
        sys_prompt, style = SOLVER_SYSTEM_PROMPT_FREE, "rule_free"

    gt = "" if state.args.no_label else generated["answer"]
    entry = {
        "data_source": "climb_gen",
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
        "reward_model": {"style": style, "ground_truth": gt},
        "extra_info": {
            "question_id": uuid.uuid4().hex,
            "index": state.question_counter,
            "split": "train",
            "source": "climb_mm_gen",
            "cycle": cycle,
            "target_idx": target_idx,
            "format": generated["format"],
            "question": question_text,
            "answer": generated["answer"],
            "options": generated.get("options", {}) if generated["format"] == "mcq" else {},
            "modality": media_items[0]["clinical_modality"],
            "media": [{"rel": it["rel"], "modality": it["modality"]} for it in media_items],
        },
    }
    if images_seq:
        entry["images"] = images_seq
    if videos_seq:
        entry["videos"] = videos_seq
    return entry


# ======================================================================
# SFT teacher-trace generation
# ----------------------------------------------------------------------
# In SFT-distillation mode the server, for each accepted entry, additionally
# asks the (current chat provider = teacher) to SOLVE the question and emit a
# full reasoning trace ending in \boxed{answer}. We verify the teacher's boxed
# answer against the ground truth and, on success, attach the trace as
# `entry["reference_response"]` in the canonical
#     <think>\n{reasoning}\n</think>\n\n\boxed{answer}
# shape the student is trained to imitate. Entries whose teacher trace is wrong
# (or unparseable) after a few retries are dropped, keeping the distillation
# data clean. This path is a no-op unless --sft_mode is set, so the RL pipeline
# is unaffected.
# ======================================================================

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
# ICD-10 code (mirror of verl/utils/reward_score/self_evolving.py): a letter,
# two digits (3rd may be A/B), optional dotted subcode. Lets the *code* drive
# the match so synonymous descriptions of one diagnosis count as equal.
_ICD_CODE_RE = re.compile(r"([A-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?)", re.IGNORECASE)

# A well-formed assistant turn is exactly: a non-empty <think>...</think>
# reasoning block followed by a single \boxed{...} final answer (nothing after).
_MIN_REASONING_CHARS = 40
_TRACE_FORMAT_RE = re.compile(
    r"\s*<think>\s*(?P<reasoning>.+?)\s*</think>\s*\\boxed\{[^{}]*\}\s*\Z",
    re.DOTALL,
)


def _valid_trace_format(ref: str) -> bool:
    """Strict 2-turn target check: the assistant turn must be a non-empty
    <think> reasoning </think> followed by exactly one trailing \\boxed{...}.

    Rejects empty/whitespace-only reasoning (e.g. a teacher that hides its
    chain-of-thought), a missing/misplaced box, or trailing junk after the box.
    """
    m = _TRACE_FORMAT_RE.fullmatch(ref or "")
    if not m:
        return False
    reasoning = m.group("reasoning").strip()
    if len(reasoning) < _MIN_REASONING_CHARS:
        return False
    # No stray second <think>/box inside the reasoning that would break parsing.
    if "<think>" in reasoning or "</think>" in reasoning:
        return False
    return True


def _extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_RE.findall(text or "")
    if not matches:
        # Fallback for nested braces like \boxed{\text{...}}: grab to last brace.
        m = re.search(r"\\boxed\{(.+)\}", text or "", re.DOTALL)
        if not m:
            return None
        ans = m.group(1)
    else:
        ans = matches[-1]
    ans = ans.strip()
    # Strip a \text{...} / \mathrm{...} LaTeX wrapper if present.
    tm = re.match(r"\\(?:text|mathrm|mathbf)\{(.*)\}$", ans)
    if tm:
        ans = tm.group(1).strip()
    return ans


def _icd_code(text: str) -> Optional[str]:
    m = _ICD_CODE_RE.search(text or "")
    return m.group(1).upper() if m else None


def _answer_matches(pred: str, gt: str, fmt: str) -> bool:
    """Teacher-trace correctness gate; mirrors the reward's check_accuracy but
    a touch more lenient for free-form (substring) since the teacher may phrase
    the same diagnosis differently than the label."""
    gtl = (gt or "").strip().lower()
    predl = (pred or "").strip().lower()
    if not gtl:
        return False
    if fmt == "mcq" or (len(gtl) == 1 and gtl in "abcd"):
        pl = re.sub(r"[^a-d]", "", predl)[:1]
        return bool(pl) and pl == gtl
    gc, pc = _icd_code(gt), _icd_code(pred)
    if gc is not None and pc is not None:
        return gc == pc
    return predl == gtl or gtl in predl or predl in gtl


def _compose_trace(raw: str, boxed: str) -> str:
    """Normalize any teacher output into <think>...</think>\n\n\boxed{ans}."""
    body = re.sub(r"</?think>", "", raw or "").strip()
    idx = body.rfind("\\boxed")
    reasoning = (body[:idx].strip() if idx != -1 else body).strip()
    return f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{boxed}}}"


def _text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if c.get("type") == "text")
    return ""


async def _teacher_solve_call(state: ServerState, system_prompt: str,
                              user_prompt: str, max_tokens: int) -> str:
    """Like _api_call but returns the FULL visible reasoning trace.

    The provider's hidden reasoning channel (TRAPI/Azure gpt-5.x) is not
    returned over the API, so we disable it and rely on the solver system
    prompt to elicit a visible chain-of-thought in `content`. For vLLM/Kimi we
    keep thinking on and stitch `reasoning_content` back in front of `content`.
    """
    provider = os.environ.get("CHAT_PROVIDER", "vllm").lower()
    payload: dict = {
        "model": state.args.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if provider == "vllm":
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.6
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    elif provider == "kimi":
        payload["max_tokens"] = max_tokens
        payload["thinking"] = {"type": "enabled"}
    elif provider == "trapi":
        # Leave reasoning at its default: the hidden channel isn't returned, so
        # the visible chain-of-thought is elicited by SFT_TEACHER_SYSTEM_PROMPT.
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.6
    headers = {"Authorization": f"Bearer {state.args.api_key}"}
    timeout = float(os.environ.get("GEN_CHAT_TIMEOUT", "1800"))
    async with timed(state, "chat_teacher"):
        resp = await state.http_client.post(
            f"{state.args.api_base}/chat/completions",
            json=payload, headers=headers, timeout=timeout,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if reasoning and "<think>" not in content:
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    return content


async def _gather_teacher_context(state: ServerState, entry: dict,
                                  question_text: str) -> str:
    """Assemble the retrieved-knowledge context the teacher sees while
    producing an SFT trace.

    The teacher must solve the question with the SAME evidence the answer is
    grounded in — not from the bare question — so it reasons correctly (more
    traces pass the GT gate) and produces grounded reasoning. For generated
    entries the exact passages the question was synthesized from are stored on
    the entry (`extra_info['passage']`); reuse them verbatim. For raw/direct
    seeds (no stored passage) fall back to a fresh Milvus retrieval on the
    question so the teacher is still grounded. Best-effort: returns "" if
    nothing is available, in which case the teacher solves from the question
    alone (previous behavior)."""
    extra = entry.get("extra_info") or {}
    passage = (extra.get("passage") or "").strip()
    if passage:
        return passage
    try:
        hits = await _milvus_search(state, question_text, top_k=state.args.milvus_top_k)
    except Exception as e:
        logger.warning(f"teacher-context retrieval failed: {type(e).__name__}: {e}")
        return ""
    if not hits:
        return ""
    knowledge = "\n\n".join(
        f"[passage {i + 1} / source={h.get('source', '?')}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return knowledge[:4000]


# ======================================================================
# CLIMB remote media. The trainer and this server run on GPU nodes that cannot
# see /scratch/high_modality on disk, so CLIMB images/videos are referenced as
# "climb://<relpath>" handles and fetched over authenticated HTTP from the local
# file server. _CLIMB_FILE_BASE is set from --climb_file_base at startup; the
# token is read from the CLIMB_FILE_TOKEN env on every call (never captured at
# import time).
# ======================================================================
_CLIMB_FILE_BASE = ""


def _climb_relpath(ref) -> Optional[str]:
    """high_modality-relative path for a CLIMB media reference, else None.

    Accepts a "climb://<rel>" string, or a dict carrying that string under
    `image`/`video`, or a dict with a bare `climb_path`. Plain local paths
    return None (they are not CLIMB handles)."""
    cand = None
    if isinstance(ref, str):
        cand = ref
    elif isinstance(ref, dict):
        cand = ref.get("climb_path") or ref.get("image") or ref.get("video")
    if not isinstance(cand, str):
        return None
    if cand.startswith("climb://"):
        cand = cand[len("climb://"):]
    elif not (isinstance(ref, dict) and ref.get("climb_path")):
        return None
    return cand[len("high_modality/"):] if cand.startswith("high_modality/") else cand


def _climb_fetch_bytes_sync(rel: str, max_pixels: Optional[int] = None) -> Optional[bytes]:
    """Fetch one CLIMB media file from the local file server (sync; call via
    to_thread). Optional server-side image downscale via `max_pixels`."""
    if not _CLIMB_FILE_BASE:
        logger.warning("climb media requested but --climb_file_base is unset")
        return None
    url = f"{_CLIMB_FILE_BASE.rstrip('/')}/file/{rel}"
    params = {"max_pixels": int(max_pixels)} if max_pixels else None
    headers = {"Authorization": f"Bearer {os.environ.get('CLIMB_FILE_TOKEN', '')}"}
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=120.0)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"climb media fetch failed for {rel}: {type(e).__name__}: {e}")
        return None


def _climb_video_frame_uris(rel: str, n_frames: int) -> list:
    """Best-effort: fetch a CLIMB video and return up to `n_frames` evenly
    spaced frames as PNG data URIs so the teacher can SEE the clip. Returns []
    if video decoding is unavailable or fails (teacher then uses the text Q/A)."""
    data = _climb_fetch_bytes_sync(rel)
    if not data:
        return []
    try:
        import tempfile

        import cv2  # type: ignore

        suffix = os.path.splitext(rel)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tf:
            tf.write(data)
            tf.flush()
            cap = cv2.VideoCapture(tf.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total <= 0:
                cap.release()
                return []
            n = max(1, n_frames)
            idxs = [min(int(total * (k + 0.5) / n), total - 1) for k in range(n)]
            uris = []
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                ok2, buf = cv2.imencode(".png", frame)
                if ok2:
                    uris.append("data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii"))
            cap.release()
            return uris
    except Exception as e:
        logger.warning(f"climb video frame extraction failed for {rel}: {type(e).__name__}: {e}")
        return []


def _entry_images_present(entry: dict) -> bool:
    """True if every LOCAL media file the entry references exists on disk.

    Remote CLIMB handles ("climb://...") are assumed present — they are served
    by the file server, not on this node's disk — and pass the gate. SFT serves
    media rows through the multimodal tokenizer, which opens each local file, so
    a missing local file would crash the trainer's __getitem__."""
    for media in list(entry.get("images") or []) + list(entry.get("videos") or []):
        if _climb_relpath(media) is not None:
            continue  # remote: trust the file server
        p = media.get("image") if isinstance(media, dict) else media
        if isinstance(media, dict) and not isinstance(p, str):
            p = media.get("video")
        if not isinstance(p, str) or not os.path.isfile(p):
            return False
    return True


def _image_to_data_uri(img) -> Optional[str]:
    """Encode one entry image into a data: URI for the OpenAI/vLLM chat
    image_url field. Accepts a local path (dict {"image": path, "max_pixels": N}
    or a path string) OR a remote CLIMB handle ("climb://rel" / {"image":
    "climb://rel"}), which is fetched from the file server.

    Best-effort downscale to `max_pixels` so the teacher sees the SAME
    resolution the student's multimodal row will. Returns None if unreadable.
    Sync (blocking I/O + PIL) — call via to_thread."""
    rel = _climb_relpath(img)
    max_pixels = img.get("max_pixels") if isinstance(img, dict) else None
    if rel is not None:
        # File server already downscaled when max_pixels is passed.
        data = _climb_fetch_bytes_sync(rel, max_pixels)
        if data is None:
            return None
        mime = "image/png" if max_pixels else (mimetypes.guess_type(rel)[0] or "image/png")
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")

    path = img.get("image") if isinstance(img, dict) else img
    if not isinstance(path, str) or not path:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as e:
        logger.warning(f"teacher image read failed for {path}: {type(e).__name__}: {e}")
        return None
    if max_pixels:
        try:
            from PIL import Image

            im = Image.open(io.BytesIO(data)).convert("RGB")
            if im.width * im.height > max_pixels:
                scale = (max_pixels / float(im.width * im.height)) ** 0.5
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.warning(f"teacher image resize failed for {path}: {type(e).__name__}: {e}")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


async def _build_teacher_content(entry: dict, user_text: str, knowledge: str):
    """Build the teacher's user message content.

    Text-only entries → a plain string. Multimodal entries (direct/raw seeds
    carrying `images`, e.g. an ECG/chest-xray) → a list of OpenAI content parts
    with the seed images interleaved at their `<image>` placeholders, so the
    teacher SEES the same clinical image the student does (the student's SFT row
    carries the image too). The retrieved knowledge is prepended as grounding
    the teacher is told not to cite (SFT_TEACHER_SYSTEM_PROMPT)."""
    prefix = ""
    if knowledge:
        prefix = (
            "Reference medical knowledge (background for your own grounding — "
            "do NOT cite it; the student will not see it):\n"
            f"{knowledge}\n\n"
            "Question to solve and teach:\n"
        )
    images = entry.get("images") or []
    videos = entry.get("videos") or []
    if not images and not videos:
        return prefix + user_text

    # Show video clips to the teacher as N evenly-spaced frames (image_url
    # parts) — the SAME image-only treatment the student gets (the dataset
    # flattens <video> into video_frames <image>s), so teacher and student see
    # consistent inputs. Frame count matches the dataset default.
    n_vf = int(os.environ.get("GEN_SFT_VIDEO_FRAMES", "6"))
    content: list = []
    if prefix:
        content.append({"type": "text", "text": prefix})
    parts = re.split(r"(<image>|<video>)", user_text)
    img_idx = vid_idx = 0
    for p in parts:
        if p == "<image>":
            uri = (
                await asyncio.to_thread(_image_to_data_uri, images[img_idx])
                if img_idx < len(images) else None
            )
            content.append(
                {"type": "image_url", "image_url": {"url": uri}} if uri
                else {"type": "text", "text": "<image>"}
            )
            img_idx += 1
        elif p == "<video>":
            rel = _climb_relpath(videos[vid_idx]) if vid_idx < len(videos) else None
            uris = (
                await asyncio.to_thread(_climb_video_frame_uris, rel, n_vf)
                if rel else []
            )
            if uris:
                for u in uris:
                    content.append({"type": "image_url", "image_url": {"url": u}})
            else:
                content.append({"type": "text", "text": "<video>"})
            vid_idx += 1
        elif p:
            content.append({"type": "text", "text": p})
    # Trailing images without a matching placeholder.
    while img_idx < len(images):
        uri = await asyncio.to_thread(_image_to_data_uri, images[img_idx])
        if uri:
            content.append({"type": "image_url", "image_url": {"url": uri}})
        img_idx += 1
    return content


async def attach_teacher_trace(state: ServerState, entry: dict) -> bool:
    """Solve the entry's question with the teacher and attach a verified trace.

    Returns True and sets entry["reference_response"] on success; False if no
    correct, parseable trace was obtained (caller should drop the entry).
    """
    gt = entry.get("reward_model", {}).get("ground_truth", "")
    if not gt:
        return False  # SFT distillation needs a label to verify the trace
    style = entry.get("reward_model", {}).get("style", "")
    fmt = entry.get("extra_info", {}).get("format") or ("mcq" if style == "rule_mcq" else "free")

    user_text = None
    for m in entry.get("prompt", []):
        if m.get("role") == "user":
            user_text = _text_from_content(m.get("content"))
    if not user_text:
        return False
    # Ground the teacher in the retrieved medical knowledge (and original
    # training context carried in user_text) so its trace is correct and
    # well-supported — NOT a solve from the bare question. The student's prompt
    # (entry["prompt"]) is left untouched: it never sees this material, so the
    # teacher is told to write self-contained reasoning (SFT_TEACHER_SYSTEM_PROMPT).
    knowledge = await _gather_teacher_context(state, entry, user_text)
    # Text entries -> a string; multimodal entries -> a content list with the
    # seed image(s) interleaved so the teacher SEES the clinical image.
    teacher_prompt = await _build_teacher_content(entry, user_text, knowledge)
    # Teacher uses the visible-reasoning prompt (NOT the student's SOLVER prompt),
    # so the distillation trace contains an actual chain-of-thought.
    sys_prompt = SFT_TEACHER_SYSTEM_PROMPT

    for _ in range(state.args.teacher_retries + 1):
        try:
            raw = await _teacher_solve_call(
                state, sys_prompt, teacher_prompt, state.args.teacher_max_tokens,
            )
        except Exception as e:
            logger.warning(f"teacher solve failed: {type(e).__name__}: {e!r}")
            continue
        boxed = _extract_boxed(raw)
        if boxed is None:
            continue
        if not _answer_matches(boxed, gt, fmt):
            continue
        trace = _compose_trace(raw, boxed)
        # Reject unless the assistant turn is a clean
        # <think> reasoning </think> \boxed{answer} (e.g. drop empty-reasoning
        # traces from a teacher that hides its chain-of-thought).
        if not _valid_trace_format(trace):
            logger.debug("teacher trace rejected: bad format (reasoning len/box)")
            continue
        entry["reference_response"] = trace
        entry["extra_info"]["teacher_answer"] = boxed
        return True
    return False


# ======================================================================
# Worker loop
# ======================================================================
async def _process_query_inner(state: ServerState, query: str, required_format: str,
                               target_question: str) -> tuple[list[tuple], list[dict]]:
    """One query -> milvus -> generator -> validator. Returns (accepted, rejected)
    where accepted holds (gen_dict, knowledge_str, query_str) tuples to be built
    into entries by the caller."""
    accepted: list[tuple] = []
    rejected: list[dict] = []
    try:
        hits = await _milvus_search(state, query, top_k=state.args.milvus_top_k)
    except Exception as e:
        logger.warning(f"milvus failed on query: {e}")
        return accepted, rejected
    if not hits:
        return accepted, rejected

    knowledge = "\n\n".join(
        f"[passage {i + 1} / source={h.get('source', '?')}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    stats = state.accuracy_stats()
    for _ in range(state.args.questions_per_query):
        try:
            gen = await agent_question_generator(
                state, target_question, knowledge, stats, required_format,
            )
        except Exception as e:
            logger.warning(
                f"generator failed ({required_format}): {type(e).__name__}: {e!r}"
            )
            continue
        try:
            ok, reason = await agent_validator(state, gen)
        except Exception as e:
            ok, reason = True, f"validator error: {e}"
        if ok:
            accepted.append((gen, knowledge, query))
        else:
            rejected.append({
                "question": gen,
                "reason": reason,
                "retrieval_query": query,
                "passage": knowledge[:500],
            })
    return accepted, rejected


async def _process_query(state: ServerState, query: str, required_format: str,
                         target_question: str) -> tuple[list[tuple], list[dict]]:
    async with timed(state, "process_query_total"):
        return await _process_query_inner(state, query, required_format, target_question)


async def worker_loop(state: ServerState, worker_id: int):
    logger.info(f"worker {worker_id} started")
    while True:
        try:
            # Backoff while pool is full so we don't keep generating into a
            # blocked queue.put (also gives the trainer slack on bursty fetches).
            while state.pool.full():
                await asyncio.sleep(0.5)

            # Pick the most-deficit mode, then sample a seed of the right
            # origin uniformly at random. target_idx / cycle become loose
            # iteration counters used only for logging now.
            mode = _pick_mode(state)
            cur_target_idx = state.target_idx
            cur_cycle = state.cycle
            state.target_idx += 1
            if state.target_idx >= len(state.seeds):
                state.target_idx = 0
                state.cycle += 1
                logger.info(f"completed cycle {state.cycle}")

            if mode == "direct":
                target = random.choice(state.train_seeds)
                entry = _build_raw_entry(target, cur_target_idx, cur_cycle)
                # SFT serves image rows multimodally (teacher + trainer open the
                # files); skip raw seeds whose images are missing on this pod so
                # we neither waste a teacher solve nor serve a row that crashes
                # the trainer's tokenizer.
                if state.args.sft_mode and not _entry_images_present(entry):
                    continue
                if state.args.sft_mode and not await attach_teacher_trace(state, entry):
                    state.stats["sft_traces_skipped"] += 1
                    continue
                async with state.log_lock:
                    with open(state.accepted_log, "a") as f:
                        f.write(json.dumps({
                            "ts": datetime.now().isoformat(),
                            "question_id": entry["extra_info"]["question_id"],
                            "target_idx": cur_target_idx,
                            "cycle": cur_cycle,
                            "direct_insert": True,
                            "entry": entry,
                        }) + "\n")
                state.stats["direct_inserted"] += 1
                state.stats["total_accepted"] += 1
                state.mix_counts["direct"] += 1
                state.history.append(entry)
                await state.pool.put(entry)
                # Mirror into the Milvus history collection so /report can
                # update perf counters and future propose cycles can retrieve
                # neighbors. Fire-and-forget — failures must not block the
                # pool.
                asyncio.create_task(history_insert(state, entry, "direct"))
                _maybe_log_sample(entry, "direct")
                logger.info(
                    f"+ direct qid={entry['extra_info']['question_id']} "
                    f"pool=({state.pool.qsize()}/{state.args.max_pool_size}) "
                    f"accepted={state.stats['total_accepted']}"
                )
                continue

            if mode == "gen_mm":
                seed = random.choice(state.climb_seeds)
                entries: list[dict] = []
                # A fraction of CLIMB output is the REAL seed row served as-is
                # (grounded multimodal training data); the rest are newly
                # synthesized multimodal questions over retrieved neighbors.
                if random.random() < float(getattr(state.args, "mm_direct_prob", 0.3)):
                    entry = {
                        "data_source": "climb_gen",
                        "prompt": list(seed.get("prompt", [])),
                        "reward_model": dict(seed.get("reward_model", {})),
                        "extra_info": dict(seed.get("extra_info", {})),
                    }
                    for k in ("images", "videos"):
                        if seed.get(k):
                            entry[k] = list(seed[k])
                    entry["extra_info"]["question_id"] = uuid.uuid4().hex
                    entry["extra_info"]["split"] = "train"
                    entry["extra_info"]["source"] = "climb_direct"
                    entries.append(entry)
                else:
                    seed_item = _climb_media_item_from_seed(seed)
                    if seed_item is None:
                        continue
                    n_media = max(1, int(getattr(state.args, "mm_images_per_query", 3)))
                    hits = []
                    try:
                        hits = await _milvus_search_climb(
                            state, seed_item["question"], top_k=n_media + 4,
                            images_only=not bool(getattr(state.args, "mm_include_videos", True)),
                        )
                    except Exception as e:
                        logger.warning(f"climb retrieval failed: {type(e).__name__}: {e}")
                    media_items = [seed_item]
                    seen = {seed_item["rel"]}
                    for h in hits:
                        it = _climb_media_item_from_hit(h)
                        if it and it["rel"] not in seen:
                            media_items.append(it)
                            seen.add(it["rel"])
                        if len(media_items) >= n_media:
                            break
                    fmt = "mcq" if state.format_counter % 2 == 0 else "free"
                    state.format_counter += 1
                    try:
                        gen = await agent_question_generator_mm(
                            state, media_items, state.accuracy_stats(), fmt)
                    except Exception as e:
                        logger.warning(f"mm generator failed ({fmt}): {type(e).__name__}: {e!r}")
                        continue
                    entry = _build_mm_entry(state, gen, media_items, cur_target_idx, cur_cycle)
                    if entry is None:
                        continue
                    entries.append(entry)

                for entry in entries:
                    if state.args.sft_mode and not _entry_images_present(entry):
                        state.stats["served_missing_image_skipped"] += 1
                        continue
                    if state.args.sft_mode and not await attach_teacher_trace(state, entry):
                        state.stats["sft_traces_skipped"] += 1
                        continue
                    async with state.log_lock:
                        with open(state.accepted_log, "a") as f:
                            f.write(json.dumps({
                                "ts": datetime.now().isoformat(),
                                "question_id": entry["extra_info"]["question_id"],
                                "target_idx": cur_target_idx,
                                "cycle": cur_cycle,
                                "gen_mm": True,
                                "entry": entry,
                            }) + "\n")
                    state.stats["total_accepted"] += 1
                    state.stats["total_generated"] += 1
                    state.mix_counts["gen_mm"] += 1
                    state.history.append(entry)
                    await state.pool.put(entry)
                    asyncio.create_task(history_insert(state, entry, "gen_mm"))
                    _maybe_log_sample(entry, "gen_mm")
                    logger.info(
                        f"+ gen_mm[{entry['extra_info'].get('source')}] "
                        f"qid={entry['extra_info']['question_id']} "
                        f"pool=({state.pool.qsize()}/{state.args.max_pool_size}) "
                        f"generated={state.stats['total_generated']}"
                    )
                continue

            if mode == "gen_test":
                target = random.choice(state.test_seeds)
            else:
                target = random.choice(state.train_seeds)
            target_question = (
                target.get("extra_info", {}).get("question", "")
                or _extract_user_text(target)
            )

            # Retrieve nearest neighbors from the running gen_history collection
            # so the proposer can avoid repeating questions and target gaps in
            # the solver's coverage. Best-effort: if Milvus is down we just
            # skip the context and the proposer runs unconditioned.
            history_hits = await history_search(state, target_question)
            history_context = format_history_context(history_hits)

            queries: list[str] = []
            for attempt in range(3):
                try:
                    queries = await agent_query_proposer(
                        state, target, history_context=history_context,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"worker {worker_id}: query proposer attempt {attempt + 1}/3 failed: "
                        f"{type(e).__name__}: {e!r}"
                    )
            if not queries:
                continue
            state.stats["total_queries"] += len(queries)

            formats = []
            for _ in queries:
                formats.append("mcq" if state.format_counter % 2 == 0 else "free")
                state.format_counter += 1

            tasks = [
                _process_query(state, q, fmt, target_question)
                for q, fmt in zip(queries, formats)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            accepted_pairs: list[tuple] = []
            rejected_list: list[dict] = []
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"worker {worker_id}: process_query exception: {r}")
                    continue
                a, rj = r
                accepted_pairs.extend(a)
                rejected_list.extend(rj)

            for gen, passage, query in accepted_pairs:
                entry = _build_entry(state, gen, target, passage, query,
                                     cur_target_idx, cur_cycle)
                if state.args.sft_mode and not await attach_teacher_trace(state, entry):
                    state.stats["sft_traces_skipped"] += 1
                    continue
                async with state.log_lock:
                    with open(state.accepted_log, "a") as f:
                        f.write(json.dumps({
                            "ts": datetime.now().isoformat(),
                            "question_id": entry["extra_info"]["question_id"],
                            "target_idx": cur_target_idx,
                            "cycle": cur_cycle,
                            "queries_used": queries,
                            "entry": entry,
                        }) + "\n")
                state.stats["total_accepted"] += 1
                state.stats["total_generated"] += 1
                state.mix_counts[mode] += 1
                state.history.append(entry)
                await state.pool.put(entry)
                asyncio.create_task(history_insert(state, entry, mode))
                _maybe_log_sample(entry, mode)
                logger.info(
                    f"+ gen[{mode}] qid={entry['extra_info']['question_id']} "
                    f"pool=({state.pool.qsize()}/{state.args.max_pool_size}) "
                    f"generated={state.stats['total_generated']}"
                )

            for r in rejected_list:
                async with state.log_lock:
                    with open(state.rejected_log, "a") as f:
                        f.write(json.dumps({
                            "ts": datetime.now().isoformat(),
                            "target_idx": cur_target_idx,
                            "cycle": cur_cycle,
                            **r,
                        }) + "\n")
                state.stats["total_rejected"] += 1
                state.stats["total_generated"] += 1

        except asyncio.CancelledError:
            logger.info(f"worker {worker_id} cancelled")
            raise
        except Exception as e:
            logger.exception(f"worker {worker_id}: unexpected error, sleeping 5s: {e}")
            await asyncio.sleep(5)


# ======================================================================
# FastAPI app
# ======================================================================
class ReportPayload(BaseModel):
    question_id: str
    accuracy: float


class ReplayPayload(BaseModel):
    """Push previously-accepted entries back into the pool.

    Used when restarting the trainer mid-run — gen_server keeps generating
    in the background, so its log accumulates entries the dead trainer
    already consumed. Calling /replay re-injects those entries so the new
    trainer sees the same data instead of starting from scratch.
    """

    log_path: str | None = None  # default: current accepted_log
    count: int | None = None     # max entries to push; None = all parsable
    tail: bool = True            # take last `count` (True) or first (False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    args: argparse.Namespace = app.state.args
    loop = asyncio.get_running_loop()
    STATE = ServerState(args, loop)
    # Each worker can fire up to n_queries (10) embed calls + 1 generator +
    # 1 validator + 1 proposer concurrently, so the pool needs ~workers * 15
    # slots. The default httpx pool is 100/20 — too small for 8+ workers.
    STATE.http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max(args.workers * 16, 256),
            max_keepalive_connections=max(args.workers * 8, 128),
            keepalive_expiry=120.0,
        ),
    )

    # Ensure the Milvus history collection exists before any worker tries
    # to insert into it. Idempotent — does nothing if the collection is
    # already there. Failures here aren't fatal; insert/search will retry
    # later and just log warnings if Milvus is unavailable.
    try:
        await asyncio.to_thread(_history_ensure_collection_sync, STATE)
    except Exception as e:
        logger.warning(
            f"history collection init failed (will retry per-insert): "
            f"{type(e).__name__}: {e}"
        )

    workers = [
        asyncio.create_task(worker_loop(STATE, i)) for i in range(args.workers)
    ]
    logger.info(f"started {args.workers} workers; pool max={args.max_pool_size}")
    try:
        yield
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await STATE.http_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/stats")
async def stats():
    s = STATE
    total_mix = sum(s.mix_counts.values()) or 1
    return {
        "pool_size": s.pool.qsize(),
        "max_pool_size": s.args.max_pool_size,
        "replay_buffer_size": len(s.replay_buffer),
        "history_size": len(s.history),
        "max_history_size": s.history.maxlen,
        "mix_targets": s.mix_targets,
        "mix_counts": s.mix_counts,
        "mix_actual": {k: v / total_mix for k, v in s.mix_counts.items()},
        "accuracy": s.accuracy_stats(),
        "target_idx": s.target_idx,
        "cycle": s.cycle,
        "seeds": len(s.seeds),
        "timings": _summarize_timings(s.timings),
        "inflight": dict(s.inflight),
        **s.stats,
    }


def _log_served(s, entry: dict, source: str) -> None:
    qid = (entry.get("extra_info") or {}).get("question_id", "?")
    logger.info(
        f"- served qid={qid} from={source} "
        f"pool=({s.pool.qsize()}/{s.args.max_pool_size}) "
        f"served={s.stats['served']}"
    )


# Keys every served entry must carry so the trainer's DataProto batch stays
# homogeneous. A key present on only *some* rows of a batch trips DataProto's
# "key <k> length N is not equal to batch size M" assertion (this is exactly
# how a trace-less cold-start seed served alongside fully-generated SFT entries
# crashed the feedback eval). In SFT mode the verified teacher trace is also
# mandatory, since it is the supervised target.
_BASE_REQUIRED_FIELDS = ("prompt", "reward_model", "extra_info")


def _required_fields(s) -> tuple:
    if getattr(s.args, "sft_mode", False):
        return _BASE_REQUIRED_FIELDS + ("reference_response",)
    return _BASE_REQUIRED_FIELDS


def _missing_fields(s, entry: dict) -> list:
    """Names of required keys that are absent or empty on `entry`."""
    missing = []
    for k in _required_fields(s):
        v = entry.get(k)
        if v is None or (isinstance(v, (str, list, dict, tuple)) and len(v) == 0):
            missing.append(k)
    if not (entry.get("extra_info") or {}).get("question_id"):
        missing.append("extra_info.question_id")
    return missing


async def _finalize_served(s, entry: dict, source: str):
    """Completeness gate for /sample: never emit a partial entry.

    Returns `entry` if it carries every required field, else None (the caller
    must skip it and try the next source). In SFT mode an entry whose *only*
    gap is the teacher trace (e.g. a raw cold-start seed or a replayed non-SFT
    log line) is repaired in place by solving it synchronously; if the teacher
    cannot produce a verified trace the entry is rejected rather than served
    incomplete.
    """
    # Backstop: never serve an SFT image row whose files are missing on this
    # pod (would crash the trainer's multimodal tokenizer). Covers replay /
    # history / cold-start-seed sources that bypass the worker's direct-mode gate.
    if getattr(s.args, "sft_mode", False) and not _entry_images_present(entry):
        s.stats["served_missing_image_skipped"] += 1
        return None
    missing = _missing_fields(s, entry)
    if missing == ["reference_response"] and getattr(s.args, "sft_mode", False):
        try:
            if await attach_teacher_trace(s, entry):
                s.stats["served_repaired_inline"] += 1
        except Exception as e:
            logger.warning(
                f"/sample inline trace attach failed: {type(e).__name__}: {e!r}"
            )
        missing = _missing_fields(s, entry)
    if missing:
        s.stats["served_incomplete_skipped"] += 1
        qid = (entry.get("extra_info") or {}).get("question_id", "?")
        logger.warning(
            f"/sample dropping incomplete entry from={source} "
            f"missing={missing} qid={qid}"
        )
        return None
    s.stats["served"] += 1
    _log_served(s, entry, source)
    return entry


@app.get("/sample")
async def sample():
    s = STATE
    # Every candidate below is routed through `_finalize_served`, which drops
    # (or, in SFT mode, repairs) any entry missing a required field so the
    # trainer never receives a partial entry that would break its batch.

    # 1) Replay buffer (FIFO). Skip any replayed line that fails the gate
    #    (e.g. a non-SFT log replayed into an SFT run).
    while s.replay_buffer:
        out = await _finalize_served(s, s.replay_buffer.popleft(), "replay")
        if out is not None:
            return out

    # 2) Freshly-generated pool entries, pulled within a bounded total wait.
    #    If the workers can't keep up (e.g. the chat server is saturated), fall
    #    through to the history / seed fallbacks so the trainer never starves
    #    on a 503; workers keep filling the pool in the background.
    pool_deadline = time.monotonic() + 30
    while True:
        timeout = pool_deadline - time.monotonic()
        if timeout <= 0:
            break
        try:
            entry = await asyncio.wait_for(s.pool.get(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        out = await _finalize_served(s, entry, "pool")
        if out is not None:
            return out

    # 3) History fallback: a random previously-accepted entry. These carry a
    #    trace in SFT mode (history.append runs only after attach), but gate
    #    anyway. Try a handful so one stale/partial entry can't wedge us.
    if s.history:
        for entry in random.sample(s.history, k=min(len(s.history), 16)):
            out = await _finalize_served(s, entry, "history")
            if out is not None:
                s.stats["served_from_history"] += 1
                return out

    # 4) Cold-start seed fallback: a raw labeled training seed so the trainer
    #    never blocks waiting for the generator to spin up. In SFT mode raw
    #    seeds lack a trace, so `_finalize_served` solves them inline; if the
    #    teacher can't verify a trace we skip to the next seed.
    if s.train_seeds:
        for seed in random.sample(s.train_seeds, k=min(len(s.train_seeds), 6)):
            entry = dict(seed)
            extra = dict(entry.get("extra_info") or {})
            extra.setdefault("question_id", f"seed_{id(seed):x}")
            entry["extra_info"] = extra
            out = await _finalize_served(s, entry, "seeds")
            if out is not None:
                s.stats["served_from_seeds"] += 1
                return out

    raise HTTPException(
        status_code=503,
        detail="no complete entry available (pool/history/seeds empty or all incomplete)",
    )


@app.post("/report")
async def report(payload: ReportPayload):
    s = STATE
    acc = float(payload.accuracy)
    s.accuracy_by_id[payload.question_id] = acc
    s.accuracy_history.append(acc)
    s.stats["reports"] += 1
    recent_mean = sum(s.accuracy_history) / len(s.accuracy_history)
    logger.info(
        f"! feedback qid={payload.question_id} acc={acc:.2f} "
        f"reports={s.stats['reports']} recent_mean={recent_mean:.2f}"
    )
    async with s.log_lock:
        with open(s.report_log, "a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(),
                "question_id": payload.question_id,
                "accuracy": acc,
            }) + "\n")

    # Update the Milvus history collection counters. Aggregation: each report
    # is binary accuracy (we treat acc >= 0.5 as correct). The Milvus row
    # stores running num_reports + num_correct; running rate = num_correct /
    # num_reports. Use an in-memory cache to merge concurrent reports for the
    # same qid without race-clobbering the Milvus upsert.
    correct_delta = 1 if acc >= 0.5 else 0
    async with s.history_counter_lock:
        cur = s.history_counters.setdefault(
            payload.question_id, {"num_reports": 0, "num_correct": 0}
        )
        cur["num_reports"] += 1
        cur["num_correct"] += correct_delta
        snapshot = dict(cur)

    async def _upsert():
        try:
            ok = await asyncio.to_thread(
                _history_upsert_perf_sync,
                s,
                payload.question_id,
                snapshot["num_reports"],
                snapshot["num_correct"],
            )
            if not ok:
                logger.debug(
                    f"history upsert: qid {payload.question_id} not yet in "
                    f"Milvus (insert race) — counters cached in memory"
                )
        except Exception as e:
            logger.warning(
                f"history upsert failed for {payload.question_id}: "
                f"{type(e).__name__}: {e}"
            )
    asyncio.create_task(_upsert())

    return {"ok": True}


@app.post("/replay")
async def replay(payload: ReplayPayload):
    """Re-push entries from an accepted_log file into the replay buffer.

    The replay buffer is unbounded and drained by /sample before the regular
    pool, so nothing is dropped regardless of how many entries are replayed.
    Live workers keep filling the pool in the background.
    """
    s = STATE
    log_path = payload.log_path or s.accepted_log
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail=f"log not found: {log_path}")

    entries: list[dict] = []
    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = rec.get("entry")
            if isinstance(entry, dict):
                entries.append(entry)

    if payload.count is not None:
        entries = entries[-payload.count:] if payload.tail else entries[:payload.count]

    for entry in entries:
        s.replay_buffer.append(entry)
    logger.info(
        f"replay: log={log_path} parsed={len(entries)} "
        f"replay_buffer_size={len(s.replay_buffer)}"
    )
    return {
        "log_path": log_path,
        "parsed": len(entries),
        "pushed": len(entries),
        "replay_buffer_size": len(s.replay_buffer),
        "pool_size": s.pool.qsize(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_path", required=True,
                        help="JSONL of seed targets (typically train.jsonl)")
    parser.add_argument("--test_seeds_path", default="",
                        help="Optional JSONL of test seeds. Loaded with "
                             "reward_model stripped — used to drive question "
                             "generation against test-like distributions, but "
                             "never raw-inserted into the pool.")
    parser.add_argument("--direct_target", type=float, default=0.30,
                        help="Target share of pool entries that are raw "
                             "train seeds (direct-inserted, real GT). The "
                             "worker loop picks whichever mode is most below "
                             "its target each iteration. Values across the "
                             "three --*_target flags are renormalized.")
    parser.add_argument("--gen_train_target", type=float, default=0.35,
                        help="Target share of pool entries that are LLM-"
                             "generated from train seeds (synthetic GT).")
    parser.add_argument("--gen_test_target", type=float, default=0.35,
                        help="Target share of pool entries that are LLM-"
                             "generated from test_masked seeds. If no test "
                             "seeds are loaded, this share is added to "
                             "--gen_train_target automatically.")
    parser.add_argument("--api_base", required=True, help="vLLM chat /v1 base URL")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--embed_api_base", required=True, help="vLLM embed /v1 base URL")
    parser.add_argument("--embed_model", required=True)
    parser.add_argument("--milvus_uri", required=True)
    parser.add_argument("--milvus_token", default="root:Milvus")
    parser.add_argument("--milvus_collection", default="medical_knowledge")
    parser.add_argument(
        "--milvus_history_collection",
        default="gen_history",
        help="Milvus collection used to remember every proposed/accepted question + "
             "running solver accuracy. Auto-created on startup if missing.",
    )
    parser.add_argument(
        "--milvus_embedding_dim",
        type=int,
        default=2048,
        help="Embedding dim for the history collection (must match --embed_model).",
    )
    parser.add_argument(
        "--history_retrieve_top_k",
        type=int,
        default=10,
        help="How many neighbor questions to retrieve from history when seeding the "
             "proposer.",
    )
    parser.add_argument("--milvus_top_k", type=int, default=16)
    parser.add_argument("--n_queries", type=int, default=10)
    parser.add_argument("--questions_per_query", type=int, default=1)
    parser.add_argument("--no_label", action="store_true")
    parser.add_argument(
        "--sft_mode", action="store_true",
        help="SFT-distillation mode: for each accepted entry, solve it with the "
             "teacher (current chat provider) and attach a verified reasoning "
             "trace as entry['reference_response']. Entries whose teacher answer "
             "is wrong are dropped. No-op for the RL pipeline.")
    parser.add_argument("--teacher_retries", type=int, default=2,
                        help="Extra teacher attempts when the boxed answer is "
                             "wrong/unparseable before dropping the entry.")
    parser.add_argument("--teacher_max_tokens", type=int, default=4096,
                        help="max_tokens for the teacher solve call (SFT mode).")
    parser.add_argument("--accuracy_window", type=int, default=64)
    parser.add_argument("--max_pool_size", type=int, default=200)
    parser.add_argument("--history_size", type=int, default=5000,
                        help="Capacity of the in-memory ring of every "
                             "accepted pool entry. /sample falls back to "
                             "a random entry from here when the pool is "
                             "empty so the trainer never sees a 503.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent generation workers (each runs the full "
                             "pipeline; use ~1 per N target seeds for steady throughput)")
    parser.add_argument(
        "--log_dir",
        default=os.path.expanduser("~/scratch/dvdai/self_evolving_datasets/logs"),
    )
    # --- CLIMB multimodal generation -----------------------------------
    parser.add_argument("--climb_seeds_path", default="",
                        help="JSONL of CLIMB train seeds (verl shape, "
                             "climb:// media handles) for multimodal generation.")
    parser.add_argument("--climb_file_base", default="",
                        help="Base URL of the CLIMB media file server "
                             "(e.g. http://mib.media.mit.edu:18080). Token read "
                             "from the CLIMB_FILE_TOKEN env, never hardcoded.")
    parser.add_argument("--gen_mm_target", type=float, default=0.0,
                        help="Target pool share of multimodal CLIMB entries. "
                             "Folded into gen_train when no climb seeds are loaded.")
    parser.add_argument("--mm_images_per_query", type=int, default=3,
                        help="Number of media items (seed + retrieved neighbors) "
                             "shown to the teacher per multimodal generation.")
    parser.add_argument("--mm_video_frames", type=int, default=2,
                        help="Frames sampled from each video media item for the "
                             "teacher's view (best-effort, requires opencv).")
    parser.add_argument("--mm_max_pixels", type=int, default=1048576,
                        help="max_pixels for images shown to the teacher / stored "
                             "on generated entries.")
    parser.add_argument("--mm_direct_prob", type=float, default=0.3,
                        help="Probability a gen_mm iteration serves the real seed "
                             "row directly instead of synthesizing a new question.")
    parser.add_argument("--mm_include_videos", action="store_true", default=True,
                        help="Allow video-modality CLIMB neighbors in retrieval.")
    parser.add_argument("--mm_images_only", dest="mm_include_videos",
                        action="store_false",
                        help="Restrict CLIMB retrieval/generation to images.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()

    # Make the file-server base available to the sync media helpers.
    global _CLIMB_FILE_BASE
    _CLIMB_FILE_BASE = args.climb_file_base

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # pymilvus logs every gRPC retry at ERROR with full traceback even when
    # our wrapper recovers; silence it — real failures are logged by
    # _milvus_search_sync when both attempts fail.
    logging.getLogger("pymilvus.decorators").setLevel(logging.CRITICAL)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
