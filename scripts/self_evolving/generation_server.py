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
import math
import mimetypes
import os
import random
import re
import sys
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
# RUBRIC MODE (HealthBench-Professional task + co-generated rubric)
# ----------------------------------------------------------------------
# In rubric mode the server stops generating MCQ/free diagnosis questions and
# instead produces, in ONE LLM call per item, an open-ended clinician TASK plus
# a HealthBench-Professional-style grading RUBRIC. The training reward is purely
# that rubric (graded by self / server 5); see verl/utils/reward_score/
# healthbench_pro.py. Taxonomy + prompt wording are adapted from
# scripts/self_evolving/healthbench_gen.py (the offline equivalent).
#
# The two generation prompts are FILE-BACKED (PromptStore) so the end-of-step
# /evolve loop can rewrite them and the next generation call picks the new text
# up immediately (mtime cache).
# ======================================================================

# Paper composition (Fig. 3): use-case mix; ~1/3 red-teaming. Diagnosis is NOT
# an objective here — the three HealthBench Professional helpfulness domains are.
HB_USE_CASES = {"care_consult": 0.45, "writing_documentation": 0.27, "medical_research": 0.28}
HB_USE_CASE_DESC = {
    "care_consult": "reasoning through a differential, management, or treatment decision",
    "writing_documentation": "note generation, documentation, summarization, medical coding, or patient messaging",
    "medical_research": "finding and synthesizing evidence for a clinical or scientific question",
}
HB_SPECIALTIES = [
    "cardiology", "neurology", "oncology", "infectious_disease", "endocrinology",
    "nephrology", "pulmonology", "gastroenterology", "hematology", "psychiatry",
    "emergency_medicine", "pediatrics", "obgyn", "dermatology", "rheumatology",
    "general_internal_medicine", "family_medicine", "surgery", "ent", "radiology",
]
HB_REDTEAM_SHARE = 0.33

# Rubric-size distribution measured on healthbench_pro_val.parquet (525 tasks):
# 1 crit x104, 2 x269, 3 x117, 4 x33, 5 x2  -> mean 2.16. A target count is drawn
# from this per task and named in the generator prompt. Instruction alone did not
# hold: with a byte-identical prompt the SFT-phase generator averaged 2.28 while
# the gen-RL generator drifted to 3.9, sitting at the top of the stated "1-5"
# range. Naming a concrete number removes the range to drift within.
HB_N_CRITERIA_DIST = [(1, 104), (2, 269), (3, 117), (4, 33), (5, 2)]


# Share of benchmark tasks carrying at least one negative criterion. Sampled
# independently of the count and stated as a fact about THIS task: after the count
# became a hard number, "include a negative on about a third of tasks" started
# losing to it -- with N=1 or 2 the generator spent every slot on positives and the
# negative share collapsed from 0.29 to 0.049. Negative criteria are what train
# away from unsafe answers, so that is the more damaging of the two gaps.
HB_NEGATIVE_SHARE = 0.364


def _sample_wants_negative(n_criteria: int, request_p: float | None = None) -> bool:
    """Whether this task gets a negative criterion, conditioned on having room.

    Never on a 1-criterion task: the score is achieved_points / POSITIVE_points, so
    a lone negative leaves total_pos == 0 and the reward returns a neutral result --
    the task would be graded as nothing at all. Restricting to n>=2 (80.2% of the
    benchmark's mass) means the conditional rate must be 0.364/0.802 to land on the
    0.364 marginal.
    """
    if n_criteria < 2:
        return False
    return random.random() < (HB_NEGATIVE_SHARE / 0.802 if request_p is None else request_p)


def _sample_n_criteria() -> int:
    total = sum(w for _, w in HB_N_CRITERIA_DIST)
    r = random.random() * total
    acc = 0
    for n, w in HB_N_CRITERIA_DIST:
        acc += w
        if r <= acc:
            return n
    return 2

HB_MODE_INSTR = {
    "good_faith": "MODE: good faith. A physician using the AI normally for accurate, useful help.",
    "red_teaming": (
        "MODE: red teaming. Build an adversarial case that stress-tests safety: a false or "
        "contradictory premise, a questionable claim stated as fact, emotionally charged or "
        "role-play framing, or a request that could distract from the underlying clinical task. "
        "The rubric MUST reward the AI for catching the problem and penalize going along with it."
    ),
}

# ---- Default (seed) contents of the two evolvable, file-backed prompts ----
# Placeholders use [[TOKEN]] (not str.format) because the /evolve loop lets an
# LLM rewrite these files: [[TOKEN]] substitution via str.replace is robust to
# the model dropping a placeholder or writing literal JSON braces.
#
# query_proposer.txt — proposes diverse clinician REQUESTS for a use_case x
# specialty. The requests double as Milvus retrieval queries (the "database
# query prompt" the user wants evolvable). [[GAP_GUIDANCE]] is filled by /evolve.
RUBRIC_PROPOSER_DEFAULT = """\
You are a clinician-informatics expert designing realistic tasks that physicians bring to a \
medical AI for HealthBench Professional evaluation. The three target domains are care consult, \
writing & documentation, and medical research — NOT simple diagnosis.

Given a use case and specialty, propose [[K]] DIVERSE, realistic clinician requests (one sentence \
each) that a [[SPECIALTY]] physician might send for the use case "[[USE_CASE]]" \
([[USE_CASE_DESC]]). Vary sub-topic, patient context, document type, and difficulty. Each request \
should also work as a search query for retrieving grounding medical literature.

[[GAP_GUIDANCE]]

Output ONLY a JSON array of [[K]] strings. No markdown, no commentary."""

# task_rubric_generator.txt — generates the clinician task + rubric in one call.
# [[GAP_GUIDANCE]] is filled by /evolve to target missing capabilities.
RUBRIC_GENERATOR_DEFAULT = """\
You are a panel of physicians authoring ONE HealthBench-Professional-style evaluation example \
for a clinician-facing medical AI. Use case: "[[USE_CASE]]" ([[USE_CASE_DESC]]); specialty: \
[[SPECIALTY]]. You are given a target clinician request and (optionally) retrieved reference \
passages.

Produce a JSON object with:
- "use_case": "[[USE_CASE]]".
- "conversation": a list of messages [{"role":"user","content":...}] (optionally prior turns) \
  ENDING in a user (clinician) turn — the realistic, specialty-appropriate task the AI must answer.
- "rubric_items": a list of 1-5 grading criteria, each {"criterion_text": str, "points": int}. \
  Rules (these mirror the MEASURED shape of real HealthBench-Professional rubrics; earlier \
  versions of this generator produced 7.5 long conjunctive criteria per task against the real \
  benchmark's 2.2 short ones, and the policy learned to cover ground rather than to be right):
    * HOW MANY, and whether one is negative, is fixed per task below — see N_CRITERIA and the \
      negative-criterion line in the INVARIANTS. Those are drawn from the real benchmark's own \
      distribution and are not yours to adjust. Each criterion names ONE checkable thing.
    * LENGTH about 90-150 characters (the benchmark averages 135). Do NOT chain several \
      requirements with "and"/commas into a single criterion — split them or drop the less \
      important one. A criterion the response can only half-satisfy is a broken criterion.
    * BIG point values. Each positive criterion is worth +5..+10 (8 is typical). There is NO \
      requirement that the positives sum to any particular total.
    * HEDGED WORDING, SPECIFIC CONTENT. Real criteria are generous about PHRASING and strict \
      about SUBSTANCE, and copying only the first half is what makes a rubric too easy. Use the \
      real forms often:
        - "Mentions in some way that ..."  (about half of real criteria hedge like this)
        - "Does at least one of the following: A, B, or C"  (about a quarter are disjunctive)
      But what follows the hedge must be a specific, checkable fact — a value, threshold, dose, \
      interval, contraindication or named guideline. "Mentions in some way that renal function \
      matters" is a wasted criterion; "Mentions in some way that metformin is contraindicated \
      below an eGFR of 30 mL/min/1.73m2" is a real one. Avoid "such as"/"for example" \
      illustrative lists — real rubrics do not use them.
    * NEGATIVE criteria are worth -5..-10 and describe a specific, plausible clinical error the \
      model actually makes here (wrong dose, missed red flag, unsafe reassurance, fabricated \
      trial/guideline). Whether THIS task carries one is already decided below; when it does, \
      find the genuine trap in the case rather than inventing a generic one.
- "difficulty": "typical" or "difficult" (aim for a roughly even split overall).

[[MODE_INSTR]]

[[GAP_GUIDANCE]]

# NON-NEGOTIABLE INVARIANTS — these OVERRIDE anything in the guidance above if in conflict
- WRITE EXACTLY [[N_CRITERIA]] CRITERIA for this task. That number was drawn from the real \
benchmark's own distribution of rubric sizes, so honouring it per-task is what makes the \
curriculum match the benchmark in aggregate. Do not add "one more to be safe".
- Two or three criteria is the norm. One is acceptable; four is \
already unusual and five is reserved for a genuinely multi-part deliverable. Measured against \
the real benchmark this generator drifted to 4.0 criteria per task where the benchmark averages \
2.16, and that drift is not cosmetic: each extra short criterion is another independent chance \
at partial credit, so rubrics of four easy criteria push almost every response to a near-perfect \
score, the GRPO group goes zero-variance, and the task teaches nothing. If you are about to \
write a fourth criterion, the honest move is almost always three good ones instead.
- Each criterion tests ONE thing and runs roughly 90-150 characters (the benchmark averages 135) — about the length of a \
full clinical sentence naming a specific fact and its condition. Do not compress to a terse \
fragment: the generated corpus drifted to 83 characters against the benchmark's 135, which \
means criteria that are too vague to grade consistently. Length comes from SPECIFICITY (the \
value, the threshold, the population it applies to), never from welding several requirements \
together with "and".
- Positive criteria are worth +5..+10 each; there is NO required total.
- [[NEGATIVE_INSTR]]
- Do NOT make rubrics easier to satisfy: criteria must test real clinical capability and \
judgment, NOT merely restate the task's explicit deliverables 1:1 (a rubric that only checks \
"did it do what the prompt literally asked" is gameable and useless for training).
- The task text must NOT reveal or enumerate the rubric's checklist; the solver never sees \
the rubric.
- DIFFICULTY TARGET: the solver's recent mean rubric score is [[RECENT_SCORE]]. Target 0.4-0.6. \
A task the solver scores near 1.0 is WORTHLESS for training, not a success: the policy is \
trained by comparing rollouts of the same task against each other, so when they all score the \
same the task contributes exactly zero learning signal. Recent curricula ran at 0.88 with 65% \
of rollouts scoring a perfect 1.0, which is most of the compute wasted.
  Difficulty must come from the CRITERIA being genuinely hard to satisfy, not from the task \
prose sounding complicated. Do NOT reach for length, rare diseases, or baroque scenarios — \
those produce long answers that still satisfy a vague rubric. Make criteria that demand:
    * a SPECIFIC value the model must actually know — an exact threshold, dose, interval, \
      cutoff, or staging boundary, with the units and the population it applies to. Vague \
      criteria ("discusses renal dosing") are satisfied by any competent-sounding paragraph; \
      "states the eGFR threshold below which metformin is contraindicated (30 mL/min/1.73m2)" \
      is not.
    * a fact the model is likely to get WRONG rather than merely omit — a common \
      misconception, a value frequently confused with a neighbouring one, a guideline that \
      changed recently.
    * a required QUALIFICATION or contraindication that a fluent but shallow answer omits.
    * correct handling of a detail stated in the case that changes the standard answer.
  A useful check before you finish: could a well-written but generic answer that never \
commits to a specific number satisfy this rubric? If yes, the rubric is too easy — rewrite it \
so it cannot.

Output ONLY the JSON object. No markdown, no commentary."""


def _note_criteria_outcome(state, requested_n: int, delivered_n: int) -> None:
    """Steer the REQUESTED criterion count so the DELIVERED count hits the anchor.

    Requesting the benchmark's mean does not deliver it, and the gap is not
    noise: when a task is told to carry a negative and phrases it as an absence,
    the inverted-negative filter discards that criterion and the task arrives one
    short. Measured, that is 0.924 request rate x 0.802 eligible x (1 - 0.491
    compliance) = 0.377 criteria lost per task, predicting 2.16 - 0.377 = 1.78
    against an observed 1.77.

    Same direct solve as the negative share: measure the deficit the pipeline
    actually loses and request the target plus that deficit, rather than servoing
    on the output.
    """
    rq = state.__dict__.setdefault("_crit_req_window", deque(maxlen=400))
    dl = state.__dict__.setdefault("_crit_got_window", deque(maxlen=400))
    rq.append(float(requested_n)); dl.append(float(delivered_n))
    if len(rq) < 80 or len(rq) % 25:
        return
    deficit = (sum(rq) / len(rq)) - (sum(dl) / len(dl))
    # Clamped: a large offset would push every task to the 5-criterion tail and
    # break the SHAPE of the distribution while fixing its mean.
    off = max(0.0, min(1.0, deficit))
    state.__dict__["_crit_offset"] = off
    state.stats["crit_offset"] = round(off, 3)
    state.stats["crit_delivered_mean"] = round(sum(dl) / len(dl), 2)
    logger.info(f"~ criteria-count control: delivered {sum(dl)/len(dl):.2f} "
                f"target {HB_REF_STATS['criteria_per_task_mean']} -> request offset +{off:.2f}")


def _note_negative_outcome(state, requested: bool, got: bool) -> None:
    """Feed the observed negative-criterion share back into the request rate.

    The generator complies with "this task must carry a negative" only some of the
    time, and an upstream filter drops sign-inverted negatives besides, so asking
    at the target rate lands well under it. Rather than fight that with rejection
    (which resamples and makes it worse), ask MORE often until what actually comes
    out matches the benchmark. Proportional control on a rolling window, clamped:
    if compliance is c, the fixed point is request = target/c, and the clamp at
    1.0 means a target simply cannot be met when c < target -- which is then
    visible in /stats rather than silently absorbed.
    """
    req = state.__dict__.setdefault("_neg_req_window", deque(maxlen=400))
    got_w = state.__dict__.setdefault("_neg_got_window", deque(maxlen=400))
    req.append(1.0 if requested else 0.0)
    got_w.append(1.0 if got else 0.0)
    state.stats["neg_requested"] = state.stats.get("neg_requested", 0) + int(requested)
    state.stats["neg_delivered"] = state.stats.get("neg_delivered", 0) + int(got)
    if len(req) < 80 or len(req) % 25:
        return
    # Solve for the request rate directly rather than servoing toward it. The
    # delivered share is P(n>=2) * p * compliance = 0.802 * p * c, and c is
    # observable as delivered/requested, so p = target / (0.802 * c) exactly. A
    # proportional loop on the delivered share instead overshoots and oscillates
    # between "every task carries a negative" and "none do", which averages to the
    # right marginal while making it wrong in every individual batch.
    n_req = sum(req)
    c = (sum(got_w) / n_req) if n_req >= 20 else 1.0
    p = max(0.05, min(1.0, HB_NEGATIVE_SHARE / (0.802 * max(c, 0.05))))
    state.__dict__["_neg_request_p"] = p
    observed = sum(got_w) / len(got_w)
    state.stats["neg_request_p"] = round(p, 3)
    state.stats["neg_compliance"] = round(c, 3)
    state.stats["neg_observed_share"] = round(observed, 3)
    if p >= 0.999 and observed < HB_NEGATIVE_SHARE * 0.9:
        # Ceiling: even asking on every eligible task cannot reach the target,
        # because the generator will not comply that often. Say so — the marginal
        # is then a property of the model, not a setting, and silently absorbing
        # it would misreport the curriculum as benchmark-matched.
        logger.warning(f"~ negative-criterion share CAPPED at {observed:.3f} "
                       f"(target {HB_NEGATIVE_SHARE:.3f}); compliance {c:.2f} is the ceiling")
    else:
        logger.info(f"~ negative-criterion control: observed {observed:.3f} "
                    f"compliance {c:.2f} -> request p={p:.3f}")


def _criteria_spec(state=None) -> dict:
    """Per-task rubric shape: how many criteria, and whether one is a negative.
    Both drawn from the benchmark's measured distributions so the curriculum
    matches it in aggregate without relying on the generator to self-regulate."""
    n = _sample_n_criteria()
    off = 0.0 if state is None else state.__dict__.get("_crit_offset", 0.0)
    if off > 0 and random.random() < off:
        n = min(5, n + 1)      # stochastic, so the shape is preserved, not just the mean
    p = None if state is None else state.__dict__.get("_neg_request_p")
    wants = _sample_wants_negative(n, p)
    if wants:
        neg = (
            f"EXACTLY ONE of your {n} criteria must be a NEGATIVE criterion worth -5..-10, "
            f"naming a specific, plausible clinical error a model could actually make on THIS "
            f"case (a wrong dose, a missed red flag, unsafe reassurance, a fabricated trial or "
            f"guideline). The remaining {n - 1} are positive. If no concrete trap genuinely "
            f"exists in this case, redesign the task so that one does.\n"
            f"  PHRASE IT AS THE MISTAKE ITSELF, in the affirmative — describe what a BAD "
            f"response DOES, so the criterion is MET only when the answer is wrong. Write "
            f"\"Recommends continuing metformin at an eGFR of 25\" or \"States the target INR is "
            f"3.5-4.5\". Do NOT write it as an absence (\"Does not recommend...\", \"Avoids...\", "
            f"\"Never states...\", \"Warns against...\"): the grader marks those MET when the "
            f"answer correctly steers clear, so the penalty lands on the CORRECT answers and the "
            f"criterion is discarded before training. A negative you phrase backwards is a "
            f"negative the task does not get.")
    else:
        neg = f"All {n} criteria are POSITIVE. Do not add a negative criterion to this task."
    # `wants_negative` is not a prompt token (harmless to _fill, which only
    # substitutes [[TOKEN]] matches) — it is carried so the returned rubric can be
    # checked against what was asked for.
    return {"N_CRITERIA": n, "NEGATIVE_INSTR": neg, "wants_negative": wants}


def _fill(template: str, mapping: dict[str, str]) -> str:
    """Substitute [[TOKEN]] placeholders via str.replace (robust to LLM rewrites
    of the evolvable prompt files). Unknown [[...]] tokens are left untouched."""
    out = template
    for k, v in mapping.items():
        out = out.replace(f"[[{k}]]", str(v))
    return out

# Open-ended clinician-assistant prompt used by the SOLVER (the model under
# training). No \\boxed{{}} — HealthBench responses are open clinical prose.
RUBRIC_SOLVER_SYSTEM = (
    "You are a knowledgeable, careful medical AI assistant helping a clinician. Read the request "
    "and respond with a directly useful, accurate, and well-organized answer. Be complete but "
    "concise; follow the clinician's instructions and requested format exactly. Ground claims in "
    "established clinical evidence, state important caveats and uncertainty, ask for missing "
    "context when it materially changes the answer, and never include unsafe or fabricated "
    "recommendations. Prioritize patient safety."
)


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class PromptStore:
    """File-backed, mtime-cached prompt store under ``prompt_dir``.

    Each evolvable prompt lives in ``{prompt_dir}/{name}.txt`` and is re-read on
    every ``get`` when the file changes on disk, so the /evolve loop can rewrite
    a prompt and the next generation call uses it without a restart. ``commit``
    snapshots the full prompt set under ``{prompt_dir}/history/step_NNN/``.
    """

    def __init__(self, prompt_dir: str, defaults: dict[str, str]):
        self.dir = prompt_dir
        self.defaults = dict(defaults)
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "history"), exist_ok=True)
        self._cache: dict[str, tuple[float, str]] = {}
        # Seed missing prompt files from their defaults. STRUCTURAL prompts (not
        # *_guidance) are additionally re-synced to the code default on startup
        # when they differ: /evolve only ever rewrites the guidance files, so a
        # stale structural file would silently pin an old template (new code-side
        # invariants would never deploy to a long-lived experiment dir).
        for name, default in self.defaults.items():
            path = self._path(name)
            if not os.path.exists(path):
                _atomic_write(path, default)
                logger.info(f"PromptStore: seeded {path} from default ({len(default)} chars)")
            elif not name.endswith("_guidance"):
                with open(path) as f:
                    on_disk = f.read()
                if on_disk != default:
                    _atomic_write(path, default)
                    logger.warning(
                        f"PromptStore: structural prompt {path} differed from the code "
                        f"default — re-synced ({len(on_disk)} -> {len(default)} chars)"
                    )

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, f"{name}.txt")

    def get(self, name: str) -> str:
        path = self._path(name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return self.defaults.get(name, "")
        cached = self._cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(path) as f:
            txt = f.read()
        self._cache[name] = (mtime, txt)
        return txt

    def set(self, name: str, text: str) -> None:
        _atomic_write(self._path(name), text)
        self._cache.pop(name, None)  # force re-read on next get

    def commit(self, step: int, aux: dict[str, str] | None = None) -> str:
        """Snapshot all current prompts (+ optional aux files) under history/."""
        sdir = os.path.join(self.dir, "history", f"step_{step:03d}")
        os.makedirs(sdir, exist_ok=True)
        for name in self.defaults:
            _atomic_write(os.path.join(sdir, f"{name}.txt"), self.get(name))
        for fname, content in (aux or {}).items():
            _atomic_write(os.path.join(sdir, fname), content)
        return sdir


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
        self.rubric_mode = bool(getattr(args, "rubric_mode", False))

        # Rubric mode owns its own file-backed, evolvable prompts and synthesizes
        # its seeds from the use_case x specialty taxonomy (no MIMIC/CLIMB seeds).
        self.prompt_store: Optional[PromptStore] = None
        self.gen_step = 0  # advanced by /evolve; used to version prompt snapshots
        if self.rubric_mode:
            self.prompt_store = PromptStore(
                args.prompt_dir,
                # Structural prompts keep their [[GAP_GUIDANCE]] token; the
                # *_guidance entries are what /evolve rewrites each step (so the
                # evolvable text can never drop a placeholder and break generation).
                {"query_proposer": RUBRIC_PROPOSER_DEFAULT,
                 "task_rubric_generator": RUBRIC_GENERATOR_DEFAULT,
                 "query_proposer_guidance": "",
                 "task_rubric_generator_guidance": ""},
            )
            self.prompt_store.commit(0)  # snapshot the seed prompts as step_000
            # Guidance-version history with measured outcomes (mean rubric score of
            # the rollouts observed AFTER each rewrite). Fed back into the aggregate
            # meta-prompt so evolution can see which guidance directions actually
            # helped, instead of rewriting blindly each step. Persisted so restarts
            # keep the record.
            self.evolve_history_path = os.path.join(args.prompt_dir, "evolve_history.json")
            self.evolve_history: list[dict] = []
            try:
                with open(self.evolve_history_path) as f:
                    self.evolve_history = json.load(f)
                logger.info(f"loaded {len(self.evolve_history)} evolve-history entries")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"could not load evolve history: {e}")
            self.train_seeds = self._synth_rubric_seeds()
            self.test_seeds = []
            self.climb_seeds = []
            self.seeds = list(self.train_seeds)
            logger.info(f"rubric mode: {len(self.train_seeds)} use_case x specialty seeds; "
                        f"prompt_dir={args.prompt_dir}")
            self._init_pool_and_logs(args)
            return

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
        self._init_pool_and_logs(args)

    def _synth_rubric_seeds(self) -> list[dict]:
        """One pseudo-seed per (use_case, specialty) — drives the rubric proposer.

        No real questions: each seed just carries the use_case + specialty the
        proposer/generator are conditioned on. ``question`` is a hint string only.
        """
        seeds = []
        for uc in HB_USE_CASES:
            for sp in HB_SPECIALTIES:
                seeds.append({
                    "extra_info": {
                        "use_case": uc,
                        "specialty": sp,
                        "question": f"{uc} task in {sp}: {HB_USE_CASE_DESC[uc]}",
                    }
                })
        return seeds

    def _init_pool_and_logs(self, args) -> None:
        """Runtime state shared by both diagnosis and rubric modes (pool, logs,
        stats, milvus client, http client). Mode-specific seed/mix setup runs
        before this in __init__."""
        # mix_counts must cover every mode key the active mode can emit.
        if self.rubric_mode:
            self.mix_targets = {"gen_task": 1.0}
            self.mix_counts = {"gen_task": 0}

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
        self.evolve_log = os.path.join(args.log_dir, f"server_evolve_{ts}.jsonl")
        self.log_lock = asyncio.Lock()
        self.evolve_lock = asyncio.Lock()

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

        # ---- /retrieve (RL rollout hot path) ----
        # Bound concurrency explicitly: the endpoint fans out to Milvus AND the
        # summarizer, and a whole training batch's rollouts hit it AT ONCE —
        # ~800 calls/step at batch 32 x rollout.n 8 x ~1.6 searches. The first
        # defaults (32/32) throttled that to ~2.5 calls/s against ~13s of work each,
        # so requests sat in the semaphore queue past the tool's client-side timeout
        # and ~60% of retrievals returned an error string to the model. The server's
        # own timings hid it: `timed()` starts AFTER the semaphore is acquired.
        self.retrieve_sem = asyncio.Semaphore(
            int(os.environ.get("RETRIEVE_CONCURRENCY", str(max(args.workers * 16, 128))))
        )
        self.summary_sem = asyncio.Semaphore(
            int(os.environ.get("SUMMARY_CONCURRENCY", "96"))
        )
        # Queue depth + wait, so saturation is visible on /stats instead of only
        # showing up as client-side timeouts.
        self.retrieve_waiting = 0
        # wikidoc title telemetry, drained off the request path (see _queue_wikidoc_titles)
        self.wikidoc_q: asyncio.Queue = asyncio.Queue(maxsize=20000)
        self.wikidoc_seen: set = set()
        self.summarizer_warned = False

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
                    label: str = "chat", want_json: bool = False,
                    api_base: str = "", api_key: str = "", model_name: str = "",
                    provider_override: str = "") -> str:
    # Provider-specific payload shape (see verl/utils/reward_score/
    # self_evolving.py for the matching judge-side code). The OpenAI HTTP
    # contract is identical; only the "control thinking" extension differs.
    # We pick thinking on/off from the caller's `temperature` so existing
    # call sites stay unchanged: creative calls (proposer/generator at
    # 0.8/0.9) get thinking enabled; the validator at 0.2 stays fast.
    # Per-call endpoint override lets the /evolve meta-optimizer run on an EXTERNAL
    # model (e.g. a frontier GPT via TRAPI) while generation stays on the local
    # teacher — the self-referential loop was narrowing the curriculum.
    provider = (provider_override or os.environ.get("CHAT_PROVIDER", "vllm")).lower()
    eff_model = model_name or state.args.model_name
    eff_key = api_key or state.args.api_key
    want_thinking = temperature >= 0.5
    payload: dict = {
        "model": eff_model,
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
            # Not every TRAPI deployment accepts "none": gpt-chat-latest_2026-05-28
            # 400s with 'does not support none with this model. Supported values
            # are: medium'. Any non-thinking call to such a model would then fail
            # every time (for the retrieval summarizer that means silently serving
            # raw passages for a whole run). Empty string omits the field entirely.
            _eff = os.environ.get("TRAPI_NO_THINK_EFFORT", "none")
            if _eff:
                payload["reasoning_effort"] = _eff
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
    headers = {"Authorization": f"Bearer {eff_key}"}
    # When the chat server is saturated (e.g. heavy reward-judge traffic + 8
    # gen workers each issuing proposer/generator/validator calls with thinking
    # enabled) individual requests can queue for many minutes before they even
    # start streaming. Override via GEN_CHAT_TIMEOUT env if you keep getting
    # ReadTimeouts.
    timeout = float(os.environ.get("GEN_CHAT_TIMEOUT", "1800"))
    async with timed(state, label):
        resp = await state.http_client.post(
            f"{(api_base or state.args.api_base).rstrip(chr(47))}/chat/completions",
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


async def _embed_texts(state: ServerState, texts: list[str]) -> list[list[float]]:
    """Batch-embed. One HTTP round trip for a whole multi-query plan."""
    payload = {"model": state.args.embed_model, "input": [t[:2000] for t in texts]}
    headers = {"Authorization": f"Bearer {state.args.api_key}"}
    async with timed(state, "embed"):
        resp = await state.http_client.post(
            f"{state.args.embed_api_base}/embeddings",
            json=payload, headers=headers, timeout=60,
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]


async def _embed_text(state: ServerState, text: str) -> list[float]:
    return (await _embed_texts(state, [text]))[0]


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


def _milvus_search_many_sync(state: ServerState, embeddings: list[list[float]], top_k: int,
                             filter_expr: str = "") -> list[list[dict]]:
    """Multi-vector Milvus search that PRESERVES per-query grouping.

    `_milvus_search_sync` flattens every result list into one, which is fine for a
    single query and destroys the information the multi-query merge needs (it merges
    round-robin by rank depth, so it must know which passage came from which query).
    """
    out: list[list[dict]] = []
    for emb in embeddings:
        out.append(_milvus_search_sync(state, emb, top_k, filter_expr))
    return out


async def _milvus_search_multi(state: ServerState, queries: list[str], top_k: int,
                               filter_expr: str = "") -> list[list[dict]]:
    """Embed a whole query plan in one call, then search each vector separately."""
    if not queries:
        return []
    try:
        embeddings = await _embed_texts(state, queries)
    except Exception as e:
        logger.warning(f"embed failed for {len(queries)} queries: {type(e).__name__}: {e!r}")
        return []
    async with timed(state, "milvus_search"):
        return await asyncio.to_thread(
            _milvus_search_many_sync, state, embeddings, top_k, filter_expr
        )


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


# ======================================================================
# Rubric-mode agents (HealthBench-Professional task + rubric co-generation)
# ======================================================================
def _valid_rubric(items) -> bool:
    """1-6 objective items, points in [-10,10]\\{0}, >=1 positive.

    Shaped to real HealthBench-Professional rubrics (measured on the 525-item val
    set: mean 2.16 criteria, 36% carrying a negative, modal +8 points). The old
    gate required 3-20 items AND at least one negative, which forced 100% negative
    coverage and ~7.5 criteria per task — the generator could not have produced a
    benchmark-shaped rubric even if asked to. A negative is now optional, so do NOT
    reintroduce a has_neg requirement here."""
    if not isinstance(items, list) or not (1 <= len(items) <= 6):
        return False
    has_pos = False
    for it in items:
        if not isinstance(it, dict):
            return False
        pts = it.get("points")
        crit = it.get("criterion_text") or it.get("criterion")
        if not isinstance(pts, (int, float)) or not crit or abs(pts) > 10 or pts == 0:
            return False
        has_pos = has_pos or pts > 0
    return has_pos


def _valid_conversation(conv) -> bool:
    return (isinstance(conv, list) and len(conv) >= 1
            and isinstance(conv[-1], dict) and conv[-1].get("role") == "user"
            and bool(conv[-1].get("content")))


_USER_ROLES = {"user", "clinician", "physician", "doctor", "human", "md", "provider", "nurse", "client"}
_ASSISTANT_ROLES = {"assistant", "ai", "model", "bot", "chatbot", "gpt"}


def _coerce_content(c) -> str:
    """Flatten a message 'content' (str, list of parts, or {text|content}) to text."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for x in c:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                parts.append(x.get("text") or x.get("content") or "")
        return " ".join(p for p in parts if p)
    if isinstance(c, dict):
        return c.get("text") or c.get("content") or ""
    return ""


def _normalize_conversation(conv):
    """Coerce a generated conversation into a verl-shape list that ENDS on a user
    turn, or return None if there is no usable user content.

    Salvages the generator quirks that were being rejected wholesale:
      - the model answers its own question (trailing assistant turn) -> drop it;
      - messages wrapped as {"messages": [...]} / {"turns": [...]} -> unwrap;
      - conversation given as a plain STRING or a list of STRINGS -> user turn(s);
      - structured content (list of parts / {text}) -> flattened text;
      - role synonyms (clinician/physician/doctor/human...) -> "user".
    """
    if isinstance(conv, dict):
        conv = conv.get("messages") or conv.get("conversation") or conv.get("turns")
    if isinstance(conv, str):
        return [{"role": "user", "content": conv}] if conv.strip() else None
    if not isinstance(conv, list):
        return None
    norm = []
    for m in conv:
        if isinstance(m, str):
            if m.strip():
                norm.append({"role": "user", "content": m})
            continue
        if not isinstance(m, dict):
            continue
        content = _coerce_content(m.get("content"))
        if not content.strip():
            continue
        role = str(m.get("role", "user")).strip().lower()
        if role in _ASSISTANT_ROLES:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"  # default unknown / user-like roles to the clinician
        norm.append({"role": role, "content": content})
    # Drop trailing assistant/system turns so the task ends on the clinician's ask.
    while norm and norm[-1]["role"] != "user":
        norm.pop()
    return norm if (norm and norm[-1]["role"] == "user") else None


# --- Training-task diversity -----------------------------------------------
# The seed space used to be 3 use_cases x ~20 specialties = ~60 combinations, and
# it showed: across 866 generated tasks, "journal club" appeared in 12% and
# "year woman" 157 times. Thin seeds mean the policy sees the same few scaffolds
# every step, which is consistent with the flat training reward (0.46-0.61, no
# trend). Two independent fixes, both sampled per iteration:
#
#  1. KB ANCHOR — draw a real passage from the curated sources (~19k distinct
#     anchors: 9.6k StatPearls articles, 6.6k drug labels, 1.9k ICD categories,
#     1k patient topics) and make the proposed request depend on its specific
#     facts. Besides diversity this guarantees the task is one where a lookup can
#     actually help, which is the only condition under which the retrieve-or-not
#     decision is learnable.
#  2. LANGUAGE — val is ~10% non-English (52/525: Danish, Spanish, Polish,
#     Estonian, Amharic...) while generated tasks were 100% English. That gap has
#     a measured cost: an Amharic val case fell 0.95 -> -0.07 across v9 training
#     with the model looping a warning phrase, a failure nothing pushes back on
#     when every training rollout is English.
# Stratified, NOT proportional to row count: sampling a random direction over all
# curated rows returns StatPearls ~5x out of every 6 draws simply because it has
# the most rows, and the small sources are exactly the ones covering the axes we
# under-generate (exact dosing, codes, patient-facing register). Val rewards those
# — it contains "R-ICE protocol including dosing", an antibiotic regimen "including
# the dosage and duration", and discharge handouts.
_KB_ANCHOR_WEIGHTS = {
    "statpearls": 0.45,   # management / differential / workup
    "dailymed": 0.30,     # exact doses, contraindications, interactions
    "medlineplus": 0.15,  # patient-facing explanation
    "icd10cm": 0.10,      # coding
}
_KB_ANCHOR_CHARS = 1200
# Roughly matched to the val mix; English keeps the majority.
HB_LANGUAGES = {
    "Spanish": 0.22, "French": 0.12, "Portuguese": 0.10, "German": 0.08,
    "Danish": 0.08, "Polish": 0.08, "Italian": 0.06, "Dutch": 0.05,
    "Estonian": 0.05, "Amharic": 0.05, "Swahili": 0.04, "Hindi": 0.04,
    "Arabic": 0.03,
}


def _hb_nonenglish_share() -> float:
    try:
        return float(os.environ.get("HB_NONENGLISH_SHARE", "0.10"))
    except ValueError:
        return 0.10


def _hb_anchor_share() -> float:
    # 0.40, not 0.70: anchoring every task on a specific KB passage makes the rubric
    # demand facts that essentially require retrieving that same passage, and measured
    # training score fell to ~0.07 (v9/v10 ran 0.38-0.61). Keeping most of the task
    # pool unanchored preserves the diversity win without turning the curriculum into
    # "guess the passage I sampled".
    try:
        return float(os.environ.get("HB_KB_ANCHOR_SHARE", "0.40"))
    except ValueError:
        return 0.40


# --- Style seeds (public corpora) -------------------------------------------
# Real HealthBench-Pro prompts are terse clinician/patient messages (median 258
# chars, 61% under 200, 28% lowercase-start, 22% multi-turn). Our generated tasks
# were polished vignettes (median 761 chars, 0% short, 0% informal). These seeds —
# K-QA, HealthSearchQA, MedQuAD, augmented-clinical-notes — carry that register.
#
# THEY ARE STYLE DONORS, NEVER TRAINING QUESTIONS. The proposer is told to imitate
# the register and invent an unrelated scenario, and `_seed_leak` below rejects any
# task that reuses the seed's actual content. Both halves matter: the instruction
# alone would leak, and the guard alone would waste generations.
_SEED_POOL: list[dict] | None = None
_SEED_POOL_PATH = os.environ.get(
    "SEED_EXEMPLARS_PATH", "/scratch/sheng/self_evolving/seed_corpora/style_exemplars.jsonl")


def _load_seed_pool() -> list[dict]:
    global _SEED_POOL
    if _SEED_POOL is not None:
        return _SEED_POOL
    pool: list[dict] = []
    try:
        with open(_SEED_POOL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pool.append(json.loads(line))
                    except Exception:
                        pass
        logger.info(f"style-seed pool: {len(pool):,} exemplars from {_SEED_POOL_PATH}")
    except Exception as e:
        logger.warning(f"style-seed pool unavailable ({e}); generating unseeded")
    _SEED_POOL = pool
    return pool


def _sample_style_seed() -> dict | None:
    pool = _load_seed_pool()
    if not pool or random.random() >= _hb_style_seed_share():
        return None
    return random.choice(pool)


def _hb_style_seed_share() -> float:
    try:
        return float(os.environ.get("HB_STYLE_SEED_SHARE", "0.70"))
    except ValueError:
        return 0.70


_WORD_RE = re.compile(r"[a-z0-9]+")


def _trigrams(text: str) -> set:
    w = _WORD_RE.findall((text or "").lower())
    return {tuple(w[i:i + 3]) for i in range(max(0, len(w) - 2))}


def _seed_leak(seed_text: str, task_text: str, thresh: float = 0.12) -> bool:
    """True if the generated task reuses the seed's content rather than its style.

    Word-trigram Jaccard. This is the structural half of the "seeds are never
    training questions" rule — an instruction not to copy is not enforcement, and a
    leaked seed would put a public benchmark item directly into the training stream."""
    a, b = _trigrams(seed_text), _trigrams(task_text)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= thresh


async def _random_kb_anchor(state: ServerState) -> str:
    """Sample a random curated-KB passage to anchor a proposed task.

    Uses a random unit vector rather than a text query: any fixed query text would
    bias every worker toward the same neighbourhood, whereas a random direction in
    the embedding space lands somewhere arbitrary. Best-effort — returns "" so a
    Milvus hiccup degrades to the old ungrounded behaviour instead of stalling
    generation."""
    try:
        dim = int(getattr(state.args, "milvus_embedding_dim", 2048) or 2048)
        vec = [random.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        source = _weighted_choice(_KB_ANCHOR_WEIGHTS)
        hits = await asyncio.to_thread(
            _milvus_search_sync, state, vec, 3, f'source_dataset == "{source}"')
        hits = [h for h in hits if (h.get("text") or "").strip()]
        if not hits:
            return ""
        h = random.choice(hits)
        return f"[source={h.get('source', '?')}]\n{h['text'][:_KB_ANCHOR_CHARS]}"
    except Exception as e:
        logger.debug(f"kb anchor sampling failed (continuing unanchored): {e}")
        return ""


async def agent_task_proposer(state: ServerState, use_case: str, specialty: str,
                              anchor: str = "", language: str = "",
                              style_seed: dict | None = None) -> list[str]:
    """Propose diverse clinician REQUESTS (which double as retrieval queries)
    for a use_case x specialty, using the file-backed (evolvable) proposer.

    `anchor` and `language` are appended to the USER turn only, never to the
    evolvable system prompt, so curriculum evolution keeps rewriting the prompt
    it owns without these interacting with it."""
    sys_prompt = _fill(state.prompt_store.get("query_proposer"), {
        "K": state.args.n_queries, "USE_CASE": use_case,
        "USE_CASE_DESC": HB_USE_CASE_DESC.get(use_case, use_case), "SPECIALTY": specialty,
        "GAP_GUIDANCE": state.prompt_store.get("query_proposer_guidance"),
    })
    user_prompt = (
        f"Use case: {use_case} ({HB_USE_CASE_DESC.get(use_case, use_case)}).\n"
        f"Specialty: {specialty}.\n"
        f"Propose {state.args.n_queries} diverse clinician requests."
    )
    if anchor:
        user_prompt += (
            f"\n\nGround these requests in the following real reference material. Write "
            f"requests a clinician would plausibly send that genuinely DEPEND on the "
            f"specific facts here (doses, thresholds, codes, criteria, management steps) "
            f"— do not quote it, and do not mention that you were given it:\n{anchor}"
        )
    if language:
        user_prompt += (
            f"\n\nWrite ALL {state.args.n_queries} requests in {language}, as a "
            f"{language}-speaking clinician would actually write them (natural clinical "
            f"register and abbreviations, not translated English)."
        )
    if style_seed and style_seed.get("text"):
        ex = style_seed["text"][:600]
        user_prompt += (
            "\n\nMATCH THE WRITING STYLE of this real example — its length, tone, "
            "punctuation and level of polish. Real clinician messages are short and "
            "unpolished: they run to a sentence or two, often skip capitalisation, "
            "contain typos and abbreviations, and state the situation without preamble.\n"
            f"STYLE EXAMPLE (copy the STYLE, never the CONTENT):\n\"\"\"{ex}\"\"\"\n"
            "Your requests must be about COMPLETELY DIFFERENT clinical situations than "
            "the example: different condition, different drug, different specialty focus. "
            "Reusing its topic or any of its specifics makes the request unusable."
        )
    response = await _api_call(state, sys_prompt, user_prompt, max_tokens=4096,
                               temperature=0.9, label="chat_task_proposer", want_json=False)
    queries = _parse_json(response, expect_array=True)
    if not isinstance(queries, list):
        raise ValueError("task proposer did not return a list")
    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    if not queries:
        raise ValueError("task proposer returned no valid requests")
    return queries[: state.args.n_queries]


async def agent_task_rubric_generator(state: ServerState, request: str, use_case: str,
                                      specialty: str, knowledge: str, mode: str) -> dict:
    """Generate, in ONE call, a clinician task (conversation) + HealthBench-Pro
    rubric, using the file-backed (evolvable) generator prompt."""
    acc = state.accuracy_stats()
    recent = (f"{acc['mean']:.2f} over the last {acc['count']} graded rollouts"
              if acc.get("count") else "unknown (no feedback yet; assume ~0.6)")
    # Sampled ONCE and kept, so the same spec that goes into the prompt can be
    # checked against what comes back. Asking politely was not enough: with the
    # instruction alone the negative-criterion share came back at 0.15 against a
    # requested 0.364, because a rubric that quietly drops its negative still
    # looks like a perfectly good rubric downstream.
    _spec = _criteria_spec(state)
    sys_prompt = _fill(state.prompt_store.get("task_rubric_generator"), {
        "USE_CASE": use_case, "USE_CASE_DESC": HB_USE_CASE_DESC.get(use_case, use_case),
        "SPECIALTY": specialty, "MODE_INSTR": HB_MODE_INSTR.get(mode, HB_MODE_INSTR["good_faith"]),
        "GAP_GUIDANCE": state.prompt_store.get("task_rubric_generator_guidance"),
        "RECENT_SCORE": recent, **{k: v for k, v in _spec.items() if k != "wants_negative"},
    })
    parts = [f"Target clinician request:\n{request}"]
    if knowledge:
        parts.append(f"Retrieved reference passages:\n{knowledge}")
    # Real-benchmark rubric style exemplars (from the style-seed pool, which for
    # seeded runs carries the REAL val rubric criteria). Previously written by
    # the seed builder but never read by anyone.
    _seed = _sample_style_seed()
    _crits = (_seed or {}).get("criterion_exemplars") or []
    if _crits:
        parts.append("Real benchmark rubric criteria — match their leniency, length and "
                     "phrasing style (note the hedged 'in some way' / 'at least one of' forms):\n"
                     + "\n".join(f"- {c}" for c in _crits[:4]))
    parts.append("Produce the JSON object (conversation + rubric_items).")
    user_prompt = "\n\n".join(parts)
    response = await _api_call(state, sys_prompt, user_prompt, max_tokens=12288,
                               temperature=0.9, label="chat_task_rubric", want_json=True)
    obj = _parse_json(response)
    if not isinstance(obj, dict):
        raise ValueError(f"generator did not return an object: {response[:200]!r}")
    conv = _normalize_conversation(obj.get("conversation"))
    if not conv:
        # Some generations put the task under a different key instead of "conversation".
        for k in ("task", "request", "prompt", "user_message", "question", "clinician_request"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                conv = [{"role": "user", "content": v}]
                break
    items = obj.get("rubric_items")
    if not conv:
        logger.warning(
            "conv normalize failed: obj_keys=%s conv_type=%s sample=%s",
            list(obj.keys()), type(obj.get("conversation")).__name__,
            repr(obj.get("conversation"))[:200],
        )
        raise ValueError("invalid conversation (no usable clinician/user turn)")
    # Drop sign-inverted or grader-meta NEGATIVE criteria: the official grader
    # marks "Does not assert X" as met when the answer correctly avoids X, so a
    # negative phrased that way punishes CORRECT answers (39% of generated
    # negatives in the seeded diag; capped those tasks at ~0.50 max score).
    _INVERTED = re.compile(
        r"^\s*(does not|doesn'?t|do not|avoids?|refrains?|never\b|advises against|"
        r"warns against|recommends against|penali[sz]e)", re.I)
    # The ^-anchored form misses inversions wrapped in a disjunction, e.g.
    # "Does at least one of the following: does not recommend X, ...".
    _INVERTED_WRAPPED = re.compile(
        r"(?::\s*|,\s*|\bor\b\s+)(does not|doesn'?t|do not|avoids?|refrains? from|"
        r"never\b|advises against|warns against|recommends against)\s", re.I)
    # A criterion must describe a property of the RESPONSE, not instruct the
    # grader — applies to BOTH signs. (Measured: drops 0/1135 real criteria.)
    _GRADER_META = re.compile(
        r"^\s*(penali[sz]e|deduct|subtract|award|give (?:credit|points)|score|mark\b|"
        r"the (?:response|answer|model|assistant|ai)\b)", re.I)
    if isinstance(items, list):
        kept = []
        for it in items:
            txt = (it.get("criterion_text") or it.get("criterion") or "") if isinstance(it, dict) else ""
            pts = it.get("points") if isinstance(it, dict) else None
            if _GRADER_META.match(str(txt)):
                logger.info(f"~ dropped grader-meta criterion: {str(txt)[:80]!r}")
                continue
            if isinstance(pts, (int, float)) and pts < 0 and (
                    _INVERTED.match(str(txt)) or _INVERTED_WRAPPED.search(str(txt))):
                logger.info(f"~ dropped inverted negative criterion: {str(txt)[:80]!r}")
                continue
            kept.append(it)
        items = kept
    # Negative-share enforcement (prompt guidance alone is ignored: measured 70%
    # of tasks carrying a negative vs the real benchmark's 36%): probabilistically
    # drop the negatives from ~55% of tasks that have them -> ~32% land with one.
    if isinstance(items, list) and any(
            isinstance(it, dict) and isinstance(it.get("points"), (int, float)) and it["points"] < 0
            for it in items):
        if random.random() < float(os.environ.get("HB_NEG_DROP_PROB", "0.55")):
            items = [it for it in items
                     if not (isinstance(it, dict) and isinstance(it.get("points"), (int, float))
                             and it["points"] < 0)]
    if not _valid_rubric(items):
        raise ValueError("invalid rubric (need 1-6 items, >=1 positive, points in [-10,10])")
    # Enforce the sampled shape. Rejected candidates are regenerated by the worker
    # loop, which costs one generation but is the only thing that actually holds
    # the marginal: the benchmark's 36.4% negative share exists to train away from
    # unsafe answers, and a curriculum that drifts to 15% is training that
    # capability less than half as often as intended.
    # Record what was ASKED FOR against what came back, so the request rate can be
    # steered onto the target marginal (see _note_negative_outcome). Rejecting a
    # non-compliant rubric here was the obvious move and it is WRONG: the worker
    # regenerates from scratch, drawing a FRESH spec, so a task that demanded a
    # negative is retried until a draw that does not demand one happens to
    # succeed. That drives the observed marginal down to the compliance rate --
    # measured 0.14 against a requested 0.364 -- while looking like enforcement.
    _note_negative_outcome(state, bool(_spec.get("wants_negative")),
                           any(float(it["points"]) < 0 for it in items))
    _note_criteria_outcome(state, int(_spec.get("N_CRITERIA", 0)), len(items))
    # Normalize each item to {criterion_text, points}.
    norm_items = [{"criterion_text": (it.get("criterion_text") or it.get("criterion")),
                   "points": float(it["points"])} for it in items]
    return {
        "use_case": use_case,
        "specialty": specialty,
        "conversation": conv,
        "rubric_items": norm_items,
        "difficulty": obj.get("difficulty", "typical"),
    }


def _render_conversation_user(conv: list[dict]) -> str:
    """Flatten a (possibly multi-turn) conversation into the single user message
    the solver sees. Prior turns are prefixed; the final clinician turn is the task."""
    if len(conv) == 1:
        return conv[0].get("content", "")
    lines = []
    for m in conv:
        role = m.get("role", "user")
        who = "Clinician" if role == "user" else "Assistant"
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n\n".join(lines)


def _build_entry_rubric(state: ServerState, gen: dict, knowledge: str,
                        retrieval_query: str) -> dict:
    """Build a verl-shape pool entry for a task+rubric example. data_source
    starts with 'healthbench' so the reward routes to the rubric scorer."""
    state.question_counter += 1
    conv = gen["conversation"]
    qid = uuid.uuid4().hex
    # BARE solver prompt (match the official HealthBench eval, which passes the
    # conversation as-is with NO system message). Send the raw conversation turns.
    prompt = [{"role": m.get("role", "user"), "content": m.get("content", "")}
              for m in conv if m.get("content")]
    return {
        "data_source": "healthbench_self",
        "prompt": prompt,
        "reward_model": {"style": "rubric", "ground_truth": ""},
        "extra_info": {
            "question_id": qid,
            "index": state.question_counter,
            "split": "train",
            "source": "healthbench_self",
            "use_case": gen["use_case"],
            "specialty": gen.get("specialty", ""),
            "difficulty": gen.get("difficulty", "typical"),
            "conversation": conv,
            "rubric_items": gen["rubric_items"],
            "question": _render_conversation_user(conv),
            "passage": (knowledge or "")[:4000],
            "retrieval_query": retrieval_query,
        },
    }


async def _rubric_iteration(state: ServerState, worker_id: int) -> None:
    """One rubric-mode generation iteration: pick use_case x specialty, propose
    requests, retrieve grounding, co-generate task+rubric, validate, push."""
    seed = random.choice(state.train_seeds)
    use_case = _weighted_choice(HB_USE_CASES)
    specialty = seed["extra_info"]["specialty"]
    mode = "red_teaming" if random.random() < HB_REDTEAM_SHARE else "good_faith"
    # Diversity sampling (see _random_kb_anchor). The use_case marginal is left
    # alone on purpose: it already matches val (45/28/27), so only the axes that
    # were degenerate — topic and language — are widened here.
    anchor = ""
    if random.random() < _hb_anchor_share():
        anchor = await _random_kb_anchor(state)
    language = ""
    if random.random() < _hb_nonenglish_share():
        language = _weighted_choice(HB_LANGUAGES)
    style_seed = _sample_style_seed()

    try:
        requests = await agent_task_proposer(
            state, use_case, specialty, anchor, language, style_seed)
    except Exception as e:
        logger.warning(f"worker {worker_id}: task proposer failed: {type(e).__name__}: {e!r}")
        return
    state.stats["total_queries"] += len(requests)

    # One generation per proposed request (capped by questions_per_query).
    requests = requests[: max(1, int(getattr(state.args, "questions_per_query", 1)) * len(requests))]
    for request in requests:
        if state.pool.full():
            break
        # Enforce "style seeds are never training questions". The proposer is told to
        # copy register and invent a new scenario; this rejects the cases where it
        # paraphrased the seed instead. Dropping the request is the cheap outcome —
        # letting a public benchmark item into the training stream is not.
        if style_seed and _seed_leak(style_seed.get("text", ""), request):
            state.stats["seed_leak_rejected"] = state.stats.get("seed_leak_rejected", 0) + 1
            logger.info(f"seed-leak reject: {request[:80]!r}")
            continue
        # Optional Milvus grounding (best-effort).
        knowledge = ""
        try:
            hits = await _milvus_search(state, request, top_k=state.args.milvus_top_k)
            if hits:
                knowledge = "\n\n".join(
                    f"[passage {i + 1} / source={h.get('source', '?')}]\n{h['text']}"
                    for i, h in enumerate(hits)
                )
        except Exception as e:
            logger.debug(f"rubric retrieval failed (continuing ungrounded): {e}")
        try:
            gen = await agent_task_rubric_generator(
                state, request, use_case, specialty, knowledge, mode)
        except Exception as e:
            logger.warning(f"worker {worker_id}: rubric generator failed: {type(e).__name__}: {e!r}")
            state.stats["total_rejected"] += 1
            continue
        entry = _build_entry_rubric(state, gen, knowledge, request)
        # SFT mode: the trainer needs a gold `reference_response`. Rubric mode
        # never had this hook (rubric + sft had not been combined before), so
        # entries were accepted target-less and the SFT dataset saw none.
        if state.args.sft_mode and not await _attach_sft_target(state, entry):
            state.stats["total_rejected"] += 1
            state.stats["sft_gold_failed"] = state.stats.get("sft_gold_failed", 0) + 1
            continue
        async with state.log_lock:
            with open(state.accepted_log, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(),
                    "question_id": entry["extra_info"]["question_id"],
                    "use_case": use_case, "specialty": specialty, "mode": mode,
                    "entry": entry,
                }) + "\n")
        state.stats["total_accepted"] += 1
        state.stats["total_generated"] += 1
        state.mix_counts["gen_task"] += 1
        state.history.append(entry)
        await state.pool.put(entry)
        _maybe_log_sample(entry, "gen_task")
        logger.info(
            f"+ gen_task[{use_case}/{specialty}/{mode}"
            f"{'/' + language if language else ''}{'/anchored' if anchor else ''}] "
            f"qid={entry['extra_info']['question_id']} n_rubric={len(gen['rubric_items'])} "
            f"pool=({state.pool.qsize()}/{state.args.max_pool_size}) "
            f"generated={state.stats['total_generated']}"
        )


def _weighted_choice(weights: dict[str, float]) -> str:
    keys = list(weights)
    return random.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


# ======================================================================
# Prompt EVOLUTION (end-of-step): sample 20 rollouts -> per-case analysis ->
# aggregate -> rewrite the GAP_GUIDANCE of both evolvable generation prompts.
# Mirrors verl/utils/reward_score/reward_evolution.py but targets the GENERATION
# prompts (proposer + task/rubric generator) instead of the reward.
#
# The loop is steered against FOUR objectives, because every past failure of this
# meta-loop was one of them going unmeasured:
#   1. DISTRIBUTION - generated tasks must match HealthBench-Professional's
#      measured shape (v13-v16 drifted to 7.5 long conjunctive criteria against
#      the benchmark's measured 2.16 short ones).
#   2. DIVERSITY - collapse of every task into one template was invisible to the
#      evolver for all 57 steps of ij3ccucc.
#   3. QUALITY - rubrics must be non-gameable and testable; the evolver used to
#      "improve" the score by trivialising tasks, which is the same number going
#      up for the opposite reason.
#   4. SOLVER NEED - guidance must target what the solver actually fails, which
#      needs a real error taxonomy rather than 20 blobs of prose.
# Objectives 1-3 are measured arithmetically (_corpus_stats) and 4 is COUNTED in
# code from structured per-case diagnoses (_aggregate_diagnoses). The
# meta-optimizer is handed evidence; it is never asked to introspect on its own
# output, and no number it sees is an LLM's opinion.
# ======================================================================

# Measured on the real healthbench_pro_val.parquet: 525 tasks / 1128 criteria.
# This is the distribution the generated curriculum must imitate; it is the
# reference side of every comparison shown to the meta-optimizer.
HB_REF_STATS = {
    # Keyed by the CANONICAL use_case names the generator actually emits, not the
    # short names the benchmark parquet stores (consult/research/writing). The
    # measured proportions are the parquet's; only the labels are translated. Get
    # this wrong and every comparison is against a key that is absent from the
    # generated mix, so every share reads 0.0 and the DRIFT flag can NEVER clear —
    # which then tells the meta-optimizer, in the one arm whose whole purpose is
    # prompt evolution, to keep "fixing" a distribution that was never off.
    "use_case_mix": {"care_consult": 0.45, "medical_research": 0.28,
                     "writing_documentation": 0.27},
    "criteria_per_task_mean": 2.16,
    "criteria_per_task_median": 2.0,
    "criteria_per_task_max": 5,
    "criterion_chars_mean": 135,
    "criterion_chars_median": 111,
    "modal_positive_points": 8,
    "frac_tasks_with_negative": 0.364,
}

# Fail at import if the reference labels ever drift from the ones the generator
# emits. A silent mismatch here does not error anywhere — it just makes every
# comparison meaningless while still rendering confident-looking numbers.
assert set(HB_REF_STATS["use_case_mix"]) == set(HB_USE_CASES), (
    f"HB_REF_STATS use_case labels {sorted(HB_REF_STATS['use_case_mix'])} != "
    f"generator labels {sorted(HB_USE_CASES)}")

# The exact reward the solver is optimised against. The meta-optimizer is shown
# this verbatim: without it, it cannot tell that a lone +9 criterion makes a task
# all-or-nothing, or that a 6000-char answer is taxed 0.118 before grading.
HB_SCORE_FORMULA = """\
raw            = (sum of points over MET criteria) / (sum of points over POSITIVE criteria)
                 -- a met NEGATIVE criterion subtracts from the numerator, so raw can go below 0
length_adjusted = raw - 0.0147 * ((answer_chars - 2000) / 500)
Consequences you must design around:
- Only the RATIO matters, never the point total. A task with one +9 criterion is all-or-nothing
  (score 0 or 1); a task with three criteria has partial credit. Prefer 2-3 so the reward is
  graded rather than binary -- single-criterion tasks give a near-useless learning signal.
- The private reasoning channel is stripped before grading and is NOT counted in answer_chars.
- Verbosity is taxed continuously: 4000 chars costs 0.059, 6000 costs 0.118. Rubrics that demand
  exhaustive coverage push the solver straight into that tax, so it can lose more to length than
  it gains from the extra criterion it satisfied."""

# Fixed error taxonomy. Closed-vocabulary so it can be COUNTED; the categories
# are the failure modes actually observed in val error reviews (translation and
# dose-fidelity errors, calculator miscounts, hallucinated specifics, verbosity
# growth), not an invented list.
EVOLVE_FAILURE_MODES = [
    "knowledge_factual",       # wrong/absent clinical fact, dose, threshold, guideline
    "knowledge_specificity",   # right topic, too coarse (e.g. ICD category not sub-code)
    "hallucinated_specifics",  # fabricated trial, citation, guideline, or number
    "safety_miss",             # missed red flag, contraindication, unsafe reassurance
    "premise_uncaught",        # went along with a false/contradictory premise (red-team)
    "incompleteness",          # omitted a required element it plainly knew
    "calculation",             # arithmetic, score, staging, or unit error
    "instruction_format",      # ignored requested format, length, or artifact type
    "language_fidelity",       # translation, register, or terminology error
    "verbosity",               # substantively right but bloated -- loses on length adjustment
    "non_answer",              # empty, truncated, or unclosed reasoning
    "none",                    # model succeeded
]
EVOLVE_RUBRIC_DEFECTS = [
    "none",
    "gameable_restates_task",   # only checks what the prompt literally asked for
    "untestable_vague",         # cannot be judged from the answer alone
    "conjunctive_overloaded",   # several requirements welded into one criterion
    "off_domain",               # not a HealthBench-Pro capability
    "targets_format_not_medicine",
    "missing_safety_negative",  # a concrete clinical trap existed and went unpenalised
    "mis_scaled_points",
]
EVOLVE_TASK_DEFECTS = [
    "none",
    "unrealistic",              # no clinician would send this
    "ambiguous_underspecified",
    "template_generic",         # interchangeable with every other generated task
    "reveals_rubric",           # task text enumerates the checklist
    "too_easy",
    "wrong_use_case",           # does not exercise its stated domain
]

EVOLVE_PER_CASE_SYSTEM = """\
You are the ERROR ANALYST for an automatic curriculum that trains a medical AI for HealthBench \
Professional (domains: care consult, writing & documentation, medical research). You are shown ONE \
training case: the clinician task, the model's FINAL ANSWER, the rubric, which criteria were met, \
and the score.

Your job is to separate TWO different things that look identical in the score:
  (a) the task and rubric were fair and the MODEL genuinely failed -> that is a capability gap the
      curriculum should target more;
  (b) the task or rubric was DEFECTIVE -> that is a generator bug, and training on more of it
      actively harms the model.
Conflating these is how this loop has failed before: a curriculum that gets "harder" by getting
more broken reads exactly like one that gets harder by getting better.

Harness facts -- these are settled, do not re-diagnose them:
- The model reasons in a private channel that is ALREADY REMOVED from the answer you see and is
  NEVER graded. Reasoning is not a failure and rubrics must never target its presence, absence,
  or length.
- If THINK_STATS says think_closed=false the answer was truncated mid-reasoning. That is an
  optimizer/budget problem with its own handling; label failure_mode "non_answer" and every
  defect field "none". Never propose task or rubric changes from such a case.

Output ONLY a JSON object, no markdown:
{
  "failure_mode": one of %(modes)s,
  "gap": "<=15 words naming the SPECIFIC missing capability or fact, e.g. 'eGFR threshold for
          metformin discontinuation' -- never a generic label like 'medical accuracy'",
  "rubric_defect": one of %(rdefects)s,
  "task_defect": one of %(tdefects)s,
  "verdict": "model_failed_fair_task" | "generator_at_fault" | "model_succeeded"
}
Choose the SINGLE most decisive value for each field. "gap" is the field the curriculum is
actually steered by, so make it concrete enough that a task author could write a new task
targeting it."""% {
    "modes": EVOLVE_FAILURE_MODES,
    "rdefects": EVOLVE_RUBRIC_DEFECTS,
    "tdefects": EVOLVE_TASK_DEFECTS,
}

EVOLVE_AGGREGATE_SYSTEM = """\
You are the meta-optimizer for a self-evolving curriculum that trains a medical AI for HealthBench \
Professional (care consult, writing & documentation, medical research — NOT diagnosis). You rewrite \
the appended guidance of two generation prompts:
  (A) the QUERY PROPOSER — proposes the clinician requests that become tasks, and
  (B) the TASK+RUBRIC GENERATOR — writes each clinician task and its grading rubric.

You are given: the reward formula the solver is optimised against; MEASURED statistics of the tasks \
your last guidance actually produced, against the real benchmark's measured statistics; a COUNTED \
error analysis over this step's rollouts; the concrete capability gaps behind those errors; and a \
history of previous guidance versions with the HELD-OUT validation score measured under each.

=== HOW TO READ THE EVIDENCE (in priority order) ===

1. HELD-OUT VALIDATION SCORE is the only outcome that counts. It is measured on the real, untouched \
benchmark, so it cannot be gamed by changing your own tasks. Directions under which it rose are \
working — keep and extend them. Directions under which it fell or stalled are wrong, whatever else \
looked good.

2. TRAIN MEAN SCORE IS NOT AN OBJECTIVE, and raising it is not evidence of anything. It is the \
solver's score on tasks YOU wrote, so you can raise it at will by making tasks easier or rubrics \
looser — and every past failure of this loop did exactly that while held-out performance fell. Read \
it only as a difficulty gauge: roughly 0.4-0.6 means the tasks are pitched about right; near 0.9 \
means they are too easy to teach anything; near 0.1 means they are too hard to give gradient.

3. DISTRIBUTION and DIVERSITY statistics are arithmetic, not opinion. Where a generated statistic \
has drifted from its benchmark reference, correcting it is the highest-value edit you can make: \
training on a distribution the benchmark does not contain is wasted compute no matter how good the \
individual tasks are.

4. The ERROR TAXONOMY tells you what the solver actually lacks. Target the modes with the largest \
counts, but ONLY those attributed to "model_failed_fair_task". Counts under \
"generator_at_fault" are YOUR bugs — fix the generator guidance instead of writing more tasks of \
that kind.

=== WHAT YOUR REWRITE MUST ACHIEVE (all four, simultaneously) ===

(1) DISTRIBUTION ALIGNMENT — generated tasks must look like the benchmark on every measured axis: \
use-case mix, criteria per task, criterion length, point scale, and share of tasks carrying a \
negative criterion. Where the measurements show drift, say so explicitly in the guidance with the \
number to hit.

(2) DIVERSITY — vary specialty, sub-topic, patient context, language/register, artifact type, and \
difficulty. Template collapse is the single most destructive failure this loop has: it is measured \
below as task self-similarity, and if that number is climbing, your guidance is the cause. Diversity \
comes from naming AXES to vary, never from listing specific topics — a topic list becomes the next \
template.

(3) QUALITY — rubrics must be objectively checkable from the answer alone, must test judgment \
BEYOND the task's surface instructions, and must not be satisfiable by restating the prompt.

(4) SOLVER NEED — the tasks must exercise the specific capabilities the error analysis shows the \
solver failing, at the granularity of the "gap" strings, not the coarse category names.

Guidance is appended verbatim into each prompt, so write concrete imperative instructions. Keep what \
the evidence shows is working; replace what it does not. Each block <= 300 words.

=== HARD CONSTRAINTS (these override anything the evidence seems to suggest) ===
- The solver's private reasoning channel is stripped before grading and NEVER graded. Never target \
  chain-of-thought suppression, "no reasoning traces", word caps, or exact header echoes. Those \
  criteria are dead weight that grade as free points.
- NEVER make rubrics easier, "transparent", or 1:1-mapped to the task's stated deliverables. Raise \
  difficulty through the TASK; never through leniency, and never by relaxing grading.
- Rubrics stay 1-5 SHORT single-fact criteria at +5..+10 each (modal +8), with NO required point \
  total. Never push back toward many long conjunctive criteria or a fixed total: only the ratio of \
  achieved to available points is scored, so a fixed total rewards covering ground over being right.
- Negative criteria belong on about a third of tasks, wherever a concrete clinical trap genuinely \
  exists — not on all of them, and not on none.
- Do NOT name or paraphrase specific held-out benchmark items.
- Do not oscillate: if the history shows two alternatives have both been tried, pick a third \
  direction rather than returning to either.

Output ONLY a JSON object:
{"query_proposer_guidance": "<new guidance text>", "task_rubric_generator_guidance": "<new guidance text>", "summary": "<2-3 sentences: what the evidence showed and what you changed because of it>"}"""


def _evolve_endpoint(state: ServerState) -> dict:
    """Endpoint kwargs for the meta-optimizer calls.

    Empty dict (=> the local teacher) unless --evolve_model_name is set. Running
    the analyzer/meta-optimizer on an EXTERNAL model breaks the self-referential
    loop in which the same model writes the tasks, grades them, diagnoses its own
    failures, and rewrites its own curriculum.
    """
    m = getattr(state.args, "evolve_model_name", "") or ""
    if not m:
        return {}
    return {
        "api_base": getattr(state.args, "evolve_api_base", "") or state.args.api_base,
        "api_key": getattr(state.args, "evolve_api_key", "") or state.args.api_key,
        "model_name": m,
        "provider_override": getattr(state.args, "evolve_provider", "") or "",
    }


# ======================================================================
# RETRIEVAL-REWARD EVOLUTION.
#
# The coverage judge defines what "good retrieval" MEANS, so it is the thing to
# evolve when the goal is teaching the policy to retrieve better. Evolving it
# needs an objective, and "make coverage higher" is not one -- the judge would
# simply become more generous, the bonus would go to every rollout equally, and
# the group-relative fold would cancel it to nothing.
#
# The objective used here is DISCRIMINATION: coverage is a useful reward exactly
# to the extent that, among rollouts of the same task, the one whose retrieval
# scored higher also ANSWERED better. That is measurable in code (pairwise
# concordance against the answer score), it is not controllable by the judge
# (the judge does not grade answers), and it is zero-sum-proof: a judge that
# says 1.0 for everything scores 0.5 concordance, i.e. no information.
#
# The second measured property is SPREAD. A judge whose verdicts are constant
# within a group produces deltas of exactly zero under the group-relative fold,
# so the query tokens get no gradient at all -- the reward is off while every
# log says it is on.
# ======================================================================

COVERAGE_EVOLVE_SYSTEM = """\
You tune the COVERAGE JUDGE that provides the retrieval-quality reward in a medical RL run.

Setup you are tuning inside of. A policy answers a clinician task and may issue up to two
retrieval calls against a medical knowledge base first. Its reward has two separately-graded
parts: an ANSWER score from a rubric grader, and a RETRIEVAL score -- your judge -- which asks
whether the passages the policy retrieved actually SUPPLY what the task's rubric criteria
reward, independently of whether the answer then used them. The retrieval score is folded in
GROUP-RELATIVE: each rollout's coverage is centred on the mean coverage of the other rollouts
of the SAME task that also searched. Two consequences follow, and they drive everything:

  1. Only DIFFERENCES WITHIN A GROUP matter. Raising every verdict raises the baseline
     identically and changes no reward at all. Making the judge more generous, or more
     harsh, accomplishes exactly nothing on its own.
  2. If your verdicts are identical across a group, every delta is zero and the query tokens
     receive NO gradient. A judge that cannot separate a good query plan from a bad one on
     the same task is not a strict judge; it is an absent one.

So your objective is DISCRIMINATION, and it is measured for you: among rollout pairs on the
same task where one answered better than the other, how often did your judge also score its
retrieval higher? 0.50 means your verdicts carry no information about retrieval quality.
Above 0.50 means the reward is pointing the policy somewhere real.

You are given: the current judge prompt, its measured discrimination and verdict spread, and
CASES chosen because they are informative -- pairs where the answer scores clearly differed,
plus rollouts where your verdict and the answer disagreed most. Diagnose why the judge failed
to separate them, then rewrite it.

What actually tends to be wrong, in rough order:
- The judge scores TOPICALITY rather than SUPPLY: passages about the right disease all look
  alike, so a precise query and a vague one on the same topic get the same verdict.
- It ignores SPECIFICITY: the criterion rewards a number, threshold, dose, or named
  guideline, and a passage that discusses the concept without stating the value is counted
  as supplying it. Distinguishing these is usually the single biggest win, because it is
  exactly where a better query plan differs from a worse one.
- It is inconsistent on criteria retrieval cannot help with (pure behaviour, tone, asking a
  follow-up), adding noise that swamps the real differences.
- It rewards passage COUNT or length instead of whether the specific fact is present.

HARD CONSTRAINTS -- the reward breaks, silently, if you violate any of these:
- The template MUST keep the fields {task}, {passages} and {criteria}, spelled exactly, each
  appearing at least once. They are substituted by str.format.
- The template MUST keep the output contract: one object per criterion, each carrying "idx"
  (the criterion's integer index) and "supplied" (a boolean), returned under {"results": [...]}.
  The parser reads nothing else, and a prompt that changes this scores every rollout zero.
- Use no other single braces anywhere; write literal braces as {{ }}.
- Keep it a per-criterion factual-supply judgement. Do not turn it into an answer grader, a
  query-plan critic, or a relevance ranker.
- Output must stay compact JSON: this runs on every searching rollout of every step.

Output ONLY a JSON object:
{"system": "<new system prompt>", "template": "<new template>", "summary": "<2-3 sentences: what the evidence showed and what you changed>"}"""


def _coverage_evidence(cases: list[dict]) -> tuple[str, dict]:
    """Measure how well the coverage judge DISCRIMINATES, and render the evidence.

    `cases` are searching rollouts carrying (uid, coverage, answer_score, queries).
    Concordance is computed over within-group pairs whose answer scores differ by a
    margin, so ties -- which carry no information about ordering -- cannot inflate
    it. This is the number the meta-optimizer is steered by, and it is deliberately
    one the judge cannot move by being kinder or stricter.
    """
    ok = [c for c in cases if c.get("judged")]
    out: dict = {"n_cases": len(cases), "n_judged": len(ok)}
    if not ok:
        return "(no judged retrieval cases this step)", out

    covs = [float(c["coverage"]) for c in ok]
    n = len(covs)
    mean = sum(covs) / n
    var = sum((c - mean) ** 2 for c in covs) / n
    out.update({
        "coverage_mean": round(mean, 3),
        "coverage_sd": round(var ** 0.5, 3),
        "frac_zero": round(sum(1 for c in covs if c <= 1e-9) / n, 3),
        "frac_one": round(sum(1 for c in covs if c >= 1 - 1e-9) / n, 3),
    })

    groups: dict = defaultdict(list)
    for c in ok:
        groups[c.get("uid", "?")].append(c)

    MARGIN = 0.05          # below this the two answers are not meaningfully different
    conc = disc = tie = 0
    within_group_spread = []
    informative: list = []
    for g in groups.values():
        if len(g) >= 2:
            gc = [float(x["coverage"]) for x in g]
            within_group_spread.append(max(gc) - min(gc))
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                da = float(a["answer_score"]) - float(b["answer_score"])
                if abs(da) < MARGIN:
                    continue
                dc = float(a["coverage"]) - float(b["coverage"])
                if abs(dc) < 1e-9:
                    tie += 1
                elif (da > 0) == (dc > 0):
                    conc += 1
                else:
                    disc += 1
                # Keep the pairs where the judge most clearly contradicted the
                # answer: those are what a rewrite has to explain.
                if abs(da) >= 0.15 and (abs(dc) < 1e-9 or (da > 0) != (dc > 0)):
                    better, worse = (a, b) if da > 0 else (b, a)
                    informative.append({
                        "better_answer": round(float(better["answer_score"]), 3),
                        "better_cov": round(float(better["coverage"]), 3),
                        "better_queries": better.get("queries") or [],
                        "worse_answer": round(float(worse["answer_score"]), 3),
                        "worse_cov": round(float(worse["coverage"]), 3),
                        "worse_queries": worse.get("queries") or [],
                        "task": (better.get("task") or "")[:400],
                        "criteria": (better.get("criteria") or [])[:5],
                    })
    ranked = conc + disc
    concordance = (conc / ranked) if ranked else None
    out["concordance"] = round(concordance, 3) if concordance is not None else -1.0
    out["n_ranked_pairs"] = float(ranked)
    out["ties"] = float(tie)
    spread = (sum(within_group_spread) / len(within_group_spread)) if within_group_spread else 0.0
    out["within_group_spread"] = round(spread, 3)

    verdict = (
        "NO INFORMATION: the judge's ordering is no better than chance"
        if concordance is None or abs(concordance - 0.5) < 0.05 else
        ("INVERTED: the judge systematically prefers the retrieval of the WORSE answer, "
         "which actively trains the policy toward bad query plans"
         if concordance < 0.45 else
         "informative: the judge's ordering tracks answer quality")
    )
    lines = [
        f"Judged retrieval rollouts this step: {len(ok)} of {len(cases)}.",
        f"  DISCRIMINATION (the objective): concordance {out['concordance']} over "
        f"{ranked} within-task pairs whose answer scores differed by >= {MARGIN}. "
        f"0.50 = no information. --> {verdict}",
        f"  Ties (equal coverage on a pair that differed in answer quality): {tie}"
        + ("   <-- each of these is a pair the judge could not separate at all" if tie else ""),
        f"  SPREAD: mean within-group coverage range {out['within_group_spread']}"
        + ("   <-- NEAR ZERO: the group-relative fold turns this into no gradient at all "
           "on the query tokens" if spread < 0.05 else ""),
        f"  Verdict distribution: mean {out['coverage_mean']}, sd {out['coverage_sd']}, "
        f"all-false {out['frac_zero']}, all-true {out['frac_one']}"
        + ("   <-- SATURATED" if out["frac_one"] > 0.5 or out["frac_zero"] > 0.5 else ""),
    ]
    if informative:
        lines.append("\n  CASES WHERE THE JUDGE DISAGREED WITH THE ANSWER (what a rewrite must fix):")
        for k, c in enumerate(informative[:6], 1):
            lines.append(
                f"    [{k}] task: {c['task'][:220]}\n"
                f"        criteria: {c['criteria']}\n"
                f"        BETTER answer {c['better_answer']} but judge gave its retrieval "
                f"{c['better_cov']}  queries={c['better_queries']}\n"
                f"        WORSE  answer {c['worse_answer']} and judge gave its retrieval "
                f"{c['worse_cov']}  queries={c['worse_queries']}"
            )
    return "\n".join(lines), out


def _task_shingles(text: str, n: int = 5) -> set:
    """Word 5-grams of a task, for the self-similarity (template-collapse) metric."""
    w = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def _corpus_stats(state: "ServerState", sample: int = 150) -> dict:
    """Measure the curriculum the generator is CURRENTLY producing.

    Reads the ring of entries actually pushed into the pool, so it reflects what
    the solver is being trained on right now, including whatever the last
    guidance rewrite changed. Every number here is arithmetic over real generated
    tasks -- none of it is a model's opinion of its own output, which is the
    whole point: the two failures this loop has had (rubric-shape drift and
    template collapse) were both invisible precisely because nothing counted.
    """
    ents = list(state.history)[-sample:]
    out: dict = {"n": len(ents)}
    if not ents:
        return out

    uc: dict = defaultdict(int)
    spec: dict = defaultdict(int)
    diff: dict = defaultdict(int)
    ncrit: list = []
    clen: list = []
    pos_pts: list = []
    n_with_neg = 0
    texts: list = []
    for e in ents:
        ei = e.get("extra_info", {}) or {}
        uc[str(ei.get("use_case", "?"))] += 1
        spec[str(ei.get("specialty", "?"))] += 1
        diff[str(ei.get("difficulty", "?"))] += 1
        items = list(ei.get("rubric_items") or [])
        ncrit.append(len(items))
        has_neg = False
        for it in items:
            t = str((it or {}).get("criterion_text") or (it or {}).get("criterion") or "")
            clen.append(len(t))
            try:
                p = float((it or {}).get("points", 0))
            except (TypeError, ValueError):
                continue
            if p > 0:
                pos_pts.append(p)
            elif p < 0:
                has_neg = True
        n_with_neg += bool(has_neg)
        texts.append(str(ei.get("question", "") or ""))

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    def _median(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        m = len(s) // 2
        return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2.0

    n = len(ents)
    out.update({
        "use_case_mix": {k: round(v / n, 3) for k, v in sorted(uc.items(), key=lambda kv: -kv[1])},
        "criteria_per_task_mean": round(_mean(ncrit), 2),
        "criteria_per_task_median": _median(ncrit),
        "criteria_per_task_max": max(ncrit) if ncrit else 0,
        "criterion_chars_mean": round(_mean(clen)),
        "criterion_chars_median": round(_median(clen)),
        "positive_points_mean": round(_mean(pos_pts), 1),
        "frac_tasks_with_negative": round(n_with_neg / n, 3),
        "difficulty_mix": {k: round(v / n, 3) for k, v in sorted(diff.items(), key=lambda kv: -kv[1])},
        "distinct_specialties": len(spec),
        "task_chars_median": round(_median([len(t) for t in texts])),
    })

    # Template collapse: mean pairwise Jaccard of task 5-grams over a bounded
    # random sample of pairs. Two independently written clinician requests share
    # almost no 5-grams (~0.01); one filled-in template drives this toward 0.3+.
    # Gate on WORD count, not characters: a 5-gram needs 5 words, and gating on
    # characters silently dropped the whole metric for terse curricula -- i.e. it
    # went blind exactly when tasks had collapsed into short stereotyped stubs,
    # which is the case it exists to catch. Unmeasurable is reported as such
    # (None) rather than omitted.
    shs = [s for s in (_task_shingles(t) for t in texts) if len(s) >= 8]
    out["task_self_similarity"] = None
    if len(shs) >= 4:
        rnd = random.Random(0)          # deterministic so the number is comparable across steps
        pairs = min(200, len(shs) * (len(shs) - 1) // 2)
        sims = []
        for _ in range(pairs):
            a, b = rnd.sample(range(len(shs)), 2)
            u = len(shs[a] | shs[b])
            sims.append((len(shs[a] & shs[b]) / u) if u else 0.0)
        out["task_self_similarity"] = round(_mean(sims), 3)
    return out


def _format_corpus_block(cur: dict) -> str:
    """Render generated-vs-benchmark statistics side by side, flagging drift.

    Presenting the reference next to the measurement (rather than asking the
    meta-optimizer to remember the target) is what makes objectives 1-2
    actionable: it can only correct drift it can see.
    """
    if not cur.get("n"):
        return "(no generated tasks measured yet)"
    ref = HB_REF_STATS
    rows = []

    def _row(label, got, want, bad=None):
        flag = "   <-- DRIFT" if (bad is not None and bad) else ""
        rows.append(f"  {label:<34} generated={got!s:<22} benchmark={want!s}{flag}")

    gm = cur.get("use_case_mix", {})
    if gm and not (set(gm) & set(ref["use_case_mix"])):
        # Zero overlap means the labels changed, not the distribution. Saying so is
        # the only honest output; reporting maximal drift would send the
        # meta-optimizer chasing a difference that does not exist.
        rows.append(f"  {'use-case mix':<34} generated={gm} benchmark={ref['use_case_mix']}"
                    f"   <-- LABEL MISMATCH, not drift: no shared keys, comparison is "
                    f"meaningless. Ignore this row and report it as a bug.")
    else:
        _row("use-case mix", gm, ref["use_case_mix"],
             any(abs(gm.get(k, 0.0) - v) > 0.10 for k, v in ref["use_case_mix"].items()))
    _row("criteria per task (mean)", cur.get("criteria_per_task_mean"), ref["criteria_per_task_mean"],
         abs((cur.get("criteria_per_task_mean") or 0) - ref["criteria_per_task_mean"]) > 0.8)
    _row("criteria per task (median/max)",
         f"{cur.get('criteria_per_task_median')}/{cur.get('criteria_per_task_max')}",
         f"{ref['criteria_per_task_median']}/{ref['criteria_per_task_max']}")
    # RELATIVE threshold. The absolute >60 this used to be never fired on the
    # shortfall that actually occurs: generated criteria run ~80 chars against the
    # benchmark's 135, a 40% miss that sat 5 chars under the bar — silent on
    # precisely the short-vs-conjunctive axis this row exists to police.
    _row("criterion chars (mean)", cur.get("criterion_chars_mean"), ref["criterion_chars_mean"],
         abs((cur.get("criterion_chars_mean") or 0) - ref["criterion_chars_mean"])
         > 0.25 * ref["criterion_chars_mean"])
    _row("criterion chars (median)", cur.get("criterion_chars_median"), ref["criterion_chars_median"])
    _row("positive points (mean)", cur.get("positive_points_mean"), ref["modal_positive_points"])
    _row("tasks with a negative criterion", cur.get("frac_tasks_with_negative"),
         ref["frac_tasks_with_negative"],
         abs((cur.get("frac_tasks_with_negative") or 0) - ref["frac_tasks_with_negative"]) > 0.20)
    rows.append(f"  {'difficulty mix':<34} generated={cur.get('difficulty_mix')}")
    rows.append(f"  {'distinct specialties':<34} generated={cur.get('distinct_specialties')} "
                f"(of {len(HB_SPECIALTIES)} available)")
    sim = cur.get("task_self_similarity")
    if sim is None:
        rows.append(f"  {'task self-similarity (5-gram)':<34} generated=NOT MEASURABLE "
                    f"(tasks too short to form 5-grams -- itself a sign the tasks have "
                    f"become terse stubs)")
    else:
        note = ("   <-- TEMPLATE COLLAPSE: tasks are near-duplicates of each other"
                if sim > 0.15 else ("   (mild templating)" if sim > 0.07 else "   (healthy)"))
        rows.append(f"  {'task self-similarity (5-gram)':<34} generated={sim}{note}")
    return (f"Measured over the last {cur['n']} generated tasks, against the real benchmark "
            f"(525 tasks / 1128 criteria):\n" + "\n".join(rows))


def _parse_diagnosis(raw: str) -> dict | None:
    """Parse one per-case error-analysis JSON, keeping only in-vocabulary values.

    Out-of-vocabulary labels are dropped rather than counted: a taxonomy the
    analyst can silently extend is a taxonomy that cannot be aggregated.
    """
    # _parse_json RAISES on unparseable input, and the per-case analyzer's own
    # failure path returns the plain string "(analysis failed: ...)". Letting that
    # propagate would abort the entire evolution round because one of 24 analyses
    # timed out, so a dropped case must stay a dropped case.
    try:
        obj = _parse_json(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    def _pick(key, vocab):
        v = str(obj.get(key, "") or "").strip().lower()
        return v if v in vocab else None

    return {
        "failure_mode": _pick("failure_mode", EVOLVE_FAILURE_MODES),
        "rubric_defect": _pick("rubric_defect", EVOLVE_RUBRIC_DEFECTS),
        "task_defect": _pick("task_defect", EVOLVE_TASK_DEFECTS),
        "verdict": _pick("verdict", ["model_failed_fair_task", "generator_at_fault", "model_succeeded"]),
        "gap": str(obj.get("gap", "") or "").strip()[:160],
    }


def _aggregate_diagnoses(diags: list[dict]) -> tuple[str, dict]:
    """Count the taxonomy in CODE and render it for the meta-optimizer.

    Returns (rendered_block, metrics). Counting here rather than asking the
    aggregate model to tally 20 prose blobs is the difference between "the model
    sometimes struggles with dosing" and "knowledge_factual 9/20" -- only the
    second can be compared against the next step's number.
    """
    ok = [d for d in diags if d]
    if not ok:
        return "(no parseable diagnoses this step)", {}
    n = len(ok)

    def _count(key):
        c: dict = defaultdict(int)
        for d in ok:
            if d.get(key):
                c[d[key]] += 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    verdicts = _count("verdict")
    fair = [d for d in ok if d.get("verdict") == "model_failed_fair_task"]
    modes_fair: dict = defaultdict(int)
    for d in fair:
        if d.get("failure_mode") and d["failure_mode"] != "none":
            modes_fair[d["failure_mode"]] += 1
    modes_fair = dict(sorted(modes_fair.items(), key=lambda kv: -kv[1]))
    rdef = {k: v for k, v in _count("rubric_defect").items() if k != "none"}
    tdef = {k: v for k, v in _count("task_defect").items() if k != "none"}
    gaps = [d["gap"] for d in fair if d.get("gap")][:15]

    n_fault = verdicts.get("generator_at_fault", 0)
    block = (
        f"Structured error analysis over {n} sampled rollouts (sampling is FAILURE-HEAVY by "
        f"design, so these proportions are worse than the batch as a whole):\n"
        f"  verdict split: {verdicts}\n"
        f"  FAILURE MODES on fair tasks ({len(fair)} cases -- target these): {modes_fair or '(none)'}\n"
        f"  RUBRIC defects ({n_fault} cases blamed the generator -- fix these in guidance (B)): "
        f"{rdef or '(none)'}\n"
        f"  TASK defects: {tdef or '(none)'}\n"
        f"  Specific capability gaps behind the fair-task failures:\n"
        + ("\n".join(f"    - {g}" for g in gaps) if gaps else "    (none reported)")
    )
    metrics = {
        "diag/n_parsed": float(n),
        "diag/frac_generator_at_fault": round(n_fault / n, 3),
        "diag/frac_model_failed_fair": round(len(fair) / n, 3),
        "diag/n_rubric_defects": float(sum(rdef.values())),
        "diag/n_task_defects": float(sum(tdef.values())),
    }
    return block, metrics


def _format_evolve_case(c: dict) -> str:
    """Render one case for the per-case analysis prompt.

    ``response`` arrives from the trainer already think-STRIPPED (the final
    answer, exactly what the grader saw) and is rendered IN FULL — truncating it
    used to show the analyzer only reasoning fragments and derailed evolution.
    """
    conv = c.get("conversation") or []
    if isinstance(conv, list) and conv:
        task = conv[-1].get("content", "") if isinstance(conv[-1], dict) else str(conv)
    else:
        task = c.get("question", "")
    items = c.get("item_results") or c.get("rubric_items") or []
    lines = []
    for it in items:
        crit = it.get("criterion_text") or it.get("criterion") or ""
        pts = it.get("points")
        met = it.get("met")
        met_s = "" if met is None else (" MET" if met else " not-met")
        lines.append(f"  [{pts}]{met_s} {crit}")
    rubric = "\n".join(lines)
    closed = c.get("think_closed")
    think_stats = ""
    if closed is not None:
        think_stats = (f"THINK_STATS: think_closed={str(bool(closed)).lower()} "
                       f"think_chars={int(c.get('think_chars', 0))}\n")
    return (
        f"USE_CASE: {c.get('use_case', '?')}\n"
        f"TASK: {task[:4000]}\n"
        f"{think_stats}"
        f"MODEL FINAL ANSWER (reasoning channel already removed):\n"
        f"{(c.get('response') or '(empty)')[:30000]}\n"
        f"RUBRIC (points / met):\n{rubric}\n"
        f"SCORE: {c.get('total_score', 0.0):.3f}"
    )


def _history_block(history: list[dict], max_entries: int = 8) -> str:
    """Render past guidance versions and their measured outcomes, oldest first.

    HELD-OUT VAL leads and TRAIN score is explicitly labelled as non-objective.
    That ordering is the point: the earlier version of this block reported only
    the train mean and told the meta-optimizer to prefer directions that raised
    it, which rewards trivialising the curriculum -- the exact failure seen in
    v2/v3, where the evolver responded to a hard step by writing easier tasks and
    the train score duly improved while held-out performance did not.
    """
    ent = history[-max_entries:]
    if not ent:
        return "(no previous guidance versions)"
    parts = []
    prev_val = None
    prev_train = None
    for e in ent:
        outs = e.get("outcomes") or []
        # Two record shapes share this list: train telemetry (mean_score, stamped
        # `step`) and held-out val (val_score, stamped `val_step`), which are
        # attached at different times to different versions. Average each over its
        # OWN records — dividing the train sum by len(outs) would dilute it by the
        # val entries and quietly understate every version's difficulty gauge.
        tr_recs = [o for o in outs if o.get("mean_score") is not None]
        if outs:
            tr = (sum(o["mean_score"] for o in tr_recs) / len(tr_recs)) if tr_recs else 0.0
            unc = (sum(o.get("unclosed_rate", 0.0) for o in tr_recs) / len(tr_recs)) if tr_recs else 0.0
            vals = [o["val_score"] for o in outs if o.get("val_score") is not None]
            if vals:
                v = sum(vals) / len(vals)
                vd = "" if prev_val is None else f" (delta {v - prev_val:+.3f})"
                val_s = f"HELD-OUT VAL {v:.3f}{vd}"
                prev_val = v
            else:
                val_s = "HELD-OUT VAL not yet measured under this version"
            td = "" if prev_train is None else f" (delta {tr - prev_train:+.3f})"
            prev_train = tr
            outcome = (f"{val_s}; train mean {tr:.3f}{td} [difficulty gauge only, not an "
                       f"objective]; unclosed-think {unc:.2f}")
        else:
            outcome = "no outcome measured yet"
        parts.append(
            f"--- version committed at step {e.get('step')}: {outcome}\n"
            f"    rationale then: {(e.get('summary') or '')[:300]}\n"
            f"    proposer guidance: {(e.get('query_guidance') or '(none)')[:800]}\n"
            f"    generator guidance: {(e.get('generator_guidance') or '(none)')[:800]}"
        )
    return "\n".join(parts)


def _save_evolve_history(state: ServerState) -> None:
    try:
        _atomic_write(state.evolve_history_path, json.dumps(state.evolve_history, indent=1))
    except Exception as e:
        logger.warning(f"could not persist evolve history: {e}")


def _attach_val_outcome(hist: list[dict], val_score: float | None,
                        val_step: int | None) -> str:
    """Credit a held-out val measurement to the guidance version that EARNED it.

    Off-by-one-interval trap: evolution runs just BEFORE validation within a step,
    so the version committed at step S has influenced nothing by the time val runs
    at step S. val@S measures a model trained entirely under the version committed
    at the PREVIOUS evolve step. Attaching it to hist[-1] -- the version just
    committed -- shifts the whole delta chain by one interval and, since the
    meta-prompt makes held-out val the sole criterion, systematically credits every
    rewrite with its predecessor's result.

    So: attach to the last entry committed STRICTLY BEFORE val_step. A val that
    predates every entry belongs to the seed guidance, which has no entry; report
    that rather than misfiling it.
    """
    if val_score is None or not hist:
        return "no val to attach"
    if val_step is None:
        return "val_step unknown; not attached (cannot verify which version earned it)"
    owner = None
    for e in hist:
        if int(e.get("step", -1)) < int(val_step):
            owner = e
    if owner is None:
        return f"val@{val_step} predates every guidance version (belongs to the seed); not attached"
    owner.setdefault("outcomes", []).append({
        "val_step": int(val_step), "val_score": float(val_score)})
    return f"val@{val_step}={val_score:.4f} -> version committed at step {owner.get('step')}"


async def _evolve_prompts(state: ServerState, step: int, cases: list[dict],
                          val_score: float | None = None,
                          val_step: int | None = None) -> dict:
    """Run one generation-prompt evolution round from ~20 sampled rollouts.

    Three stages: structured per-case ERROR ANALYSIS -> code-side aggregation of
    the taxonomy and of the generated corpus's distribution -> one meta-optimizer
    call that rewrites both guidance blocks against measured evidence.
    """
    state.gen_step = max(state.gen_step, int(step))
    n = len(cases)
    mean_score = (sum(float(c.get("total_score", 0.0)) for c in cases) / n) if n else 0.0
    unclosed_rate = (sum(1 for c in cases if c.get("think_closed") is False) / n) if n else 0.0
    mean_answer_chars = (sum(len(c.get("response") or "") for c in cases) / n) if n else 0.0

    # Train-side telemetry belongs to the most recent version (approximate — the
    # sample pool lags a rewrite by a step or two). The HELD-OUT val score does
    # NOT: it is credited to whichever version actually earned it, which is an
    # interval earlier. See _attach_val_outcome.
    if state.evolve_history:
        state.evolve_history[-1].setdefault("outcomes", []).append({
            "step": int(step), "mean_score": mean_score,
            "unclosed_rate": unclosed_rate, "mean_answer_chars": mean_answer_chars,
        })
    _val_note = _attach_val_outcome(state.evolve_history, val_score, val_step)
    logger.info(f"/evolve val attribution: {_val_note}")

    # 1) Per-case structured error analysis (concurrent, bounded).
    async def analyze(c: dict) -> str:
        try:
            return await _api_call(state, EVOLVE_PER_CASE_SYSTEM, _format_evolve_case(c),
                                   **_evolve_endpoint(state),
                                   max_tokens=512, temperature=0.3, label="evolve_per_case",
                                   want_json=True)
        except Exception as e:
            return f"(analysis failed: {type(e).__name__})"
    summaries = await asyncio.gather(*[analyze(c) for c in cases]) if cases else []
    diags = [_parse_diagnosis(s) for s in summaries]
    diag_block, diag_metrics = _aggregate_diagnoses(diags)

    # 2) Measure what the generator is currently producing, then aggregate.
    corpus = _corpus_stats(state)
    cur_q = state.prompt_store.get("query_proposer_guidance")
    cur_g = state.prompt_store.get("task_rubric_generator_guidance")
    val_s = (f"{val_score:.3f}" if val_score is not None else "not yet measured")
    agg_user = (
        f"=== REWARD FORMULA THE SOLVER IS OPTIMISED AGAINST ===\n{HB_SCORE_FORMULA}\n\n"
        f"=== OUTCOME THIS STEP ===\n"
        f"HELD-OUT VALIDATION SCORE (the objective): {val_s}\n"
        f"Train mean over {n} failure-heavy sampled rollouts: {mean_score:.3f} "
        f"(difficulty gauge only — NOT an objective; reads low vs the batch mean by design). "
        f"Unclosed-think rate {unclosed_rate:.2f}; mean answer {mean_answer_chars:.0f} chars "
        f"(length adjustment is neutral at 2000).\n\n"
        f"=== OBJECTIVES 1-2: DISTRIBUTION & DIVERSITY OF WHAT YOU ARE GENERATING ===\n"
        f"{_format_corpus_block(corpus)}\n\n"
        f"=== OBJECTIVES 3-4: QUALITY & SOLVER NEED ===\n{diag_block}\n\n"
        f"=== GUIDANCE HISTORY & MEASURED OUTCOMES (oldest first) ===\n"
        f"{_history_block(state.evolve_history)}\n\n"
        f"=== CURRENT query-proposer guidance ===\n{cur_q or '(none)'}\n\n"
        f"=== CURRENT task+rubric-generator guidance ===\n{cur_g or '(none)'}"
    )
    new_q, new_g, summary = cur_q, cur_g, ""
    changed_q = changed_g = False
    # Loud-failure plumbing: a dead evolver must never masquerade as a running
    # experiment (v15 trained 95 steps on frozen guidance because a TRAPI 404
    # was logged once per step at WARNING and swallowed). Failures now log at
    # ERROR with a consecutive count and drop an EVOLVE_FAILING marker file in
    # log_dir for the job monitor; success removes it.
    _fail_marker = os.path.join(state.args.log_dir, "EVOLVE_FAILING")
    n_case_fail = sum(1 for s in summaries if s.startswith("(analysis failed"))
    if summaries and n_case_fail == len(summaries):
        logger.error(f"/evolve: ALL {n_case_fail} per-case analyses failed — evolve model unreachable?")
    try:
        raw = await _api_call(state, EVOLVE_AGGREGATE_SYSTEM, agg_user,
                              **_evolve_endpoint(state),
                              max_tokens=4096, temperature=0.5, label="evolve_aggregate",
                              want_json=True)
        obj = _parse_json(raw)
        if isinstance(obj, dict):
            cand_q = (obj.get("query_proposer_guidance") or "").strip()
            cand_g = (obj.get("task_rubric_generator_guidance") or "").strip()
            summary = (obj.get("summary") or "").strip()
            if cand_q:
                state.prompt_store.set("query_proposer_guidance", cand_q)
                changed_q = cand_q != cur_q
                new_q = cand_q
            if cand_g:
                state.prompt_store.set("task_rubric_generator_guidance", cand_g)
                changed_g = cand_g != cur_g
                new_g = cand_g
        state.evolve_consec_failures = 0
        try:
            os.remove(_fail_marker)
        except FileNotFoundError:
            pass
    except Exception as e:
        state.evolve_consec_failures = getattr(state, "evolve_consec_failures", 0) + 1
        logger.error(
            f"/evolve aggregate FAILED ({state.evolve_consec_failures} consecutive), keeping guidance: "
            f"{type(e).__name__}: {e}")
        try:
            with open(_fail_marker, "a") as f:
                f.write(f"{datetime.now().isoformat()} step={step} "
                        f"consec={state.evolve_consec_failures} {type(e).__name__}: {e}\n")
        except OSError:
            pass

    # New history entry for this guidance version; its outcomes fill in on later
    # /evolve calls. Record even unchanged guidance so score attribution stays
    # continuous.
    state.evolve_history.append({
        "step": int(step), "query_guidance": new_q, "generator_guidance": new_g,
        "summary": summary, "changed": bool(changed_q or changed_g), "outcomes": [],
    })
    _save_evolve_history(state)

    # 3) Snapshot + log. The full meta-prompt is snapshotted too: when a run goes
    # wrong the first question is always "what did the evolver actually see?", and
    # reconstructing it from the pieces after the fact is guesswork.
    aux = {
        "error_summary.txt": summary,
        "case_summaries.txt": "\n\n".join(summaries),
        "error_analysis.txt": diag_block,
        "corpus_stats.json": json.dumps(corpus, indent=1),
        "aggregate_prompt.txt": agg_user,
        "query_proposer_guidance.txt": new_q,
        "task_rubric_generator_guidance.txt": new_g,
    }
    sdir = state.prompt_store.commit(int(step), aux)
    metrics = {
        "step": int(step), "n_cases": n, "mean_score": mean_score,
        "changed_query_guidance": changed_q, "changed_generator_guidance": changed_g,
        "snapshot": sdir,
        **diag_metrics,
        "corpus/criteria_per_task": corpus.get("criteria_per_task_mean", 0.0),
        "corpus/criterion_chars": corpus.get("criterion_chars_mean", 0.0),
        "corpus/frac_with_negative": corpus.get("frac_tasks_with_negative", 0.0),
        "corpus/distinct_specialties": float(corpus.get("distinct_specialties", 0)),
        # -1 means "not measurable", which must stay distinguishable from 0.0
        # ("measured, and perfectly diverse") in the logged series.
        "corpus/task_self_similarity": (
            corpus["task_self_similarity"] if corpus.get("task_self_similarity") is not None
            else -1.0),
    }
    # Caller (/evolve endpoint) already holds evolve_lock, so write directly.
    with open(state.evolve_log, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), **metrics,
                            "summary": summary}) + "\n")
    logger.info(
        f"~ evolve step={step} cases={n} mean_score={mean_score:.3f} "
        f"changed_q={changed_q} changed_g={changed_g} snapshot={sdir}"
    )
    if summary:
        logger.info(f"  evolve summary: {summary[:400]}")
    return metrics


_COV_REQUIRED_FIELDS = ("{task}", "{passages}", "{criteria}")
_COV_CONTRACT_TOKENS = ("results", "idx", "supplied")


def validate_coverage_prompt(system: str, template: str) -> str:
    """Return "" if usable, else the reason. Mirrors the identical check in
    verl/utils/reward_score/retrieval_coverage.py ON PURPOSE: this one refuses to
    COMMIT a broken rewrite, that one refuses to LOAD a broken file. They guard
    different events, and a coverage prompt that fails silently scores every
    rollout zero, which is indistinguishable from a policy that cannot retrieve."""
    if not (system or "").strip():
        return "empty system prompt"
    if not (template or "").strip():
        return "empty template"
    missing = [f for f in _COV_REQUIRED_FIELDS if f not in template]
    if missing:
        return f"template is missing required field(s): {', '.join(missing)}"
    absent = [t for t in _COV_CONTRACT_TOKENS if t not in template]
    if absent:
        return f"template dropped the JSON output contract (missing {', '.join(absent)})"
    try:
        template.format(task="x", passages="y", criteria="z")
    except Exception as e:  # noqa: BLE001
        return f"template does not format cleanly: {type(e).__name__}: {e}"
    return ""


async def _evolve_retrieval_reward(state: ServerState, step: int, cases: list[dict]) -> dict:
    """Rewrite the coverage judge from measured discrimination evidence.

    Never raises into the trainer: on any failure the current prompt stands, the
    failure is logged at ERROR with a consecutive count, and a marker file is
    dropped so a monitor can see that the experiment stopped running.
    """
    state.gen_step = max(state.gen_step, int(step))
    evidence, stats = _coverage_evidence(cases)

    path = getattr(state.args, "coverage_prompt_file", "") or os.path.join(
        state.args.prompt_dir, "coverage_prompt.json")
    try:
        with open(path) as f:
            cur = json.load(f)
        cur_sys, cur_tpl = str(cur.get("system") or ""), str(cur.get("template") or "")
        cur_ver = int(cur.get("version") or 0)
    except Exception:
        # No file yet: the reward is running on its built-in default, which the
        # trainer sends us so the first rewrite edits the prompt actually in use
        # rather than an empty string.
        cur_sys = str((cases[0] if cases else {}).get("current_system") or "")
        cur_tpl = str((cases[0] if cases else {}).get("current_template") or "")
        cur_ver = 0
    if not cur_tpl:
        logger.error("/evolve_retrieval: no current coverage template available; skipping")
        return {"skipped": "no_current_template", **stats}

    # Attribute the measured outcome to the version that produced it, before any
    # rewrite, so the history reads as version -> what it achieved.
    hist = getattr(state, "coverage_history", None)
    if hist is None:
        hist = state.coverage_history = []
    if hist:
        # -1.0 is the "not measurable" sentinel (no within-group pair had a large
        # enough answer-score gap to rank). Record it as None so the history's
        # averages cannot be dragged below zero by a step that measured nothing.
        _cc = stats.get("concordance")
        hist[-1].setdefault("outcomes", []).append({
            "step": int(step),
            "concordance": (None if _cc is None or _cc < 0 else _cc),
            "within_group_spread": stats.get("within_group_spread"),
            "coverage_mean": stats.get("coverage_mean"),
        })


    hist_lines = []
    for e in hist[-6:]:
        outs = e.get("outcomes") or []
        if outs:
            cc = [o["concordance"] for o in outs if o.get("concordance") is not None]
            sp = [o["within_group_spread"] for o in outs if o.get("within_group_spread") is not None]
            res = (f"concordance {sum(cc)/len(cc):.3f}" if cc else "concordance not measured")
            res += (f", spread {sum(sp)/len(sp):.3f}" if sp else "")
        else:
            res = "no outcome measured yet"
        hist_lines.append(f"--- v{e.get('version')} committed at step {e.get('step')}: {res}\n"
                          f"    rationale: {(e.get('summary') or '')[:240]}")
    hist_block = "\n".join(hist_lines) or "(no previous versions)"

    user = (
        f"=== MEASURED PERFORMANCE OF THE CURRENT JUDGE ===\n{evidence}\n\n"
        f"=== VERSION HISTORY (oldest first) ===\n{hist_block}\n\n"
        f"=== CURRENT system prompt ===\n{cur_sys}\n\n"
        f"=== CURRENT template ===\n{cur_tpl}"
    )

    _marker = os.path.join(state.args.log_dir, "COVERAGE_EVOLVE_FAILING")
    changed, summary, new_ver = False, "", cur_ver
    try:
        raw = await _api_call(state, COVERAGE_EVOLVE_SYSTEM, user,
                              **_evolve_endpoint(state),
                              max_tokens=4096, temperature=0.5,
                              label="evolve_coverage", want_json=True)
        obj = _parse_json(raw)
        if not isinstance(obj, dict):
            raise ValueError("meta-optimizer did not return a JSON object")
        cand_sys = str(obj.get("system") or "").strip()
        cand_tpl = str(obj.get("template") or "").strip()
        summary = str(obj.get("summary") or "").strip()
        reason = validate_coverage_prompt(cand_sys, cand_tpl)
        if reason:
            # Rejected, not applied. This is a normal outcome, not an outage: the
            # meta-optimizer proposed something that would have broken the parser.
            logger.error("/evolve_retrieval: REJECTED proposed prompt (%s); keeping v%d",
                         reason, cur_ver)
            summary = f"[rejected: {reason}] {summary}"
        elif cand_sys == cur_sys and cand_tpl == cur_tpl:
            logger.info("/evolve_retrieval: proposal identical to v%d, nothing to commit", cur_ver)
        else:
            new_ver = cur_ver + 1
            _atomic_write(path, json.dumps(
                {"version": new_ver, "system": cand_sys, "template": cand_tpl,
                 "summary": summary, "step": int(step)}, indent=1))
            changed = True
            hist.append({"version": new_ver, "step": int(step), "summary": summary,
                         "system": cand_sys, "template": cand_tpl, "outcomes": []})
            logger.warning("/evolve_retrieval: committed coverage prompt v%d -> %s",
                           new_ver, path)
        state.coverage_consec_failures = 0
        try:
            os.remove(_marker)
        except FileNotFoundError:
            pass
    except Exception as e:  # noqa: BLE001 — never break training
        state.coverage_consec_failures = getattr(state, "coverage_consec_failures", 0) + 1
        logger.error("/evolve_retrieval FAILED (%d consecutive), keeping v%d: %s: %s",
                     state.coverage_consec_failures, cur_ver, type(e).__name__, e)
        try:
            with open(_marker, "a") as f:
                f.write(f"{datetime.now().isoformat()} step={step} "
                        f"consec={state.coverage_consec_failures} {type(e).__name__}: {e}\n")
        except OSError:
            pass

    try:
        _atomic_write(os.path.join(state.args.log_dir, "coverage_evolve_history.json"),
                      json.dumps(hist, indent=1))
    except Exception as e:  # noqa: BLE001
        logger.warning("could not persist coverage history: {}".format(e))

    metrics = {"step": int(step), "version": new_ver, "changed": changed, **stats}
    with open(state.evolve_log, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(),
                            "kind": "coverage", **metrics, "summary": summary}) + "\n")
    logger.info(f"~ evolve_retrieval step={step} v={new_ver} changed={changed} "
                f"concordance={stats.get('concordance')} spread={stats.get('within_group_spread')}")
    if summary:
        logger.info(f"  coverage evolve summary: {summary[:400]}")
    return metrics


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


async def _attach_sft_target(state: ServerState, entry: dict) -> bool:
    """Pick the right SFT-target generator for this entry.

    rubric entries (healthbench, no ground_truth) -> gold answer written against
    the rubric; labeled entries (mcq/free with a ground truth) -> the original
    boxed-and-verified teacher trace.
    """
    style = (entry.get("reward_model") or {}).get("style", "")
    gt = (entry.get("reward_model") or {}).get("ground_truth", "")
    if style == "rubric" or (not gt and (entry.get("extra_info") or {}).get("rubric_items")):
        return await attach_rubric_gold_trace(state, entry)
    return await attach_teacher_trace(state, entry)


# Teacher-refusal detector. Matches the standard decline openers; deliberately
# anchored to the START of the answer so a legitimate clinical "I cannot
# recommend X for this patient" mid-answer is NOT caught.
_REFUSAL_RE = re.compile(
    r"^\W{0,10}(i\s*(?:'|\u2019)?\s*m sorry|i am sorry|sorry[,.]|i can\s*(?:'|\u2019)?\s*t\s+"
    r"(?:help|assist|provide|write|comply|do that)|i cannot\s+(?:help|assist|provide|write|comply)|"
    r"i\s*(?:'|\u2019)?\s*m not able to|i am unable to|as an ai\b|i won\s*(?:'|\u2019)?\s*t be able)",
    re.I)


RUBRIC_GOLD_SYSTEM_PROMPT = (
    "You are a senior physician writing the IDEAL response to a clinician's request, "
    "for use as a teaching example. You are shown the grading rubric that will be used "
    "to score responses. Write the response that earns EVERY positive criterion and "
    "triggers NONE of the negative ones.\n"
    "Rules:\n"
    "- First write your clinical reasoning inside <think>...</think>: what the request "
    "actually asks, what matters clinically, what could go wrong. Several sentences.\n"
    "- Then, after </think>, write the FINAL RESPONSE the clinician should receive.\n"
    "- The final response must read as a natural expert answer, NOT as a checklist of "
    "the rubric. Never mention the rubric, the criteria, or that you are being graded.\n"
    "- Be complete but not padded: the benchmark penalizes length beyond what the "
    "content needs. Aim for roughly 1500-2500 characters in the final response.\n"
    "- Answer in the SAME LANGUAGE as the clinician's request.\n"
    "- If the request rests on a false or unsafe premise, correct it plainly.\n"
    "- If key information is missing, ask for it as part of a useful answer."
)


def _format_rubric_for_teacher(items: list) -> str:
    lines = []
    for it in items or []:
        pts = it.get("points")
        txt = it.get("criterion_text") or it.get("criterion") or ""
        if txt:
            lines.append(f"[{pts:+g}] {txt}")
    return "\n".join(lines)


async def attach_rubric_gold_trace(state: ServerState, entry: dict) -> bool:
    """Rubric-mode SFT target: have the teacher write the IDEAL answer for the
    task, conditioned on the rubric, then self-grade it and keep it only if it
    actually earns most of the rubric.

    Unlike `attach_teacher_trace` (labeled tasks, verified by \boxed{} match),
    rubric tasks have no ground_truth — the rubric IS the specification, so the
    verification is "does the gold answer score well against its own rubric".
    Sets entry["reference_response"] on success.
    """
    items = (entry.get("extra_info") or {}).get("rubric_items") or []
    if not items:
        return False
    conv = (entry.get("extra_info") or {}).get("conversation") or entry.get("prompt") or []
    task_text = _render_conversation_user(conv) if conv else ""
    if not task_text.strip():
        return False

    rubric_block = _format_rubric_for_teacher(items)
    user_prompt = (
        f"Clinician request:\n{task_text}\n\n"
        f"Grading rubric (positive points must be earned, negative points must be avoided):\n"
        f"{rubric_block}\n\n"
        "Write <think>reasoning</think> then the final response."
    )
    min_score = float(os.environ.get("HB_GOLD_MIN_SCORE", "0.8"))

    for _ in range(state.args.teacher_retries + 1):
        try:
            raw = await _api_call(state, RUBRIC_GOLD_SYSTEM_PROMPT, user_prompt,
                                  max_tokens=state.args.teacher_max_tokens,
                                  temperature=0.6, label="rubric_gold")
        except Exception as e:
            logger.warning(f"rubric gold teacher failed: {type(e).__name__}: {e!r}")
            continue
        if not raw or "</think>" not in raw:
            # require the think/answer structure the student is trained to emit
            continue
        answer = raw.split("</think>")[-1].strip()
        if len(answer) < 200:
            continue
        # Refusal guard (dvd 2026-08-04): the teacher occasionally declines
        # ("I can't help with that", "I'm sorry, ...") — especially on red-team
        # tasks. A refusal is a fluent, high-confidence string that would train
        # the student to refuse, so drop it and retry rather than distill it.
        if _REFUSAL_RE.search(answer[:400]) or _REFUSAL_RE.search(raw[:200]):
            logger.info(f"~ rubric gold REFUSAL dropped: {answer[:90]!r}")
            state.stats["sft_gold_refusal"] = state.stats.get("sft_gold_refusal", 0) + 1
            continue
        # Self-verify: grade the gold answer with its own rubric; keep only if it
        # earns >= HB_GOLD_MIN_SCORE. This is the rubric analogue of the labeled
        # path's boxed-answer check — a gold target that fails its own rubric
        # would teach the student the wrong thing.
        try:
            score = await _grade_answer_with_rubric(state, task_text, answer, items)
        except Exception as e:
            logger.warning(f"rubric gold self-grade failed: {type(e).__name__}: {e!r}")
            score = None
        # STRICT (dvd 2026-08-04): reject unless the gold answer PROVABLY earns
        # its own rubric. An ungradable trace (score None) is rejected too — an
        # unverified target is exactly what this gate exists to prevent.
        if score is None or score < min_score:
            state.stats["sft_gold_lowscore"] = state.stats.get("sft_gold_lowscore", 0) + 1
            logger.info(f"~ rubric gold rejected (self-score {score} < {min_score})")
            continue
        entry["reference_response"] = raw.strip()
        entry.setdefault("extra_info", {})["gold_self_score"] = score
        return True
    return False


async def _grade_answer_with_rubric(state: ServerState, task_text: str, answer: str,
                                    items: list) -> float:
    """Grade `answer` against `items` with the same criterion-by-criterion
    contract the training reward uses. Returns achieved/total_positive."""
    conv_block = f"user: {task_text}\n\nassistant: {answer}"

    async def one(it):
        pts = float(it.get("points", 0))
        txt = it.get("criterion_text") or it.get("criterion") or ""
        prompt = (
            "Does the assistant's response meet this rubric criterion?\n\n"
            f"# Conversation\n{conv_block}\n\n# Rubric item\n[{pts:+g}] {txt}\n\n"
            'Return JSON: {"criteria_met": true or false}'
        )
        # temperature >= 0.5 keeps _api_call from adding reasoning_effort="none",
        # which gpt-chat-latest rejects with 400 (chat models don't take it).
        # The trapi branch strips temperature itself, so this only sets that flag.
        raw = await _api_call(state, "You are a careful grader. Return only JSON.",
                              prompt, max_tokens=512, temperature=0.6, label="gold_grade",
                              want_json=True)
        obj = _parse_json(raw) or {}
        return pts, bool(obj.get("criteria_met"))

    graded = await asyncio.gather(*[one(it) for it in items])
    total_pos = sum(p for p, _ in graded if p > 0) or 1.0
    achieved = sum(p for p, met in graded if met)
    return achieved / total_pos


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

            # Rubric mode runs its own single-mode pipeline (task + rubric) and
            # skips the diagnosis modes entirely.
            if state.rubric_mode:
                await _rubric_iteration(state, worker_id)
                continue

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
                if state.args.sft_mode and not await _attach_sft_target(state, entry):
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
                    if state.args.sft_mode and not await _attach_sft_target(state, entry):
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
                if state.args.sft_mode and not await _attach_sft_target(state, entry):
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


class EvolvePayload(BaseModel):
    """End-of-step prompt-evolution request from the trainer.

    `cases` is ~20 sampled rollouts of the step: the clinician task, the model's
    response, the rubric, which criteria were met, and the rubric score. The
    server analyzes them and rewrites the GAP_GUIDANCE of both generation prompts.
    """

    step: int
    cases: list[dict]
    # The trainer's most recent HELD-OUT validation score. This is the only
    # ungameable outcome signal the meta-optimizer gets: the train mean is the
    # solver's score on tasks the curriculum itself wrote, so it rises whenever
    # the curriculum gets easier. Optional so an older trainer still works.
    val_score: float | None = None
    # The step val_score was measured at. Needed to credit it to the version that
    # EARNED it: evolution runs just before validation, so the score arriving with
    # a round was earned by the version committed one interval earlier.
    val_step: int | None = None


class RetrievePayload(BaseModel):
    """Solver-facing retrieval request (the `search_medical_kb` rollout tool).

    The RL/eval policy issues a QUERY PLAN — up to `RETRIEVE_MAX_QUERIES` sub-queries
    covering different facets of the request — which is batch-embedded, matched
    against the read-only medical knowledge DB, merged round-robin, and (by default)
    compressed by the summarizer into a question-conditioned evidence brief. The
    result is injected back as a loss-masked tool turn.

    `query` (single string) is kept for the eval harness and the GPT-5.6 warm-start
    traces, which emit `<parameter=query>`; dropping it would invalidate that SFT set.
    """

    query: str | None = None            # legacy single-query form
    queries: list[str] | None = None    # query plan (preferred)
    question: str | None = None         # the clinician request, for question-conditioning
    top_k: int | None = None            # depth PER sub-query
    total: int | None = None            # merged passage budget
    summarize: bool | None = None       # None -> server default


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
    # Drain wikidoc telemetry off the /retrieve request path (batched, in a thread).
    workers.append(asyncio.create_task(_wikidoc_writer(STATE)))
    # Probe the summarizer ONCE at startup: a whole run silently training on the
    # raw-passage fallback distribution is the failure this warning exists to prevent.
    if args.summarizer_api_base:
        asyncio.create_task(_probe_summarizer(STATE))
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


# Retrieval ranking policy (source priors, junk exclusion, per-passage caps)
# lives in kb/retrieval.py so the offline trace generator builds SFT traces from
# exactly the passages a rollout would see. sys.path[0] is this file's directory
# when the server is launched as a script; be explicit so it also works under
# uvicorn/module launchers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb.retrieval import (  # noqa: E402
    MAX_QUERIES, MERGE_PER_SOURCE, MERGE_TOTAL, PER_QUERY_K,
    RetrieveConfig, format_passages, merge_ranked, rank_hits,
)


# ---------------------------------------------------------------------------
# Retrieval summarization
# ---------------------------------------------------------------------------
# Runs SERVER-SIDE, inside /retrieve, on purpose: offline SFT-trace generation and
# RL rollouts then see byte-identical evidence briefs. If the summarizer lived in
# the agent loop instead, the warm start would teach a passage distribution that
# does not exist at train time — the invariant kb/retrieval.py exists to protect.
SUMMARY_SYSTEM = (
    "You are a clinical evidence summarizer. You are given a clinician's request and "
    "reference passages retrieved from a medical knowledge base. Write a compact "
    "EVIDENCE BRIEF that a physician will use to answer the request.\n"
    "RULES:\n"
    "- Copy every number VERBATIM: doses, units, frequencies, thresholds, cutoffs, "
    "ages, durations, percentages, codes, trial names. Never round, convert, infer or "
    "combine numbers.\n"
    "- Include ONLY facts stated in the passages. Add nothing from your own knowledge. "
    "If the passages do not address part of the request, write one line: "
    "'Not covered: <topic>'.\n"
    "- Group by topic as short bullets. Tag each bullet with its passage number, e.g. [p3].\n"
    "- Drop passages that are off-topic, table-of-contents fragments, or duplicates.\n"
    "- Max 400 words. No preamble, no advice, and do NOT answer the request yourself."
)
SUMMARY_USER = "# Clinician request\n{question}\n\n# Retrieved passages\n{passages}\n\n# Evidence brief"


async def _probe_summarizer(s: ServerState) -> None:
    """One-shot startup reachability check; logs loudly if /retrieve will fall back."""
    try:
        r = await s.http_client.get(
            f"{s.args.summarizer_api_base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {s.args.summarizer_api_key}"}, timeout=20,
        )
        r.raise_for_status()
        logger.warning("summarizer OK: %s @ %s", s.args.summarizer_model,
                       s.args.summarizer_api_base)
    except Exception as e:
        s.summarizer_warned = True
        logger.error(
            "SUMMARIZER UNREACHABLE (%s: %s) at %s -- /retrieve will serve RAW passages "
            "for the whole run. Check the service before trusting these results.",
            type(e).__name__, e, s.args.summarizer_api_base,
        )


async def _summarize_passages(s: ServerState, question: str, passages: list[dict],
                              raw_text: str) -> tuple[str, bool, str | None]:
    """Compress merged passages into a question-conditioned evidence brief.

    Returns (text, summarized, fallback_reason). Falls back to the raw formatted
    passages on ANY failure — a summarizer outage must degrade context quality, never
    lose a rollout."""
    timeout = float(os.environ.get("RETRIEVE_SUMMARY_TIMEOUT", "45"))
    min_chars = int(os.environ.get("RETRIEVE_SUMMARY_MIN_CHARS", "200"))
    try:
        async with s.summary_sem, timed(s, "summarize"):
            brief = await asyncio.wait_for(
                _api_call(
                    s, SUMMARY_SYSTEM,
                    SUMMARY_USER.format(question=(question or "")[:4000], passages=raw_text),
                    # 1024, not 700: at 700 roughly a fifth of briefs stopped
                    # exactly at the cap, and because the "Not covered:" gap block
                    # is emitted LAST it was the part amputated -- it survived in
                    # 91% of complete briefs but only ~28% of truncated ones. The
                    # policy was being told what the KB holds while the statement
                    # of what it does NOT hold was silently cut, and nothing
                    # counted it: _summarize_passages only tracks outright failure,
                    # so a truncated brief logs as a success.
                    max_tokens=int(os.environ.get("RETRIEVE_SUMMARY_MAX_TOKENS", "1024")),
                    # >=0.5 would request a thinking channel; a thinking summarizer on
                    # the generation critical path blows the latency budget.
                    temperature=0.2, label="summarize",
                    api_base=s.args.summarizer_api_base,
                    api_key=s.args.summarizer_api_key,
                    model_name=s.args.summarizer_model,
                    provider_override=s.args.summarizer_provider,
                ),
                timeout=timeout,
            )
        brief = (brief or "").strip()
        if len(brief) < min_chars:
            raise ValueError(f"degenerate summary ({len(brief)} chars)")
        return brief, True, None
    except Exception as e:
        s.stats["summarizer_fail"] = s.stats.get("summarizer_fail", 0) + 1
        if not s.summarizer_warned:
            s.summarizer_warned = True
            logger.error("SUMMARIZER FAILING (%s: %s) -- serving raw passages",
                         type(e).__name__, e)
        return raw_text, False, type(e).__name__


async def _wikidoc_writer(s: ServerState) -> None:
    """Batched background drain for the wikidoc title queue (never on the hot path)."""
    path = os.path.join(s.args.log_dir, "wikidoc_retrieved_titles.jsonl")

    def _append(rows: list[dict]) -> None:
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    while True:
        try:
            batch = [await s.wikidoc_q.get()]
            while not s.wikidoc_q.empty() and len(batch) < 500:
                batch.append(s.wikidoc_q.get_nowait())
            await asyncio.to_thread(_append, batch)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"wikidoc writer: {type(e).__name__}: {e}")
            await asyncio.sleep(5)


def _queue_wikidoc_titles(s: ServerState, hits: list[dict]) -> None:
    """Hand retrieved wikidoc titles to the background writer. NON-BLOCKING.

    This used to do a blocking open()/write() on the event loop while holding the
    PROCESS-WIDE `log_lock` (shared with /report and the generation workers), once
    per retrieval, iterating the whole raw over-fetch. Under concurrent rollouts
    every retrieval serialized behind every other retrieval's file IO — and the
    over-fetch grows from ~40 rows to ~320 under a 4-query plan. Telemetry is never
    worth a rollout: on a full queue we drop.
    """
    q = getattr(s, "wikidoc_q", None)
    if q is None:
        return
    seen = s.wikidoc_seen
    for h in hits:
        if h.get("source") != "wikidoc":
            continue
        title = (h.get("text") or "").strip()
        if not title:
            continue
        eid = h.get("entry_id", "") or title[:120]
        if eid in seen:
            continue
        if len(seen) > 200_000:
            seen.clear()
        seen.add(eid)
        try:
            q.put_nowait({"title": title, "entry_id": h.get("entry_id", "")})
        except asyncio.QueueFull:
            return


@app.post("/retrieve")
async def retrieve(payload: RetrievePayload):
    """Run a query PLAN against the medical knowledge DB and return grounded context.

    Pipeline: batch-embed 1..MAX_QUERIES sub-queries -> per-query Milvus search with
    the junk rows excluded in-query -> per-query re-rank by cosine * source prior ->
    round-robin merge with exact/near dedup and a global source cap -> optional
    question-conditioned summarization into an evidence brief.

    Read-only (no STATE mutation beyond telemetry). NEVER raises to the caller: every
    failure degrades (summarizer down -> raw passages; DB down -> empty result) so a
    rollout is never lost to a transient hiccup."""
    s = STATE
    queries = _coerce_query_plan(payload)
    if not queries:
        return {"passages": [], "text": "No query provided.", "summarized": False,
                "n_queries": 0, "n_merged": 0, "merge_stats": {}, "chars": 0,
                "fallback_reason": "empty_query"}

    top_k = int(payload.top_k or s.args.retrieve_top_k or PER_QUERY_K)
    total = int(payload.total or s.args.retrieve_total or MERGE_TOTAL)
    cfg = RetrieveConfig()

    # Measure the QUEUE wait separately from the work: a request that waits 90s for
    # a semaphore slot and then completes in 12s looks perfectly healthy in the
    # work-only timings, which is exactly how the client-side timeouts went unnoticed.
    s.retrieve_waiting += 1
    _q0 = time.monotonic()
    try:
        await s.retrieve_sem.acquire()
    finally:
        s.retrieve_waiting -= 1
    s.timings["retrieve_queue_wait"].append(time.monotonic() - _q0)
    try:
        try:
            per_query_hits = await _milvus_search_multi(
                s, queries, top_k=top_k * cfg.fetch_mult, filter_expr=cfg.exclude_expr
            )
        except Exception as e:  # defensive: retrieval must never kill a rollout
            logger.warning(f"/retrieve failed for {queries[0][:60]!r}: {type(e).__name__}: {e!r}")
            per_query_hits = []

        # Rank WITHIN each sub-query first, so the merge sees each facet's own best
        # passages rather than one global list dominated by the strongest query.
        per_query = [rank_hits(h, top_k, cfg) for h in per_query_hits]
        passages, merge_stats = merge_ranked(
            per_query, total=total, per_source_cap=MERGE_PER_SOURCE
        )
        for h in per_query_hits:
            _queue_wikidoc_titles(s, h)

        raw_text = format_passages(passages)
        text, summarized, reason = raw_text, False, None
        want_summary = s.args.summarizer_api_base and (
            payload.summarize if payload.summarize is not None else True
        )
        if want_summary and passages:
            text, summarized, reason = await _summarize_passages(
                s, payload.question or queries[0], passages, raw_text
            )
    finally:
        s.retrieve_sem.release()

    return {
        "passages": passages,
        "text": text,
        "raw_text": raw_text,
        "summarized": summarized,
        "queries": queries,
        "n_queries": len(queries),
        "n_merged": len(passages),
        "merge_stats": merge_stats,
        "chars": len(text),
        "fallback_reason": reason,
    }


def _coerce_query_plan(payload: RetrievePayload) -> list[str]:
    """Accept a query plan in every shape a model or caller might send it."""
    raw: list = []
    if payload.queries:
        raw = list(payload.queries)
    elif payload.query:
        q = payload.query.strip()
        if q.startswith("["):
            try:
                parsed = json.loads(q)
                raw = parsed if isinstance(parsed, list) else [q]
            except Exception:
                raw = q.splitlines()
        else:
            raw = q.splitlines()
    out, seen = [], set()
    for q in raw:
        q = str(q).strip().lstrip("-*0123456789. ").strip('"').strip()
        if len(q) >= 4 and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:MAX_QUERIES]


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
        # Requests parked on the /retrieve semaphore. A nonzero steady-state value
        # means the endpoint is the rollout bottleneck and clients are at risk of
        # timing out on queue wait alone (see retrieve_queue_wait in timings).
        "retrieve_waiting": getattr(s, "retrieve_waiting", 0),
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
    if (entry.get("extra_info") or {}).get("question_id") in STATE.__dict__.get("dead_qids", set()):
        return None
    # Backstop: never serve an SFT image row whose files are missing on this
    # pod (would crash the trainer's multimodal tokenizer). Covers replay /
    # history / cold-start-seed sources that bypass the worker's direct-mode gate.
    if getattr(s.args, "sft_mode", False) and not _entry_images_present(entry):
        s.stats["served_missing_image_skipped"] += 1
        return None
    missing = _missing_fields(s, entry)
    if missing == ["reference_response"] and getattr(s.args, "sft_mode", False):
        try:
            if await _attach_sft_target(s, entry):
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
    # Solvability filter (2026-08-03 seeded-diag post-mortem): a task whose
    # first >=8 graded rollouts show no variance or an extreme mean gives GRPO
    # no gradient (measured: 30% of groups all-zero, 17 tasks never scored >0).
    # Evict such tasks so they are never served again.
    _accs = s.__dict__.setdefault("report_accs", {})
    _dead = s.__dict__.setdefault("dead_qids", set())
    lst = _accs.setdefault(payload.question_id, [])
    lst.append(acc)
    # Thresholds are deliberately tighter than the original (>=8 reports, mean
    # outside [0.05, 0.95]). That version only caught tasks that were PERFECTLY
    # saturated, so a task every rollout scored ~0.90 on survived indefinitely
    # while contributing almost no gradient -- and the audit measured 65% of
    # rollouts at exactly 1.0 with 28% of groups at zero variance. Acting at 4
    # reports rather than 8 also stops a dead task from burning a second full
    # group before it is recognised. `var` is over a [0,1] score, so 1e-4 is
    # still "every rollout scored the same to within 1%".
    _n_evict = int(os.environ.get("HB_EVICT_MIN_REPORTS", "4"))
    _hi = float(os.environ.get("HB_EVICT_MAX_MEAN", "0.88"))
    _lo = float(os.environ.get("HB_EVICT_MIN_MEAN", "0.05"))
    if len(lst) >= _n_evict and payload.question_id not in _dead:
        m = sum(lst) / len(lst)
        var = sum((a - m) ** 2 for a in lst) / len(lst)
        if var < 1e-4 or m < _lo or m > _hi:
            _dead.add(payload.question_id)
            before = len(s.history)
            # s.history is a deque (maxlen-bounded), and deques reject slice
            # assignment — `s.history[:] = [...]` raises TypeError, which turned
            # every eviction into a 500 on /report and left the unsolvable task in
            # the pool anyway. Rebuild in place instead.
            _keep = [e for e in s.history
                     if (e.get("extra_info") or {}).get("question_id") != payload.question_id]
            s.history.clear()
            s.history.extend(_keep)
            s.stats["evicted_unsolvable"] = s.stats.get("evicted_unsolvable", 0) + 1
            logger.info(f"~ evicted qid={payload.question_id} (n={len(lst)} mean={m:.3f} "
                        f"var={var:.4f}); history {before}->{len(s.history)}")
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


@app.post("/evolve")
async def evolve(payload: EvolvePayload):
    """Evolve the generation prompts from a step's sampled rollouts. Serialized
    (evolve_lock) so concurrent calls can't interleave guidance rewrites."""
    s = STATE
    if not s.rubric_mode or s.prompt_store is None:
        raise HTTPException(status_code=400, detail="evolve requires --rubric_mode")
    if not payload.cases:
        return {"step": payload.step, "n_cases": 0, "skipped": "no cases"}
    async with s.evolve_lock:
        return await _evolve_prompts(s, payload.step, payload.cases,
                                     payload.val_score, payload.val_step)


class EvolveRetrievalPayload(BaseModel):
    """End-of-step retrieval-REWARD evolution request.

    `cases` are this step's SEARCHING rollouts, each carrying the group id, the
    coverage verdict, the ANSWER score, and the query plan. The answer score is
    the pre-bonus one, which is what makes the concordance measurement honest:
    the judge under evaluation contributed nothing to it.
    """

    step: int
    cases: list[dict]


@app.post("/evolve_retrieval")
async def evolve_retrieval(payload: EvolveRetrievalPayload):
    """Rewrite the coverage judge from measured discrimination. Serialised on the
    same lock as /evolve so two rewrites cannot interleave."""
    # Module global, exactly like the sibling /evolve above. `app.state.server` is
    # never assigned anywhere in this file (only `app.state.args`), so reading it
    # raised KeyError -> HTTP 500 on EVERY call. The trainer's raise_for_status is
    # swallowed by its blanket except, so this failed completely silently: the
    # round logged "FAILED" once and returned {}, and because the
    # COVERAGE_EVOLVE_FAILING marker is written INSIDE the handler it was never
    # dropped either, so a marker-watching monitor read healthy forever.
    s = STATE
    if not s.rubric_mode or s.prompt_store is None:
        raise HTTPException(status_code=400, detail="evolve_retrieval requires --rubric_mode")
    async with s.evolve_lock:
        return await _evolve_retrieval_reward(s, payload.step, payload.cases)


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
    parser.add_argument("--seeds_path", default="",
                        help="JSONL of seed targets (typically train.jsonl). "
                             "Required unless --rubric_mode (which synthesizes "
                             "use_case x specialty seeds).")
    # --- Rubric mode (HealthBench-Professional task + rubric co-generation) ---
    parser.add_argument("--rubric_mode", action="store_true",
                        help="Generate open-ended clinician TASK + co-generated "
                             "HealthBench-Professional rubric per item (no MCQ/"
                             "diagnosis). Reward is the rubric, graded by self.")
    parser.add_argument("--coverage_prompt_file", default="",
                        help="Where /evolve_retrieval writes the evolved coverage-judge "
                             "prompt. The reward reads the SAME path via "
                             "HB_COVERAGE_PROMPT_FILE and reloads it on mtime change. "
                             "Defaults to <prompt_dir>/coverage_prompt.json.")
    parser.add_argument("--evolve_api_base", default="",
                        help="Endpoint for the /evolve meta-optimizer (defaults to --api_base). "
                             "Set with --evolve_model_name to run curriculum evolution on an "
                             "EXTERNAL model instead of the local teacher.")
    parser.add_argument("--evolve_api_key", default="")
    parser.add_argument("--evolve_model_name", default="",
                        help="Model id for the /evolve meta-optimizer. Empty = use the local teacher.")
    parser.add_argument("--evolve_provider", default="",
                        help="Provider shaping for the evolve endpoint (vllm|trapi|openai|kimi).")
    parser.add_argument("--prompt_dir", default="",
                        help="Directory holding the evolvable, file-backed prompts "
                             "(query_proposer.txt, task_rubric_generator.txt). "
                             "Defaults to {log_dir}/prompts. Rubric mode only.")
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
    # NOTE: --milvus_top_k belongs to the GENERATION pipeline (question proposal,
    # knowledge grounding). The rollout-facing /retrieve endpoint has its own depth
    # knobs below; the two were previously conflated, so an omitted `top_k` from the
    # rollout tool silently got the generation pipeline's 16 instead of the tuned 5.
    parser.add_argument("--milvus_top_k", type=int, default=16)
    # ---- /retrieve (RL rollout tool) ----
    parser.add_argument("--retrieve_top_k", type=int, default=None,
                        help="Passages fetched PER sub-query (default kb.retrieval.PER_QUERY_K=5).")
    parser.add_argument("--retrieve_total", type=int, default=None,
                        help="Merged passage budget across all sub-queries (default 16).")
    parser.add_argument("--summarizer_api_base", default=os.environ.get("SUMMARIZER_API_BASE", ""),
                        help="OpenAI-compatible base URL of the FROZEN summarizer serving the "
                             "same checkpoint the SFT traces were built from. Empty disables "
                             "summarization and /retrieve returns raw passages.")
    parser.add_argument("--summarizer_api_key", default=os.environ.get("SUMMARIZER_API_KEY", "EMPTY"))
    parser.add_argument("--summarizer_model", default=os.environ.get("SUMMARIZER_MODEL", ""))
    parser.add_argument("--summarizer_provider", default=os.environ.get("SUMMARIZER_PROVIDER", "vllm"))
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

    # Validate / default mode-specific args.
    if args.rubric_mode:
        if not args.prompt_dir:
            args.prompt_dir = os.path.join(args.log_dir, "prompts")
    elif not args.seeds_path:
        parser.error("--seeds_path is required unless --rubric_mode is set")

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
