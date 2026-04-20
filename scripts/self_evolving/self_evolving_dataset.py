"""
Self-evolving dataset with multi-agent RAG-based question generation.

Three LLM agents (same API, different system prompts) work together:

1. **Query Proposer**: Takes a target training question, outputs 10 search queries
   (comprehensive sentences) used to retrieve relevant medical knowledge from Milvus.

2. **Question Generator**: For each retrieved knowledge passage, synthesizes a NEW
   training question using the target question + retrieved knowledge + solver's
   recent accuracy. Generates both multiple-choice AND free-response questions.
   Creates complex scenarios combining multiple knowledge sources; does not copy
   directly from the retrieved text.

3. **Question Validator**: Takes the generated question and runs a fresh Milvus
   query to check if it contradicts database knowledge. Lenient — questions
   outside the DB's knowledge are still accepted. Only rejects clear contradictions.

Each accepted question is trained exactly once. Rejected questions are logged but
not served.
"""

import json
import logging
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import requests
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


# ======================================================================
# AGENT SYSTEM PROMPTS
# ======================================================================

QUERY_PROPOSER_SYSTEM_PROMPT = """\
You are a medical information retrieval expert. Given a training question, propose 10 \
diverse search queries for retrieving medical knowledge from a multimodal database \
(PubMedQA abstracts, MIRAGE MCQs, MedRAG textbooks, PubMed, Wikipedia, PMC-VQA, CLIMB \
clinical QA across chest X-ray, derm, CT, ECG, fundus, MRI, mammography, ultrasound, \
pathology).

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

Output ONLY a JSON array of 10 strings in the above order. No markdown, no explanation.
["query 1", "query 2", ..., "query 10"]"""


QUESTION_GENERATOR_SYSTEM_PROMPT = """\
You are a medical educator creating training questions for a medical AI. You are given: \
(1) a reference training question, (2) a relevant passage, (3) the solver's recent \
accuracy, (4) the REQUIRED format for this question.

SYNTHESIZE a NEW question combining the reference topic with the retrieved passage. The \
question MUST:

- NOT be a direct copy or paraphrase of either source.
- Require the PASSAGE to answer — a well-informed clinician without the passage should \
  have to guess between at least two plausible options / phrasings. If the answer is \
  obvious from general medical training alone, the question is too easy — reject it \
  yourself and regenerate.
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

Output ONLY a JSON object. No markdown, no explanation.

For MCQ (required_format="mcq"):
{{"format": "mcq", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}

For free response (required_format="free"):
{{"format": "free", "question": "...", "answer": "short expected answer"}}"""


QUESTION_VALIDATOR_SYSTEM_PROMPT = """\
You are a medical fact-checker evaluating whether a candidate training question is good \
enough to train a medical AI. You are given:
(1) A candidate question + its proposed answer (+ options if MCQ).
(2) Retrieved passages from a medical knowledge database.

Decide a verdict. Reject (verdict="reject") if ANY of the following:

A. CONTRADICTION — the passages directly and unambiguously state something that makes \
   the proposed answer wrong.
B. UNGROUNDED TRIVIA — the answer can be decided with confidence from general medical \
   training alone, without any passage. In other words, it's a fact so common that the \
   retrieved knowledge adds nothing (e.g. "what organ produces insulin?"). We want \
   questions that NEED the passages.
C. AMBIGUOUS — for MCQ, more than one option is defensibly correct, OR the "correct" \
   answer is only marginally better than a distractor. For free-response, multiple \
   short phrases would all be correct.
D. MALFORMED — poorly phrased, incoherent, grammatically broken, or has silly \
   distractors ("none of the above", obvious nonsense) that make the right answer \
   trivial to pick by elimination.

Otherwise verdict="ok". Out-of-database but well-formed + grounded questions are OK.

Output ONLY a JSON object with a short reason. No markdown, no explanation.
{{"verdict": "ok" or "reject", "reason": "..."}}"""


# ======================================================================
# SOLVER PROMPTS
# ======================================================================

SOLVER_SYSTEM_PROMPT_MCQ = (
    "You are a medical expert. Read the question carefully and choose the best answer. "
    "You FIRST think about the reasoning process as an internal monologue enclosed in "
    "<think> </think> tags. The final answer MUST BE a single letter (A, B, C, or D) "
    "wrapped in \\boxed{}.\n\n"
    "Example format:\n"
    "<think>\n[Your reasoning here]\n</think>\n\\boxed{C}"
)

SOLVER_SYSTEM_PROMPT_FREE = (
    "You are a medical expert. Answer the question with a short specific phrase. "
    "You FIRST think about the reasoning process as an internal monologue enclosed in "
    "<think> </think> tags. The final answer MUST BE a short phrase (1-15 words) "
    "wrapped in \\boxed{}.\n\n"
    "Example format:\n"
    "<think>\n[Your reasoning here]\n</think>\n\\boxed{acute pancreatitis}"
)


class SelfEvolvingDataset(RLHFDataset):
    """Multi-agent RAG-based self-evolving dataset.

    Dynamic mode pipeline (per target question):
      1. QueryProposer generates 10 search queries.
      2. For each query, retrieve top-K relevant entries from Milvus.
      3. For each retrieved entry, QuestionGenerator synthesizes 1 new question.
      4. QuestionValidator checks each new question against the DB (re-retrieve).
      5. Accepted questions are added to the pool. Rejected are logged.

    Inherits from RLHFDataset so the standard `__getitem__` path (message building,
    processor/image handling, tokenizer access) applies to generated questions.
    `self.dataframe` is replaced by the growing `_generated_questions` list.

    Only used for the training dataset. Validation uses RLHFDataset directly
    (see `create_rl_dataset` in verl/trainer/main_ppo.py).
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Load seed targets (the original test.jsonl questions) via the parent
        # loader so we also inherit tokenizer/processor/prompt-key setup.
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )
        self.target_questions = list(self.dataframe)

        se_config = config.self_evolving
        self.api_base = se_config.api_base
        self.api_key = se_config.get("api_key", "EMPTY")
        self.model_name = se_config.model_name
        self.accuracy_window = se_config.get("accuracy_window", 32)
        self.log_dir = se_config.get("log_dir", "/scratch/self_evolving_datasets/logs")
        self.dataset_length = se_config.get("dataset_length", 100000)

        # Milvus config
        self.milvus_uri = se_config.get("milvus_uri", "http://localhost:19531")
        self.milvus_token = se_config.get("milvus_token", "root:Milvus")
        self.milvus_collection = se_config.get("milvus_collection", "medical_knowledge")
        self.milvus_top_k = se_config.get("milvus_top_k", 16)

        # Embedding API config (same vLLM instance, but pooling runner)
        self.embed_api_base = se_config.get("embed_api_base", self.api_base)
        self.embed_model = se_config.get("embed_model", "Qwen/Qwen3-VL-Embedding-2B")

        # Pipeline params
        self.n_queries = se_config.get("n_queries", 10)
        self.questions_per_query = se_config.get("questions_per_query", 1)
        self.no_label = se_config.get("no_label", False)

        print(f"SelfEvolvingDataset: dynamic mode with {len(self.target_questions)} seed targets")
        print(f"  Milvus: {self.milvus_uri} / {self.milvus_collection}")
        print(f"  LLM API: {self.api_base} / {self.model_name}")

        # Set up question log file
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.question_log_path = os.path.join(self.log_dir, f"multi_agent_questions_{timestamp}.jsonl")
        self.rejected_log_path = os.path.join(self.log_dir, f"rejected_questions_{timestamp}.jsonl")
        print(f"  Questions log: {self.question_log_path}")
        print(f"  Rejected log: {self.rejected_log_path}")

        # Lazy-init Milvus client (so dataloader workers can share config but each
        # worker creates its own connection).
        self._milvus_client = None

        # State
        self.target_idx = 0
        self.cycle = 0
        self._question_counter = 0
        self._format_counter = 0  # alternates MCQ/free across queries for ~50/50 mix
        self._wandb_question_table = None
        self._generated_questions: list[dict] = []
        self._accuracy_history: list[float] = []

        # Per-target stats for logging
        self._stats = {
            "total_queries": 0,
            "total_generated": 0,
            "total_accepted": 0,
            "total_rejected": 0,
        }

        # Parent's __getitem__ reads from self.dataframe; point it at the growing
        # list of generated questions.
        self.dataframe = self._generated_questions

        # Pre-generate a seed batch
        self._ensure_questions_available(32)

    # --------------------------------------------------------------
    # Dataset protocol (override RLHFDataset)
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, item: int) -> dict:
        self._ensure_questions_available(item + 1)
        return super().__getitem__(item)

    def on_batch_end(self, batch: DataProto) -> None:
        if "acc" in batch.non_tensor_batch:
            for acc in batch.non_tensor_batch["acc"]:
                self._accuracy_history.append(float(acc))
        elif "rm_scores" in batch.batch:
            scores = batch.batch["rm_scores"].sum(dim=-1).tolist()
            for s in scores:
                self._accuracy_history.append(1.0 if s > 0.5 else 0.0)
        if len(self._accuracy_history) > self.accuracy_window:
            self._accuracy_history = self._accuracy_history[-self.accuracy_window:]

    def _get_accuracy_stats(self) -> dict:
        if not self._accuracy_history:
            return {"mean": 0.5, "count": 0}
        return {
            "mean": sum(self._accuracy_history) / len(self._accuracy_history),
            "count": len(self._accuracy_history),
        }

    # --------------------------------------------------------------
    # Milvus retrieval
    # --------------------------------------------------------------
    def _get_milvus_client(self):
        if self._milvus_client is None:
            from pymilvus import MilvusClient
            self._milvus_client = MilvusClient(uri=self.milvus_uri, token=self.milvus_token)
        return self._milvus_client

    def _embed_text(self, text: str) -> list[float]:
        """Get embedding via vLLM OpenAI-compatible /v1/embeddings endpoint."""
        url = f"{self.embed_api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.embed_model, "input": [text[:2000]]}
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    def _milvus_search(self, query_text: str, top_k: int = None) -> list[dict]:
        """Embed query text and search Milvus. Returns list of hit entities."""
        top_k = top_k or self.milvus_top_k
        try:
            embedding = self._embed_text(query_text)
        except Exception as e:
            logger.warning(f"Embedding failed for query '{query_text[:60]}': {e}")
            return []
        try:
            client = self._get_milvus_client()
            results = client.search(
                collection_name=self.milvus_collection,
                data=[embedding],
                limit=top_k,
                output_fields=["source_dataset", "modality", "content_type",
                               "text_content", "question", "answer"],
            )
            hits = []
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
                        "score": hit["distance"],
                    })
            return hits
        except Exception as e:
            logger.warning(f"Milvus search failed: {e}")
            return []

    # --------------------------------------------------------------
    # LLM API
    # --------------------------------------------------------------
    def _api_call(self, system_prompt: str, user_prompt: str,
                  max_tokens: int = 2048, temperature: float = 0.8) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _parse_json_response(self, response: str, expect_array: bool = False):
        if expect_array:
            m = re.search(r"\[.*\]", response, re.DOTALL)
        else:
            m = re.search(r"\{.*\}", response, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in response: {response[:300]}")
        return json.loads(m.group())

    # --------------------------------------------------------------
    # Agent 1: Query Proposer
    # --------------------------------------------------------------
    def _agent_query_proposer(self, target: dict) -> list[str]:
        target_question = target.get("extra_info", {}).get("question", "") or \
                          self._extract_user_text(target)
        user_prompt = (
            f"Target training question:\n{target_question}\n\n"
            f"Generate {self.n_queries} diverse search queries."
        )
        response = self._api_call(
            QUERY_PROPOSER_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=1024,
            temperature=0.8,
        )
        queries = self._parse_json_response(response, expect_array=True)
        if not isinstance(queries, list):
            raise ValueError(f"Query proposer did not return a list: {response[:300]}")
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            raise ValueError("Query proposer returned no valid queries")
        return queries[:self.n_queries]

    # --------------------------------------------------------------
    # Agent 2: Question Generator
    # --------------------------------------------------------------
    def _agent_question_generator(self, target_question: str,
                                  knowledge_passage: str,
                                  accuracy_stats: dict,
                                  required_format: str) -> dict:
        sys_prompt = QUESTION_GENERATOR_SYSTEM_PROMPT.format(
            required_format=required_format,
            accuracy=accuracy_stats["mean"],
            accuracy_count=accuracy_stats["count"],
        )
        user_prompt = (
            f"Reference training question:\n{target_question}\n\n"
            f"Retrieved medical knowledge:\n{knowledge_passage}\n\n"
            f"Synthesize one new training question in the required format "
            f"({required_format})."
        )
        response = self._api_call(sys_prompt, user_prompt, max_tokens=1024, temperature=0.9)
        q = self._parse_json_response(response, expect_array=False)

        fmt = q.get("format", "").lower()
        question = q.get("question", "").strip()
        answer = str(q.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"Missing question/answer: {q}")
        if fmt != required_format:
            raise ValueError(f"Generator returned format={fmt}, required {required_format}")
        if fmt == "mcq":
            options = q.get("options", {})
            if not isinstance(options, dict) or len(options) < 2:
                raise ValueError(f"MCQ missing options: {q}")
            answer_upper = answer.upper()[:1]
            if answer_upper not in options:
                raise ValueError(f"MCQ answer {answer} not in options {list(options)}")
            q["format"] = "mcq"
            q["answer"] = answer_upper
            q["options"] = options
        else:
            q["format"] = "free"
            q["answer"] = answer
        return q

    # --------------------------------------------------------------
    # Agent 3: Question Validator
    # --------------------------------------------------------------
    def _agent_validator(self, generated: dict) -> tuple[bool, str]:
        query_text = generated["question"]
        if generated.get("format") == "mcq":
            query_text += " " + " ".join(generated.get("options", {}).values())
        hits = self._milvus_search(query_text, top_k=3)

        if not hits:
            return True, "no retrieval results — out-of-knowledge, accepted"

        passages = "\n\n".join(
            f"[{h['source']}] {h['text']}" for h in hits[:3]
        )
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
            f"{q_text}\n\n"
            f"Retrieved passages from the database:\n{passages}\n\n"
            "Does the proposed answer CONTRADICT the retrieved knowledge?"
        )
        try:
            response = self._api_call(
                QUESTION_VALIDATOR_SYSTEM_PROMPT,
                user_prompt,
                max_tokens=256,
                temperature=0.2,
            )
            result = self._parse_json_response(response, expect_array=False)
            verdict = result.get("verdict", "").lower()
            reason = result.get("reason", "")
            if verdict in ("reject", "contradict"):
                return False, reason
            return True, reason
        except Exception as e:
            logger.warning(f"Validator failed, accepting: {e}")
            return True, f"validator error: {e}"

    # --------------------------------------------------------------
    # Full generation pipeline for one target
    # --------------------------------------------------------------
    def _extract_user_text(self, target: dict) -> str:
        for msg in target.get("prompt", []):
            if msg.get("role") == "user":
                content = msg["content"]
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return " ".join(texts)
        return ""

    def _build_entry(self, generated: dict, target: dict, passage: str, query: str) -> dict:
        self._question_counter += 1
        target_id = target.get("extra_info", {}).get("pubmed_id", "") or \
                    target.get("id", "") or f"t{self.target_idx}"

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

        if self.no_label:
            gt = ""

        return {
            "data_source": "self_evolving",
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "reward_model": {
                "style": style,
                "ground_truth": gt,
            },
            "extra_info": {
                "index": self._question_counter,
                "split": "train",
                "source": "self_evolving_multi_agent",
                "cycle": self.cycle,
                "target_idx": self.target_idx,
                "target_id": str(target_id),
                "format": generated["format"],
                "question": generated["question"],
                "answer": generated["answer"],
                "options": generated.get("options", {}) if generated["format"] == "mcq" else {},
                "passage": passage[:1200],
                "retrieval_query": query,
            },
        }

    def _ensure_questions_available(self, min_count: int) -> None:
        while len(self._generated_questions) < min_count:
            self._generate_next_target()

    def _generate_next_target(self) -> None:
        target = self.target_questions[self.target_idx]
        stats = self._get_accuracy_stats()
        target_question = target.get("extra_info", {}).get("question", "") or \
                          self._extract_user_text(target)

        logger.info(f"Target {self.target_idx} (cycle {self.cycle}): "
                    f"{target_question[:100]}")

        # Agent 1: Query Proposer
        queries = []
        for attempt in range(3):
            try:
                queries = self._agent_query_proposer(target)
                break
            except Exception as e:
                logger.warning(f"QueryProposer attempt {attempt + 1}/3 failed: {e}")
        if not queries:
            logger.warning("QueryProposer failed 3x — advancing target with no questions.")
            self._advance_target()
            return
        self._stats["total_queries"] += len(queries)

        # Agents 2 & 3: for each retrieved passage, generate + validate.
        # Alternate MCQ/free across queries so we get a balanced mix (~50/50)
        # rather than all-MCQ output.
        new_entries = []
        rejected = []
        for q_idx, query in enumerate(queries):
            hits = self._milvus_search(query, top_k=1)
            if not hits:
                continue
            passage = hits[0]["text"]
            required_format = "mcq" if self._format_counter % 2 == 0 else "free"
            self._format_counter += 1
            for _ in range(self.questions_per_query):
                try:
                    gen = self._agent_question_generator(
                        target_question, passage, stats, required_format
                    )
                except Exception as e:
                    logger.warning(f"Generator failed on query {q_idx} ({required_format}): {e}")
                    continue
                self._stats["total_generated"] += 1
                try:
                    ok, reason = self._agent_validator(gen)
                except Exception as e:
                    ok, reason = True, f"validator error: {e}"
                if ok:
                    entry = self._build_entry(gen, target, passage, query)
                    new_entries.append(entry)
                    self._stats["total_accepted"] += 1
                else:
                    rejected.append({"question": gen, "reason": reason,
                                     "retrieval_query": query, "passage": passage[:300]})
                    self._stats["total_rejected"] += 1

        self._generated_questions.extend(new_entries)
        self._log_accepted(new_entries, target, queries)
        self._log_rejected(rejected, target)
        logger.info(
            f"  accepted={len(new_entries)}  rejected={len(rejected)}  "
            f"(total accepted={self._stats['total_accepted']}, "
            f"rejected={self._stats['total_rejected']})"
        )
        self._advance_target()

    def _advance_target(self) -> None:
        self.target_idx += 1
        if self.target_idx >= len(self.target_questions):
            self.target_idx = 0
            self.cycle += 1
            print(f"SelfEvolvingDataset: completed cycle {self.cycle}")

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------
    def _log_accepted(self, entries: list[dict], target: dict, queries: list[str]) -> None:
        target_question = target.get("extra_info", {}).get("question", "") or \
                          self._extract_user_text(target)[:200]
        stats = self._get_accuracy_stats()
        records = []
        with open(self.question_log_path, "a") as f:
            for entry in entries:
                ei = entry.get("extra_info", {})
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "cycle": self.cycle,
                    "target_idx": self.target_idx,
                    "solver_accuracy": stats["mean"],
                    "solver_accuracy_count": stats["count"],
                    "target_question": target_question,
                    "format": ei.get("format", ""),
                    "question": ei.get("question", ""),
                    "answer": ei.get("answer", ""),
                    "options": ei.get("options", {}),
                    "retrieval_query": ei.get("retrieval_query", ""),
                    "passage": ei.get("passage", ""),
                    "queries_used": queries,
                    "index": ei.get("index", -1),
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)
        self._log_wandb(records)

    def _log_rejected(self, rejected: list[dict], target: dict) -> None:
        if not rejected:
            return
        target_question = target.get("extra_info", {}).get("question", "") or \
                          self._extract_user_text(target)[:200]
        with open(self.rejected_log_path, "a") as f:
            for r in rejected:
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "cycle": self.cycle,
                    "target_idx": self.target_idx,
                    "target_question": target_question,
                    "rejected_question": r.get("question", {}),
                    "reason": r.get("reason", ""),
                    "retrieval_query": r.get("retrieval_query", ""),
                    "passage": r.get("passage", ""),
                }
                f.write(json.dumps(record) + "\n")

    def _log_wandb(self, records: list[dict]) -> None:
        try:
            import wandb
            if wandb.run is None:
                return
            columns = ["cycle", "target_idx", "solver_accuracy", "target_question",
                       "format", "question", "answer", "options",
                       "retrieval_query", "passage"]
            if self._wandb_question_table is None:
                self._wandb_question_table = wandb.Table(columns=columns)
            new_table = wandb.Table(columns=columns, data=self._wandb_question_table.data)
            for r in records:
                new_table.add_data(
                    r["cycle"],
                    r["target_idx"],
                    r["solver_accuracy"],
                    r["target_question"][:500],
                    r["format"],
                    r["question"][:500],
                    str(r["answer"])[:200],
                    json.dumps(r["options"])[:500] if r.get("options") else "",
                    r.get("retrieval_query", "")[:300],
                    r.get("passage", "")[:800],
                )
            self._wandb_question_table = new_table
            wandb.log({"proposer/questions": new_table}, commit=False)
            wandb.log({
                "proposer/total_accepted": self._stats["total_accepted"],
                "proposer/total_rejected": self._stats["total_rejected"],
                "proposer/total_queries": self._stats["total_queries"],
                "proposer/total_generated": self._stats["total_generated"],
            }, commit=False)
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")
