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
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto

logger = logging.getLogger(__name__)


# ======================================================================
# AGENT SYSTEM PROMPTS
# ======================================================================

QUERY_PROPOSER_SYSTEM_PROMPT = """\
You are a medical information retrieval expert. Given a training question, your job is to \
propose 10 diverse search queries that will retrieve relevant medical knowledge from a \
large multimodal medical database.

The database contains: PubMedQA abstracts, MIRAGE MCQs, medical textbooks (MedRAG), \
PubMed articles, Wikipedia medical articles, PMC-VQA radiology images with questions, \
and CLIMB clinical QA (chest X-ray, dermoscopy, CT, ECG, fundus, MRI, mammography, \
ultrasound, pathology).

REQUIREMENTS:
- Output EXACTLY 10 search queries.
- Each query must be a COMPLETE sentence, not just keywords.
- Queries must cover DIFFERENT angles: pathophysiology, diagnosis, treatment, prognosis, \
epidemiology, imaging findings, clinical presentation, differential diagnosis, \
complications, related conditions.
- Each query should be specific enough to retrieve focused results (not generic).
- If the target question involves a medical image/modality, include at least one query \
about imaging findings / visual patterns.

Output ONLY a JSON array of 10 strings. No markdown, no explanation.
["query 1", "query 2", ..., "query 10"]"""


QUESTION_GENERATOR_SYSTEM_PROMPT = """\
You are a medical educator creating training questions for a medical AI. You have been \
given: (1) a reference training question, (2) a relevant passage from a medical \
knowledge base, (3) the solver model's recent accuracy.

Your job: SYNTHESIZE a NEW training question that combines the reference topic with \
insights from the retrieved knowledge. The new question MUST NOT be a direct copy or \
paraphrase of either source. Instead, create something ORIGINAL:

- A complex clinical scenario drawing on multiple aspects of the retrieved knowledge.
- A rare corner case that tests edge-case reasoning.
- A differential diagnosis challenge where multiple diagnoses fit partially.
- A treatment decision weighing tradeoffs mentioned in the knowledge.
- An unusual presentation or atypical finding from the retrieved passage.

FORMAT: Randomly choose one of these two formats:
(A) **Multiple choice**: question + 4 options (A/B/C/D) + correct letter.
(B) **Free response**: question + short expected answer (one phrase or sentence).

DIFFICULTY CALIBRATION:
- Solver's recent accuracy: {accuracy:.0%} over {accuracy_count} questions.
- Target ~50% accuracy. If current is high, create harder questions (more nuance, \
subtler distinctions). If low, create clearer questions.

ANSWER RULES:
- The answer must be verifiable from medical knowledge (standard textbook facts).
- Do not require external private data.
- For MCQ: distractors should be plausible, not silly.
- For free response: answer should be a short specific phrase (1-15 words).

Output ONLY a JSON object. No markdown, no explanation.

For multiple choice:
{{"format": "mcq", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}

For free response:
{{"format": "free", "question": "...", "answer": "short expected answer"}}"""


QUESTION_VALIDATOR_SYSTEM_PROMPT = """\
You are a medical fact-checker. You are given:
(1) A candidate training question + its proposed answer.
(2) Retrieved passages from a medical knowledge database that relate to the question.

Your job: Decide whether the question's answer CONTRADICTS the retrieved database \
knowledge.

DECISION RULES (be LENIENT — only reject obvious contradictions):
- If the retrieved knowledge directly and unambiguously STATES something that makes \
the proposed answer WRONG → "contradict"
- If the retrieved knowledge is silent, tangential, or only partially relevant → "ok"
- If the question is about something NOT in the retrieved passages (out-of-knowledge) \
→ "ok" (we accept new knowledge)
- If the question is well-formed but the answer seems questionable without direct \
contradiction from the passages → "ok"
- If the question is poorly formed / ungrammatical / incoherent → "contradict"

Also give a one-sentence reason.

Output ONLY a JSON object. No markdown, no explanation.
{{"verdict": "ok" or "contradict", "reason": "..."}}"""


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


class SelfEvolvingDataset(Dataset):
    """Multi-agent RAG-based self-evolving dataset.

    Dynamic mode pipeline (per target question):
      1. QueryProposer generates 10 search queries.
      2. For each query, retrieve top-K relevant entries from Milvus.
      3. For each retrieved entry, QuestionGenerator synthesizes 1 new question.
      4. QuestionValidator checks each new question against the DB (re-retrieve).
      5. Accepted questions are added to the pool. Rejected are logged.

    Static mode: serves JSONL directly (for validation).
    """

    # Trainer creates train dataset first, then val dataset. We use this counter
    # to distinguish them even when train_files == val_files (unsupervised setup
    # where we train on the test set with labels/context masked).
    _instance_counter = 0

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.config = config

        if isinstance(data_files, str):
            data_files_list = [data_files]
        else:
            data_files_list = list(data_files)

        instance_idx = SelfEvolvingDataset._instance_counter
        SelfEvolvingDataset._instance_counter += 1
        self.is_static = instance_idx > 0

        all_entries = self._load_entries(data_files_list)
        if max_samples > 0:
            all_entries = all_entries[:max_samples]

        if self.is_static:
            print(f"SelfEvolvingDataset: static mode with {len(all_entries)} items (validation)")
            self.current_dataset = all_entries
            return

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
        self.milvus_top_k = se_config.get("milvus_top_k", 3)

        # Embedding API config (same vLLM instance, but pooling runner)
        self.embed_api_base = se_config.get("embed_api_base", self.api_base)
        self.embed_model = se_config.get("embed_model", "Qwen/Qwen3-VL-Embedding-2B")

        # Pipeline params
        self.n_queries = se_config.get("n_queries", 10)
        self.questions_per_query = se_config.get("questions_per_query", 1)
        self.no_label = se_config.get("no_label", False)

        self.target_questions = all_entries
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

        # Pre-generate a seed batch
        self._ensure_questions_available(32)

    # --------------------------------------------------------------
    # Loading & dataset protocol
    # --------------------------------------------------------------
    def _load_entries(self, data_files: list[str]) -> list[dict]:
        entries = []
        for path in data_files:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def __len__(self) -> int:
        if self.is_static:
            return len(self.current_dataset)
        return self.dataset_length

    def __getitem__(self, item: int) -> dict:
        if self.is_static:
            entry = self.current_dataset[item]
        else:
            self._ensure_questions_available(item + 1)
            entry = self._generated_questions[item]

        return {
            "data_source": entry.get("data_source", "self_evolving"),
            "prompt": entry["prompt"],
            "raw_prompt": entry["prompt"],
            "reward_model": entry["reward_model"],
            "extra_info": entry.get("extra_info", {}),
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": entry.get("extra_info", {}).get("index", item),
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size: int = 14,
        config: DictConfig = None,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        return None, None

    def on_batch_end(self, batch: DataProto) -> None:
        if self.is_static:
            return
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
                                  accuracy_stats: dict) -> dict:
        sys_prompt = QUESTION_GENERATOR_SYSTEM_PROMPT.format(
            accuracy=accuracy_stats["mean"],
            accuracy_count=accuracy_stats["count"],
        )
        user_prompt = (
            f"Reference training question:\n{target_question}\n\n"
            f"Retrieved medical knowledge:\n{knowledge_passage[:1500]}\n\n"
            "Synthesize one new training question as specified."
        )
        response = self._api_call(sys_prompt, user_prompt, max_tokens=1024, temperature=0.9)
        q = self._parse_json_response(response, expect_array=False)

        fmt = q.get("format", "").lower()
        question = q.get("question", "").strip()
        answer = str(q.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"Missing question/answer: {q}")
        if fmt not in ("mcq", "free"):
            fmt = "mcq" if "options" in q else "free"
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
            f"[{h['source']}] {h['text'][:400]}" for h in hits[:3]
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
            if verdict == "contradict":
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

    def _build_entry(self, generated: dict, target: dict) -> dict:
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

        # Agents 2 & 3: for each retrieved passage, generate + validate
        new_entries = []
        rejected = []
        for q_idx, query in enumerate(queries):
            hits = self._milvus_search(query, top_k=1)
            if not hits:
                continue
            passage = hits[0]["text"]
            for _ in range(self.questions_per_query):
                try:
                    gen = self._agent_question_generator(target_question, passage, stats)
                except Exception as e:
                    logger.warning(f"Generator failed on query {q_idx}: {e}")
                    continue
                self._stats["total_generated"] += 1
                try:
                    ok, reason = self._agent_validator(gen)
                except Exception as e:
                    ok, reason = True, f"validator error: {e}"
                if ok:
                    entry = self._build_entry(gen, target)
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
                       "format", "question", "answer", "options"]
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
