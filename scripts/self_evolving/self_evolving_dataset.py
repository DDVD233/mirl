"""
Self-evolving dataset that dynamically generates training questions via a proposer API.

The proposer model (same spec as the solver) generates related questions calibrated
to the solver's current accuracy level. The dataset tracks per-question accuracy
and feeds this information back to the proposer for adaptive difficulty.
"""

import json
import logging
import re
from collections import deque
from typing import Optional

import requests
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto

logger = logging.getLogger(__name__)

PROPOSER_SYSTEM_PROMPT = """\
You are a biomedical question proposer. Your task is to generate PubMedQA-style questions \
for training a biomedical QA model.

Each question must include:
1. A realistic biomedical context paragraph (like a PubMed abstract excerpt)
2. A yes/no/maybe question about the context
3. The correct answer (yes, no, or maybe)

The solver model's recent accuracy is {accuracy:.1%}. Target ~50% difficulty - if accuracy \
is high, make questions harder; if low, make them easier.

Generate exactly {n_questions} questions based on this reference question and topic area.
The questions should be RELATED to the reference topic but NOT identical.

Output ONLY a JSON array with objects having keys: "context", "question", "answer"
Example:
[
  {{"context": "A study examined...", "question": "Does X cause Y?", "answer": "yes"}},
  ...
]"""

SOLVER_SYSTEM_PROMPT = (
    "You are a biomedical expert. Read the provided context from a PubMed abstract "
    "and answer the question. You FIRST think about the reasoning process as an "
    "internal monologue and then provide the final answer. The reasoning process "
    "MUST BE enclosed within <think> </think> tags. The final answer MUST BE one of: "
    "yes, no, or maybe, wrapped in \\boxed{}.\n\n"
    "Example format:\n"
    "<think>\n[Your reasoning here]\n</think>\n\\boxed{yes}"
)


class SelfEvolvingDataset(Dataset):
    """Dataset that generates training questions dynamically via a proposer API.

    Loads PubMedQA seed questions as targets, then uses an external LLM API
    to generate related questions calibrated to the solver's current level.
    Tracks accuracy history and feeds it back to the proposer.
    """

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

        # Load self-evolving config
        se_config = config.self_evolving
        self.api_base = se_config.api_base
        self.api_key = se_config.get("api_key", "EMPTY")
        self.model_name = se_config.model_name
        self.questions_per_target = se_config.get("questions_per_target", 5)
        self.target_accuracy = se_config.get("target_accuracy", 0.5)
        self.accuracy_window = se_config.get("accuracy_window", 32)
        self.bank_refill_size = se_config.get("bank_refill_size", 10)
        self.min_bank_size = se_config.get("min_bank_size", 16)

        # Load seed target questions from JSONL
        self.target_questions = self._load_targets(data_files)
        if max_samples > 0:
            self.target_questions = self.target_questions[:max_samples]
        logger.info(f"Loaded {len(self.target_questions)} target questions")

        # State
        self.target_idx = 0
        self.question_bank: list[dict] = []
        self.accuracy_history: deque = deque(maxlen=self.accuracy_window)
        self._question_counter = 0  # global counter for unique indices

        # Seed the initial question bank
        self._refill_bank()
        # Snapshot for stable iteration
        self.current_dataset: list[dict] = list(self.question_bank)

    def _load_targets(self, data_files: str | list[str]) -> list[dict]:
        """Load seed questions from JSONL file(s)."""
        if isinstance(data_files, str):
            data_files = [data_files]
        targets = []
        for path in data_files:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        targets.append(json.loads(line))
        return targets

    def __len__(self) -> int:
        return len(self.current_dataset)

    def __getitem__(self, item: int) -> dict:
        """Return a single question in verl-compatible format."""
        entry = self.current_dataset[item]
        row_dict = {
            "data_source": entry.get("data_source", "pubmedqa"),
            "prompt": entry["prompt"],
            "raw_prompt": entry["prompt"],  # messages list
            "reward_model": entry["reward_model"],
            "extra_info": entry.get("extra_info", {}),
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": entry.get("extra_info", {}).get("index", item),
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }
        return row_dict

    def on_batch_end(self, batch: DataProto) -> None:
        """Called after each training batch to update accuracy tracking.

        Extracts per-item accuracy from reward extra info and refills
        the question bank if it's running low.
        """
        # Extract accuracy from batch
        if "acc" in batch.non_tensor_batch:
            accs = batch.non_tensor_batch["acc"]
            for acc in accs:
                self.accuracy_history.append(float(acc))
        elif "rm_scores" in batch.batch:
            # Fallback: use reward scores > threshold as proxy for accuracy
            scores = batch.batch["rm_scores"].sum(dim=-1).tolist()
            for s in scores:
                self.accuracy_history.append(1.0 if s > 0.5 else 0.0)

        # Refill bank if running low and update snapshot
        if len(self.question_bank) < self.min_bank_size:
            self._refill_bank()
            self.current_dataset = list(self.question_bank)

    def _get_accuracy_stats(self) -> dict:
        """Compute accuracy statistics from recent history."""
        if not self.accuracy_history:
            return {"mean": 0.5, "count": 0}
        acc_list = list(self.accuracy_history)
        return {
            "mean": sum(acc_list) / len(acc_list),
            "count": len(acc_list),
        }

    def _refill_bank(self) -> None:
        """Generate new questions by calling the proposer API for multiple targets."""
        stats = self._get_accuracy_stats()
        n_targets = min(self.bank_refill_size, len(self.target_questions))

        logger.info(
            f"Refilling bank: generating {n_targets * self.questions_per_target} questions "
            f"(accuracy: {stats['mean']:.1%}, history: {stats['count']})"
        )

        for _ in range(n_targets):
            target = self.target_questions[self.target_idx % len(self.target_questions)]
            self.target_idx += 1

            generated = self._call_proposer(target, stats)
            self.question_bank.extend(generated)

        logger.info(f"Bank now has {len(self.question_bank)} questions")

    def _call_proposer(self, target: dict, accuracy_stats: dict) -> list[dict]:
        """Call proposer API to generate questions related to a target.

        Returns list of dicts in verl annotation format.
        """
        # Build the reference info from the target
        target_question = target.get("extra_info", {}).get("question", "")
        target_answer = target.get("reward_model", {}).get("ground_truth", "")

        # Extract context from the target prompt
        target_context = ""
        for msg in target.get("prompt", []):
            if msg.get("role") == "user":
                target_context = msg["content"]
                break

        system_prompt = PROPOSER_SYSTEM_PROMPT.format(
            accuracy=accuracy_stats["mean"],
            n_questions=self.questions_per_target,
        )

        user_prompt = (
            f"Reference question: {target_question}\n"
            f"Reference answer: {target_answer}\n"
            f"Reference context:\n{target_context}\n\n"
            f"Generate {self.questions_per_target} new questions."
        )

        try:
            response = self._api_call(system_prompt, user_prompt)
            questions = self._parse_proposer_response(response, target)
            return questions
        except Exception as e:
            logger.warning(f"Proposer API call failed: {e}. Using target as fallback.")
            # Fallback: return the target question itself
            return [self._make_entry_from_target(target)]

    def _api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make a synchronous call to the OpenAI-compatible API."""
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
            "max_tokens": 4096,
            "temperature": 0.8,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _parse_proposer_response(self, response: str, target: dict) -> list[dict]:
        """Parse the proposer's JSON response into verl annotation format."""
        # Try to extract JSON array from the response
        # The model may wrap it in markdown code blocks
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            logger.warning(f"Could not parse JSON from proposer response: {response[:200]}")
            return [self._make_entry_from_target(target)]

        try:
            questions = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from proposer: {json_match.group()[:200]}")
            return [self._make_entry_from_target(target)]

        entries = []
        target_pubmed_id = target.get("extra_info", {}).get("pubmed_id", "unknown")

        for q in questions:
            if not isinstance(q, dict) or "question" not in q or "answer" not in q:
                continue

            context = q.get("context", "No context provided.")
            question = q["question"]
            answer = q["answer"].strip().lower()

            # Validate answer
            if answer not in ("yes", "no", "maybe"):
                continue

            user_content = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Based on the context above, answer yes, no, or maybe."
            )

            self._question_counter += 1
            entry = {
                "data_source": "pubmedqa",
                "prompt": [
                    {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "index": self._question_counter,
                    "split": "train",
                    "pubmed_id": f"generated_from_{target_pubmed_id}",
                    "question": question,
                    "source": "proposer",
                },
            }
            entries.append(entry)

        if not entries:
            return [self._make_entry_from_target(target)]
        return entries

    def _make_entry_from_target(self, target: dict) -> dict:
        """Create an entry directly from a target question (fallback)."""
        self._question_counter += 1
        entry = dict(target)
        entry["extra_info"] = dict(target.get("extra_info", {}))
        entry["extra_info"]["index"] = self._question_counter
        entry["extra_info"]["source"] = "target_fallback"
        return entry
