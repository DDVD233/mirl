"""
Self-evolving dataset that dynamically generates training questions via a proposer API.

Design:
- Each target question (from PubMedQA) generates exactly 5 training questions via the proposer.
- Questions are served sequentially: targets[0] -> 5 questions, targets[1] -> 5 questions, etc.
- After all targets are exhausted, cycle back to targets[0] with new questions.
- Previously proposed questions are included in the proposer prompt to avoid repetition.
- Each question is trained only once (no reuse across epochs).
- __len__ returns a large number so the dataloader doesn't limit training.
"""

import json
import logging
import os
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

PROPOSER_SYSTEM_PROMPT = """\
You are a biomedical research expert creating training questions for a medical QA system.

Your task: Given a reference PubMed question and its answer, generate {n_questions} NEW \
biomedical questions that test similar medical knowledge.

REQUIREMENTS for each question:
1. CONTEXT: Write a realistic biomedical context paragraph (2-4 sentences) that resembles \
a PubMed abstract. It must contain specific factual medical information — real conditions, \
treatments, mechanisms, or study findings. Do NOT fabricate statistics or study results \
that contradict established medical knowledge.
2. QUESTION: A yes/no/maybe question that can be answered from the context alone. \
The question should test understanding of the medical concepts in the context.
3. ANSWER: The correct answer (yes, no, or maybe) that is unambiguously supported by the context. \
For "maybe" answers, the context should present mixed or inconclusive evidence.

DIFFICULTY CALIBRATION:
- The solver model's recent accuracy is {accuracy:.0%} over {accuracy_count} questions.
- Target ~50% accuracy. If current accuracy is high, use more nuanced contexts requiring \
deeper reasoning. If low, use more straightforward evidence.

DIVERSITY: Vary the answer distribution (not all "yes"). Cover different aspects of \
the reference topic (mechanisms, treatments, diagnosis, prognosis, epidemiology).

{history_section}

Output ONLY a JSON array. No markdown, no explanation.
[
  {{"context": "...", "question": "...", "answer": "yes/no/maybe"}},
  ...
]"""

PROPOSER_NO_LABEL_SYSTEM_PROMPT = """\
You are a biomedical research expert creating training questions for a medical QA system.

Your task: Given a reference PubMed question (without its answer), generate {n_questions} NEW \
biomedical questions that test similar medical knowledge.

REQUIREMENTS for each question:
1. CONTEXT: Write a realistic biomedical context paragraph (2-4 sentences) that resembles \
a PubMed abstract. It must contain specific factual medical information — real conditions, \
treatments, mechanisms, or study findings. Do NOT fabricate statistics or study results \
that contradict established medical knowledge.
2. QUESTION: A yes/no/maybe question that can be answered from the context alone. \
The question should test understanding of the medical concepts in the context.

NOTE: Do NOT include an answer field. The answer is unknown and will be determined by \
an independent judge based on the context.

DIFFICULTY CALIBRATION:
- The solver model's recent accuracy is {accuracy:.0%} over {accuracy_count} questions.
- Target ~50% accuracy. If current accuracy is high, use more nuanced contexts requiring \
deeper reasoning. If low, use more straightforward evidence.

DIVERSITY: Cover different aspects of the reference topic \
(mechanisms, treatments, diagnosis, prognosis, epidemiology).

{history_section}

Output ONLY a JSON array. No markdown, no explanation.
[
  {{"context": "...", "question": "..."}},
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
    """Dataset that generates training questions on-the-fly via a proposer API.

    Flow:
    - Iterates through target questions sequentially.
    - For each target, calls the proposer to generate `questions_per_target` questions.
    - Serves those questions one at a time via __getitem__.
    - When the current target's questions are exhausted, moves to the next target.
    - After cycling through all targets, starts over with fresh questions (no repeats).
    - Previously proposed questions for each target are tracked to avoid repetition.
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

        # Detect if this is a val dataset (val_files != train_files).
        if isinstance(data_files, str):
            data_files_list = [data_files]
        else:
            data_files_list = list(data_files)

        train_files = config.get("train_files", "")
        if isinstance(train_files, str):
            train_files = [train_files]
        self.is_static = any(f not in train_files for f in data_files_list)

        # Load data from JSONL
        all_entries = self._load_entries(data_files_list)
        if max_samples > 0:
            all_entries = all_entries[:max_samples]

        if self.is_static:
            # Static mode: serve JSONL directly (used for validation)
            print(f"SelfEvolvingDataset: static mode with {len(all_entries)} items (validation)")
            self.current_dataset = all_entries
            return

        # Dynamic mode: use proposer to generate questions
        se_config = config.self_evolving
        self.api_base = se_config.api_base
        self.api_key = se_config.get("api_key", "EMPTY")
        self.model_name = se_config.model_name
        self.questions_per_target = se_config.get("questions_per_target", 5)
        self.accuracy_window = se_config.get("accuracy_window", 32)
        self.log_dir = se_config.get("log_dir", "/scratch/self_evolving_datasets/logs")
        self.dataset_length = se_config.get("dataset_length", 100000)
        self.no_label = se_config.get("no_label", False)

        self.target_questions = all_entries
        print(f"SelfEvolvingDataset: dynamic mode with {len(self.target_questions)} seed targets")

        # Set up question log file
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.question_log_path = os.path.join(self.log_dir, f"proposed_questions_{timestamp}.jsonl")
        print(f"Proposer question log: {self.question_log_path}")

        # State
        self.target_idx = 0  # which target we're currently generating from
        self.cycle = 0  # how many times we've cycled through all targets
        self._question_counter = 0
        self._wandb_question_table = None

        # Queue of ready-to-serve questions (generated from current target)
        self._question_queue: list[dict] = []
        # Per-target history of previously proposed questions (to avoid repeats)
        # Key: target index, Value: list of {"question": ..., "answer": ...}
        self._proposed_history: dict[int, list[dict]] = defaultdict(list)
        # Rolling accuracy from recent training
        self._accuracy_history: list[float] = []

        # Pre-generate first batch of questions
        self._generate_next_batch()

    def _load_entries(self, data_files: list[str]) -> list[dict]:
        """Load entries from JSONL file(s)."""
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
        """Return a single question in verl-compatible format."""
        if self.is_static:
            entry = self.current_dataset[item]
        else:
            # Generate more questions if queue is empty
            if not self._question_queue:
                self._generate_next_batch()
            entry = self._question_queue.pop(0)

        return {
            "data_source": entry.get("data_source", "pubmedqa"),
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
        """No vision data for text-only PubMedQA questions."""
        return None, None

    def on_batch_end(self, batch: DataProto) -> None:
        """Update accuracy tracking from training batch results."""
        if self.is_static:
            return

        if "acc" in batch.non_tensor_batch:
            for acc in batch.non_tensor_batch["acc"]:
                self._accuracy_history.append(float(acc))
            # Keep only last N
            if len(self._accuracy_history) > self.accuracy_window:
                self._accuracy_history = self._accuracy_history[-self.accuracy_window:]
        elif "rm_scores" in batch.batch:
            scores = batch.batch["rm_scores"].sum(dim=-1).tolist()
            for s in scores:
                self._accuracy_history.append(1.0 if s > 0.5 else 0.0)
            if len(self._accuracy_history) > self.accuracy_window:
                self._accuracy_history = self._accuracy_history[-self.accuracy_window:]

    def _get_accuracy_stats(self) -> dict:
        """Compute accuracy statistics from recent history."""
        if not self._accuracy_history:
            return {"mean": 0.5, "count": 0}
        return {
            "mean": sum(self._accuracy_history) / len(self._accuracy_history),
            "count": len(self._accuracy_history),
        }

    def _generate_next_batch(self) -> None:
        """Generate questions for the current target and advance to the next."""
        target = self.target_questions[self.target_idx]
        stats = self._get_accuracy_stats()
        history = self._proposed_history[self.target_idx]

        max_retries = 3
        generated = None
        for attempt in range(1, max_retries + 1):
            try:
                generated = self._call_proposer(target, stats, history)
                break
            except Exception as e:
                logger.warning(
                    f"Proposer attempt {attempt}/{max_retries} failed for target "
                    f"{self.target_idx}: {e}"
                )
                if attempt == max_retries:
                    logger.warning(
                        f"All {max_retries} proposer attempts failed. "
                        f"Reusing random questions from bank."
                    )

        if generated is None:
            # Return random existing questions from the bank if available,
            # otherwise skip this target
            if self._question_queue:
                import random as _rand
                n = min(self.questions_per_target, len(self._question_queue))
                generated = _rand.sample(self._question_queue, n)
            else:
                # Nothing in bank either — advance and hope next target works
                logger.warning("No questions in bank and proposer failed. Skipping target.")
                generated = []

        # Record proposed questions in history for this target
        for entry in generated:
            q = entry.get("extra_info", {}).get("question", "")
            a = entry.get("reward_model", {}).get("ground_truth", "")
            self._proposed_history[self.target_idx].append({"question": q, "answer": a})

        self._question_queue.extend(generated)
        self._log_questions(generated, target)

        # Advance to next target
        self.target_idx += 1
        if self.target_idx >= len(self.target_questions):
            self.target_idx = 0
            self.cycle += 1
            print(f"SelfEvolvingDataset: completed cycle {self.cycle}, starting over")

    def _call_proposer(self, target: dict, accuracy_stats: dict,
                       history: list[dict]) -> list[dict]:
        """Call proposer API to generate questions related to a target."""
        target_question = target.get("extra_info", {}).get("question", "")
        target_answer = target.get("reward_model", {}).get("ground_truth", "")

        # Extract context from the target prompt
        target_context = ""
        for msg in target.get("prompt", []):
            if msg.get("role") == "user":
                target_context = msg["content"]
                break

        # Build history section for prompt
        if history:
            history_lines = []
            for i, h in enumerate(history[-10:], 1):  # last 10 to keep prompt reasonable
                if self.no_label:
                    history_lines.append(f"  {i}. Q: {h['question']}")
                else:
                    history_lines.append(f"  {i}. Q: {h['question']} A: {h['answer']}")
            history_text = "\n".join(history_lines)
            history_section = (
                f"PREVIOUSLY PROPOSED (do NOT repeat these — generate entirely new questions):\n"
                f"{history_text}"
            )
        else:
            history_section = ""

        if self.no_label:
            system_prompt = PROPOSER_NO_LABEL_SYSTEM_PROMPT.format(
                n_questions=self.questions_per_target,
                accuracy=accuracy_stats["mean"],
                accuracy_count=accuracy_stats["count"],
                history_section=history_section,
            )
            user_prompt = (
                f"Reference PubMed question: {target_question}\n"
                f"Reference abstract context:\n{target_context}\n\n"
                f"Generate {self.questions_per_target} new training questions (no answers needed)."
            )
        else:
            system_prompt = PROPOSER_SYSTEM_PROMPT.format(
                n_questions=self.questions_per_target,
                accuracy=accuracy_stats["mean"],
                accuracy_count=accuracy_stats["count"],
                history_section=history_section,
            )
            user_prompt = (
                f"Reference PubMed question: {target_question}\n"
                f"Reference answer: {target_answer}\n"
                f"Reference abstract context:\n{target_context}\n\n"
                f"Generate {self.questions_per_target} new training questions."
            )

        response = self._api_call(system_prompt, user_prompt)
        questions = self._parse_proposer_response(response, target)
        return questions

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
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not parse JSON array from proposer response: {response[:500]}")

        questions = json.loads(json_match.group())

        entries = []
        target_pubmed_id = target.get("extra_info", {}).get("pubmed_id", "unknown")

        for q in questions:
            if not isinstance(q, dict) or "question" not in q:
                continue

            context = q.get("context", "No context provided.")
            question = q["question"]

            if self.no_label:
                answer = ""  # no ground truth — reward judge will determine correctness
            else:
                if "answer" not in q:
                    continue
                answer = q["answer"].strip().lower()

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
                    "answer": answer,
                    "context": context,
                    "source": "proposer",
                    "cycle": self.cycle,
                    "target_idx": self.target_idx,
                },
            }
            entries.append(entry)

        if not entries:
            raise ValueError(
                f"Proposer returned no valid questions for target: "
                f"{target.get('extra_info', {}).get('question', '?')[:200]}. "
                f"Raw response: {response[:500]}"
            )
        return entries

    def _log_questions(self, entries: list[dict], target: dict) -> None:
        """Append proposed questions and their answers to the log file and wandb."""
        target_question = target.get("extra_info", {}).get("question", "")
        target_answer = target.get("reward_model", {}).get("ground_truth", "")
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
                    "target_answer": target_answer,
                    "proposed_question": ei.get("question", ""),
                    "proposed_answer": ei.get("answer", ""),
                    "proposed_context": ei.get("context", ""),
                    "source": ei.get("source", ""),
                    "index": ei.get("index", -1),
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)

        self._log_questions_to_wandb(records)

    def _log_questions_to_wandb(self, records: list[dict]) -> None:
        """Log proposed questions to a wandb table."""
        try:
            import wandb

            if wandb.run is None:
                return

            columns = [
                "cycle", "target_idx", "solver_accuracy",
                "target_question", "target_answer",
                "proposed_question", "proposed_answer", "proposed_context", "source",
            ]
            if self._wandb_question_table is None:
                self._wandb_question_table = wandb.Table(columns=columns)

            new_table = wandb.Table(columns=columns, data=self._wandb_question_table.data)
            for r in records:
                new_table.add_data(
                    r["cycle"],
                    r["target_idx"],
                    r["solver_accuracy"],
                    r["target_question"],
                    r["target_answer"],
                    r["proposed_question"],
                    r["proposed_answer"],
                    r["proposed_context"][:500],
                    r["source"],
                )
            self._wandb_question_table = new_table
            wandb.log({"proposer/questions": new_table}, commit=False)
        except Exception as e:
            logger.warning(f"Failed to log questions to wandb: {e}")

