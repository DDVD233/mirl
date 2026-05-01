"""
Self-evolving dataset client.

Fetches pre-generated training samples from the generation server (see
`scripts/self_evolving/generation_server.py`). All proposer / validator /
Milvus logic lives in the server; this class is a thin HTTP client that
inherits from RLHFDataset for tokenizer / processor / message-building.

Each `__getitem__(item)` either returns a previously fetched entry at
that index or pulls the next one from `GET /sample`. Entries arrive
already shaped for RLHFDataset (prompt + reward_model + extra_info, with
`extra_info.question_id` so the reward function can POST accuracy back).

Trainer reports accuracy via the reward function (see
`verl.utils.reward_score.self_evolving._report_to_gen_server`).
"""

import logging
import time
from typing import Optional

import requests
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class SelfEvolvingDataset(RLHFDataset):
    """HTTP client over the self-evolving generation server.

    Inherits from RLHFDataset purely to reuse `__getitem__`'s
    tokenization / multimodal / message-building path. The parent's
    `data_files` is loaded only to set up the tokenizer; we then point
    `self.dataframe` at our growing list of server-fetched entries.
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        se_config = config.self_evolving
        self.gen_server_url = se_config.gen_server_url.rstrip("/")
        self.dataset_length = int(se_config.get("dataset_length", 100000))
        self.fetch_timeout = float(se_config.get("fetch_timeout", 600))
        self.connect_wait = float(se_config.get("connect_wait", 600))

        self._wait_for_server()

        # Replace parent's seed dataframe with our growing list of fetched entries.
        self._fetched: list[dict] = []
        self.dataframe = self._fetched

        print(f"SelfEvolvingDataset: connected to {self.gen_server_url} "
              f"(dataset_length={self.dataset_length})")

    # --------------------------------------------------------------
    # HTTP
    # --------------------------------------------------------------
    def _wait_for_server(self) -> None:
        url = f"{self.gen_server_url}/healthz"
        deadline = time.time() + self.connect_wait
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    return
                last_err = f"status {r.status_code}"
            except Exception as e:
                last_err = e
            time.sleep(2)
        raise RuntimeError(
            f"generation server unreachable at {self.gen_server_url} "
            f"after {self.connect_wait}s (last error: {last_err})"
        )

    def _fetch_one(self) -> dict:
        url = f"{self.gen_server_url}/sample"
        last_err = None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=self.fetch_timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                logger.warning(f"/sample attempt {attempt + 1}/3 failed: {e}")
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"failed to fetch sample from {url}: {last_err}")

    # --------------------------------------------------------------
    # Dataset protocol
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, item: int) -> dict:
        while len(self._fetched) <= item:
            self._fetched.append(self._fetch_one())
        return super().__getitem__(item)
