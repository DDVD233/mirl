"""
Self-evolving dataset client.

Fetches pre-generated training samples from the generation server (see
`scripts/self_evolving/generation_server.py`). All proposer / validator /
Milvus logic lives in the server; this class is a thin HTTP client that
inherits from RLHFDataset for tokenizer / processor / message-building.

The gen server is treated as an infinite queue of fresh samples. The
first time `__getitem__(item)` sees a given `item` index it pulls a
sample — drawn at random from a small client-side shuffle buffer over
`GET /sample` (see `_next_sample`) — and binds it to that index in an
in-memory dict; subsequent calls with the same index hit the cache.
Indices do NOT need to be consecutive — this matters on checkpoint
resume, where verl walks the dataloader past already-trained batches: we
just hand out fresh samples for those skipped indices rather than
re-fetching N samples sequentially to reach the resume point.

Note: `data.shuffle` (verl's sampler) is a no-op for ordering here — in
epoch 0 every index is fresh, so any index permutation still pulls the
same arrival stream. Batch-level mixing therefore comes from the server's
random pool draining plus this client shuffle buffer, NOT from the
sampler.

Entries arrive already shaped for RLHFDataset (prompt + reward_model +
extra_info, with `extra_info.question_id` so the reward function can
POST accuracy back).

Trainer reports accuracy via the reward function (see
`verl.utils.reward_score.self_evolving._report_to_gen_server`).
"""

import logging
import os
import random
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

        # Client-side streaming shuffle buffer (defense-in-depth). The gen
        # server already serves random pool members, so consecutive /sample
        # calls are well mixed; this buffer additionally guarantees a trainer
        # batch is a mode-mix even if pointed at an older FIFO server. Set to
        # <=1 to disable. See _next_sample for the period-2-oscillation context.
        self.shuffle_buffer_size = int(se_config.get("shuffle_buffer_size", 256))
        self._buffer: list[dict] = []

        self._wait_for_server()

        # Replace parent's seed dataframe with our index → entry cache.
        # Dict (not list) so the trainer can request arbitrary indices
        # (e.g. on resume) without us walking 0..N sequentially.
        self._fetched: dict[int, dict] = {}
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
        # Cold-start patience: gen server can take 5-10 min to fill its pool
        # when the proposer is a slow remote (e.g. Qwen3.5-397B at ~270s/call).
        # Retry hard on 503/408/429/network errors; fail fast on other 4xx.
        max_attempts = int(os.environ.get("SELF_EVOLVING_SAMPLE_RETRIES", "120"))
        for attempt in range(max_attempts):
            try:
                r = requests.get(url, timeout=self.fetch_timeout)
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                last_err = e
                status = getattr(e.response, "status_code", None)
                if status is not None and 400 <= status < 500 and status not in (408, 429, 503):
                    raise
                logger.warning(
                    f"/sample attempt {attempt + 1}/{max_attempts} failed (status={status}): {e}"
                )
            except Exception as e:
                last_err = e
                logger.warning(f"/sample attempt {attempt + 1}/{max_attempts} failed: {e}")
            time.sleep(min(2 ** min(attempt, 4), 10))
        raise RuntimeError(
            f"failed to fetch sample from {url} after {max_attempts} attempts: {last_err}"
        )

    # --------------------------------------------------------------
    # Dataset protocol
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return self.dataset_length

    def _next_sample(self) -> dict:
        """Return one sample, drawn at random from a rolling shuffle buffer.

        Even though the gen server now serves random pool members, keep a
        client-side streaming shuffle (à la tf.data.shuffle): each emitted
        sample is a uniform-random pick from a rolling window of recent
        arrivals, so a trainer batch is a mode-mix (long ``direct`` clinical
        seeds + short ``gen`` questions) regardless of the order the server
        emits. This is what prevents the bursty per-mode output from aliasing
        — under data.shuffle=False / SequentialSampler — into a period-2
        prompt-length / difficulty oscillation across steps.

        The buffer grows lazily (fetch up to two per call while warming) so the
        first batch is never blocked waiting to fill the whole buffer.
        """
        cap = self.shuffle_buffer_size
        if cap <= 1:
            return self._fetch_one()
        self._buffer.append(self._fetch_one())
        if len(self._buffer) < cap:
            # Still warming up: pull an extra so the window reaches capacity
            # within ~cap calls without a large upfront stall.
            self._buffer.append(self._fetch_one())
        i = random.randrange(len(self._buffer))
        self._buffer[i], self._buffer[-1] = self._buffer[-1], self._buffer[i]
        return self._buffer.pop()

    def __getitem__(self, item: int) -> dict:
        if item not in self._fetched:
            entry = self._next_sample()
            # `reference_response` is the SFT-only verified teacher trace. The
            # gen server's /sample now guarantees every served entry carries all
            # required fields (incl. this one in SFT mode), so it is present
            # consistently across the batch — but the RL / feedback-eval path
            # has no use for the supervised target, and RLHFDataset surfaces
            # every row key as a non-tensor batch column. Drop it here to keep
            # the feedback batch lean (a heavy unused string per row otherwise).
            entry.pop("reference_response", None)
            self._fetched[item] = entry
        return super().__getitem__(item)
