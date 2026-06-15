"""
Self-evolving SFT dataset client.

The SFT-distillation counterpart of `self_evolving_dataset.py`. Instead of
prompt-only RL samples, this fetches *full teacher traces* from the generation
server (run with `GEN_MODE=sft`) and yields tokenized supervised examples with a
loss mask over the assistant turn only.

Each gen-server entry in SFT mode carries an extra `reference_response` field —
the teacher's verified `<think>…</think>\\boxed{answer}` trace. We assemble the
conversation

    [ system, user(question), assistant(reference_response) ]

and reuse `MultiTurnSFTDataset`'s tokenization / loss-mask / padding / position-id
machinery verbatim. The only thing we override is the *data source*: rather than
reading a parquet file, `self.dataframe` is a fetch-on-demand shim that pulls the
next entry from `GET /sample` the first time a given index is requested (indices
need not be consecutive — see `self_evolving_dataset.py` for why this matters on
resume).

The gen server keeps calibrating difficulty from the trainer's eval `/report`
calls exactly as in the RL pipeline; this dataset is only the supervised-trace
side of the loop.
"""

import logging
import os
import time
from typing import Optional

import requests
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import hf_tokenizer
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

logger = logging.getLogger(__name__)


class _Row:
    """Mimics `pandas.Series` enough for `MultiTurnSFTDataset.__getitem__`."""

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    def to_dict(self) -> dict:
        return self._d


class _Iloc:
    def __init__(self, outer: "SelfEvolvingSFTDataset"):
        self._outer = outer

    def __getitem__(self, item: int) -> _Row:
        return _Row(self._outer._get_row(item))


class _FetchFrame:
    """Stands in for the pandas dataframe; `.iloc[item].to_dict()` fetches."""

    def __init__(self, outer: "SelfEvolvingSFTDataset"):
        self.iloc = _Iloc(outer)


class SelfEvolvingSFTDataset(MultiTurnSFTDataset):
    """HTTP client over the SFT-mode generation server.

    Subclasses `MultiTurnSFTDataset` to reuse its `__getitem__` tokenization
    path, but replaces the parquet-backed dataframe with a fetch-on-demand shim.
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # `data_files` (matching RLHFDataset's signature so `create_rl_dataset`
        # can construct us) is only used to locate a base dir for relative image
        # paths; the actual data comes from the gen server.
        parquet_files = data_files
        config = config or {}

        # --- config parsing (mirrors MultiTurnSFTDataset.__init__) ----------
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        self.messages_key = config.get("messages_key", "messages")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get(
            "image_patch_size", processor.image_processor.patch_size if processor else None
        )
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        # Default OFF: we provide the <think>…</think> trace literally in the
        # assistant content, so the chat template must not inject its own
        # thinking scaffolding around it.
        self.enable_thinking_default = config.get("enable_thinking_default", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        # Per-turn vs whole-conversation tokenization can differ for Qwen thinking
        # templates; we put think tags in content ourselves, so tolerate it.
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", True)
        assert self.truncation in ["error", "left", "right"]

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        # Tools / enable_thinking are per-row in the parent; we have none.
        self.tools = None
        self.enable_thinking = None

        # Base dir for resolving relative image paths in entries.
        pf0 = parquet_files[0] if isinstance(parquet_files, (list, ListConfig)) else parquet_files
        self.data_source_dir = config.self_evolving.get(
            "data_source_dir",
            os.path.dirname(os.path.abspath(pf0)) if pf0 else "",
        )
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

        # --- gen server ------------------------------------------------------
        se = config.self_evolving
        self.gen_server_url = se.gen_server_url.rstrip("/")
        self.dataset_length = int(se.get("dataset_length", 100000))
        self.fetch_timeout = float(se.get("fetch_timeout", 600))
        self.connect_wait = float(se.get("connect_wait", 600))

        self._rows: dict[int, dict] = {}
        self._wait_for_server()
        self.dataframe = _FetchFrame(self)

        print(
            f"SelfEvolvingSFTDataset: connected to {self.gen_server_url} "
            f"(dataset_length={self.dataset_length}, max_length={self.max_length})"
        )

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, item):
        res = super().__getitem__(item)
        # Keep the batch HOMOGENEOUS so a mix of text and image rows survives
        # collation. We force the multimodal tokenization path for every row
        # (see _has_vision_content), so even text rows get a `multi_modal_inputs`
        # dict — but the parent only attaches the key when non-empty, and image
        # rows get a populated dict. collate_fn builds one length-batch_size
        # object array per non-tensor key, and DataProto.check_consistency
        # rejects `multi_modal_inputs` if it is present on only the image rows
        # (this is exactly what crashed a mixed text+image SFT batch). So we
        # ALWAYS emit the key: an empty {} for text rows (extract_multi_modal_inputs
        # treats it as a pure-text sample) and the real inputs for image rows.
        # `mm_token_type_ids` is per-token / variable-length and cannot be
        # torch.cat'd across varlen microbatches, so drop it on every row
        # (mirrors verl/utils/model.py::extract_multi_modal_inputs in the RL path);
        # for text rows the forced multimodal path emits ONLY this key, so the
        # dict becomes empty after the pop.
        mmi = res.get("multi_modal_inputs")
        if not isinstance(mmi, dict):
            mmi = {}
        mmi.pop("mm_token_type_ids", None)
        res["multi_modal_inputs"] = mmi
        return res

    def _has_vision_content(self, messages):
        """Force whole-conversation tokenization (the multimodal path) even for
        text-only data when a processor is available.

        MultiTurnSFTDataset's text path tokenizes each turn *separately* and
        concatenates, which Qwen3.5's chat template rejects ("No user query
        found in messages" / "System message must be at the beginning"). The
        multimodal path applies the template to the whole [system, user,
        assistant] conversation at once and derives the loss mask from the
        assistant <|im_start|>assistant ... <|im_end|> span, which is exactly
        what we want and is template-safe. With no images the processor simply
        returns text input_ids.
        """
        if self.processor is not None:
            return True
        return super()._has_vision_content(messages)

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        """Vision extraction hook the AgentLoop calls on the configured dataset
        class during rollout generation. MultiTurnSFTDataset has no such method,
        so we delegate to RLHFDataset's implementation (handles text-only
        messages → empty images/videos, and multimodal if present)."""
        from verl.utils.dataset.rl_dataset import RLHFDataset

        return await RLHFDataset.process_vision_info(
            messages, image_patch_size=image_patch_size, config=config
        )

    # ------------------------------------------------------------------
    # HTTP (identical contract to self_evolving_dataset.SelfEvolvingDataset)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Fetch-on-demand row construction
    # ------------------------------------------------------------------
    def _get_row(self, item: int) -> dict:
        if item not in self._rows:
            self._rows[item] = self._to_row(self._fetch_one())
        return self._rows[item]

    def _to_row(self, entry: dict) -> dict:
        """Convert a gen-server SFT entry into a MultiTurnSFTDataset row."""
        ref = entry.get("reference_response")
        if not ref:
            raise RuntimeError(
                "gen-server entry is missing 'reference_response'. Start the "
                "generation server in SFT mode (GEN_MODE=sft) so it attaches "
                "verified teacher traces."
            )
        messages = [dict(m) for m in entry.get("prompt", [])]
        messages.append({"role": "assistant", "content": ref})
        row = {self.messages_key: messages, "extra_info": entry.get("extra_info", {})}
        if "images" in entry:
            row[self.image_key] = entry["images"]
        return row
