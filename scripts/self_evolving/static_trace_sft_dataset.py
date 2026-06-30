"""
Static distillation-trace SFT dataset.

The on-disk counterpart of `SelfEvolvingSFTDataset`: it reads a static JSONL of
distillation annotations (produced by `make_distill_traces.py`) where every row
carries a verified `reference_response` — the teacher's
`<think>…</think>\\boxed{answer}` trace — and assembles the supervised example

    [ system, user(question), assistant(reference_response) ]

reusing `MultiTurnSFTDataset`'s tokenization / loss-mask (over the assistant turn
only) / multimodal / padding machinery, exactly as `SelfEvolvingSFTDataset` does.
The only difference is the data SOURCE: a static in-memory list of JSONL rows
instead of a fetch-on-demand gen-server shim.

Use this for the first (SFT) stage of the two-stage recipe; the second (RL) stage
trains on the same `train.jsonl` via `SelfEvolvingDataset` / the GRPO path.
"""

import json
import logging
import os
from typing import Optional

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
    def __init__(self, outer: "StaticTraceSFTDataset"):
        self._outer = outer

    def __getitem__(self, item: int) -> _Row:
        return _Row(self._outer._get_row(item))


class _Frame:
    """Stands in for the pandas dataframe; `.iloc[item].to_dict()` builds a row."""

    def __init__(self, outer: "StaticTraceSFTDataset"):
        self.iloc = _Iloc(outer)


class StaticTraceSFTDataset(MultiTurnSFTDataset):
    """Supervised dataset over a static distillation-trace JSONL."""

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
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
        # OFF: the <think>…</think> trace is provided literally in the assistant
        # content, so the chat template must not inject its own thinking scaffold
        # (Qwen3.6's template leaves a content-provided think block untouched).
        self.enable_thinking_default = config.get("enable_thinking_default", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", True)
        assert self.truncation in ["error", "left", "right"]

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        self.tools = None
        self.enable_thinking = None

        self.prompt_key = config.get("prompt_key", "prompt")
        self.reference_key = config.get("reference_key", "reference_response")

        files = data_files if isinstance(data_files, (list, ListConfig)) else [data_files]
        pf0 = files[0] if files else ""
        self.data_source_dir = os.path.dirname(os.path.abspath(pf0)) if pf0 else ""

        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

        # CLIMB media (None for non-CLIMB datasets like mimiciv_rare).
        from verl.utils.climb import ClimbMediaConfig

        self._climb_media_cfg = ClimbMediaConfig.from_data_config(config)

        # --- load the static rows -------------------------------------------
        self._raw: list[dict] = []
        for path in files:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._raw.append(json.loads(line))
        if max_samples is not None and max_samples > 0:
            self._raw = self._raw[:max_samples]

        self._rows: dict[int, dict] = {}
        self.dataframe = _Frame(self)

        n_img = sum(1 for r in self._raw if r.get(self.image_key))
        print(
            f"StaticTraceSFTDataset: {len(self._raw)} rows from {files} "
            f"({n_img} with images, max_length={self.max_length})"
        )

    def __len__(self) -> int:
        return len(self._raw)

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------
    def _get_row(self, item: int) -> dict:
        if item not in self._rows:
            self._rows[item] = self._to_row(self._raw[item])
        return self._rows[item]

    def _to_row(self, entry: dict) -> dict:
        """Convert a distillation-annotation row into a MultiTurnSFTDataset row."""
        ref = entry.get(self.reference_key)
        if not ref:
            raise RuntimeError(
                f"row is missing '{self.reference_key}'; build the distillation "
                "annotations with scripts/self_evolving/make_distill_traces.py first."
            )
        messages = [dict(m) for m in entry.get(self.prompt_key, [])]
        messages.append({"role": "assistant", "content": ref})
        row = {self.messages_key: messages, "extra_info": entry.get("extra_info", {})}

        images = entry.get(self.image_key) or []
        videos = entry.get(self.video_key) or []
        if self._climb_media_cfg is not None and (images or videos):
            from verl.utils.climb import resolve_and_flatten_media

            images, videos = resolve_and_flatten_media(messages, images, videos, self._climb_media_cfg)
        if images:
            row[self.image_key] = images
        return row

    # ------------------------------------------------------------------
    # Multimodal homogeneity + vision hooks (identical to SelfEvolvingSFTDataset)
    # ------------------------------------------------------------------
    def __getitem__(self, item):
        res = super().__getitem__(item)
        mmi = res.get("multi_modal_inputs")
        if not isinstance(mmi, dict):
            mmi = {}
        mmi.pop("mm_token_type_ids", None)
        res["multi_modal_inputs"] = mmi
        return res

    def _has_vision_content(self, messages):
        """Force the whole-conversation (multimodal) tokenization path even for
        text-only rows when a processor is available — the per-turn text path
        trips Qwen3.x's chat template. With no images the processor just returns
        text input_ids."""
        if self.processor is not None:
            return True
        return super()._has_vision_content(messages)

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        from verl.utils.dataset.rl_dataset import RLHFDataset

        return await RLHFDataset.process_vision_info(
            messages, image_patch_size=image_patch_size, config=config
        )
