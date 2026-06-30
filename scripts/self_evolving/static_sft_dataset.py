"""
Static SFT dataset — the no-self-evolving counterpart of
`self_evolving_sft_dataset.py`.

Reads the REAL training JSONL directly (the same RLHF-format file the GRPO
baseline trains on) and builds supervised examples of the form

    [ system, user(question), assistant("\\boxed{ground_truth}") ]

i.e. plain *train input -> train output*. The assistant target is just the
dataset's own ground-truth answer wrapped in the \\boxed{...} format the task
prompt asks for (`reward_model.ground_truth`); there is NO teacher trace, NO
generation server, and NO reward evolution. This is the SFT control for the
GRPO / self-evolving runs.

We subclass `MultiTurnSFTDataset` purely to reuse its tokenization / loss-mask
(over the assistant turn only) / multimodal / padding machinery, exactly as
`SelfEvolvingSFTDataset` does — the only difference is the data SOURCE: a static
in-memory list of JSONL rows instead of a fetch-on-demand gen-server shim.

Images are passed through verbatim; unreadable paths degrade to a black
placeholder inside `vision_utils.process_image`, identical to the RLHFDataset
path the GRPO baseline used (so the two runs see the same vision input).
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
    def __init__(self, outer: "StaticSFTDataset"):
        self._outer = outer

    def __getitem__(self, item: int) -> _Row:
        return _Row(self._outer._get_row(item))


class _Frame:
    """Stands in for the pandas dataframe; `.iloc[item].to_dict()` builds a row."""

    def __init__(self, outer: "StaticSFTDataset"):
        self.iloc = _Iloc(outer)


class StaticSFTDataset(MultiTurnSFTDataset):
    """Supervised dataset over a static RLHF-format JSONL (prompt + ground_truth)."""

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # `data_files` matches RLHFDataset's signature so `create_rl_dataset` can
        # construct us. We read the rows from it directly.
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
        # We supply only the boxed answer as the assistant content (no <think>
        # trace), so the chat template must not inject its own thinking scaffold.
        self.enable_thinking_default = config.get("enable_thinking_default", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", True)
        assert self.truncation in ["error", "left", "right"]

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        # Tools / enable_thinking are per-row in the parent; we have none.
        self.tools = None
        self.enable_thinking = None

        # The answer-format wrapper. ground_truth is already "CODE: Name", and the
        # task prompt asks for \boxed{CODE: Name}; configurable for other datasets.
        self.answer_template = config.get("sft_answer_template", "\\boxed{{{answer}}}")
        self.ground_truth_key = config.get("ground_truth_key", "ground_truth")
        self.prompt_key = config.get("prompt_key", "prompt")

        # Base dir for resolving relative image paths in entries.
        files = data_files if isinstance(data_files, (list, ListConfig)) else [data_files]
        pf0 = files[0] if files else ""
        self.data_source_dir = os.path.dirname(os.path.abspath(pf0)) if pf0 else ""

        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)
        # MultiTurnSFTDataset locates the assistant span by matching
        # `generation_prompt` in the rendered ids. On Qwen3.6 the extracted prompt
        # ends with the thinking scaffold "<|im_start|>assistant\n<think>\n", but
        # with enable_thinking=False the template renders the *real* assistant turn
        # as "<|im_start|>assistant\n<think>\n\n</think>\n\n{content}" — the "\n" vs
        # "\n\n" after <think> means the prompt never matches and the loss mask
        # comes out all-zero (=> NaN SFT loss). Anchor at the assistant turn START
        # so the whole turn (empty-think scaffold + boxed answer) is supervised.
        # Guarded to ChatML templates; other templates keep the extracted prompt.
        _asst_start = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        if (
            _asst_start
            and self.generation_prompt
            and list(self.generation_prompt[: len(_asst_start)]) == list(_asst_start)
        ):
            self.generation_prompt = _asst_start

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
            f"StaticSFTDataset: {len(self._raw)} rows from {files} "
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
        """Convert an RLHF-format JSONL row into a MultiTurnSFTDataset row."""
        rm = entry.get("reward_model") or {}
        gt = rm.get(self.ground_truth_key)
        if gt is None or str(gt).strip() == "":
            raise RuntimeError(
                f"row is missing reward_model.{self.ground_truth_key}; cannot build "
                "an SFT target. Every training row needs a ground-truth answer."
            )
        target = self.answer_template.format(answer=str(gt).strip())

        messages = [dict(m) for m in entry.get(self.prompt_key, [])]
        messages.append({"role": "assistant", "content": target})
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
        # Keep the batch HOMOGENEOUS so a mix of text and image rows survives
        # collation: ALWAYS emit `multi_modal_inputs` ({} for text rows), and
        # drop the per-token `mm_token_type_ids` (cannot be cat'd across varlen
        # microbatches) — mirrors the RL path's extract_multi_modal_inputs.
        mmi = res.get("multi_modal_inputs")
        if not isinstance(mmi, dict):
            mmi = {}
        mmi.pop("mm_token_type_ids", None)
        res["multi_modal_inputs"] = mmi
        return res

    def _has_vision_content(self, messages):
        """Force whole-conversation (multimodal) tokenization even for text-only
        rows when a processor is available — the per-turn text path trips Qwen3.5's
        chat template. With no images the processor just returns text input_ids."""
        if self.processor is not None:
            return True
        return super()._has_vision_content(messages)

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        """Vision hook the AgentLoop calls during rollout; delegate to RLHFDataset
        (handles text-only -> empty images, and multimodal if present)."""
        from verl.utils.dataset.rl_dataset import RLHFDataset

        return await RLHFDataset.process_vision_info(
            messages, image_patch_size=image_patch_size, config=config
        )
