# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.import_utils import load_extern_object
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

logger = logging.getLogger(__name__)

# Medical-imaging JPEGs in the corpus are occasionally truncated. Let PIL decode
# the partial data instead of raising OSError("broken data stream when reading
# image file") so a single truncated file cannot crash dataset loading or the
# rollout's vision pipeline. Genuinely-unreadable files still fall back to a
# black placeholder via RLHFDataset._sanitize_image_messages.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.data_source_dir = os.path.dirname(os.path.abspath(data_files[0]))
        self.audio_key = config.get("audio_key", "audios")
        # Default to the processor's real patch_size to align with the rollout path.
        _default_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
        self.image_patch_size = config.get("image_patch_size") or _default_patch_size
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = config.get("mm_processor_kwargs", {})

        # CLIMB multimodal media: rows may reference images/videos as remote
        # "climb://<relpath>" handles that live only on the local file server
        # (the trainer node has no access to /scratch/high_modality on disk).
        # Resolved on demand in `_build_messages`. None unless data.climb.file_base
        # is configured, so this is a no-op for every non-CLIMB dataset.
        from verl.utils.climb import ClimbMediaConfig

        self._climb_media_cfg = ClimbMediaConfig.from_data_config(config)

        # Mirror AgentLoopWorker's tool loading so length filtering sees the
        # same schemas the rollout will.
        self.tool_config_path = config.get("tool_config_path", None)
        self.function_tool_path = config.get("function_tool_path", None)
        self.tool_schemas = None
        if self.tool_config_path or self.function_tool_path:
            try:
                from verl.tools.tool_registry import load_all_tools

                tool_list = load_all_tools(
                    tool_config_path=self.tool_config_path,
                    function_tool_path=self.function_tool_path,
                )
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning(
                    "Failed to initialize tools (tool_config_path=%s, function_tool_path=%s): %s",
                    self.tool_config_path,
                    self.function_tool_path,
                    e,
                )
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json") or parquet_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key

            if processor is not None:

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc, key=self.prompt_key)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        images, videos, audios = self._process_multi_modal_info(
                            messages, self.image_patch_size, self.config
                        )
                        if images is None and videos is None and audios is None:
                            # only text prompt
                            return len(
                                processor.tokenizer(
                                    text=raw_prompt,
                                    add_special_tokens=False,  # avoid adding special tokens
                                    return_attention_mask=False,
                                )["input_ids"]
                            )
                        else:
                            # multi-modal prompt
                            return len(
                                build_multimodal_processor_inputs(
                                    processor,
                                    text=[raw_prompt],
                                    images=images,
                                    videos=videos,
                                    audio=audios,
                                    mm_processor_kwargs=self.mm_processor_kwargs,
                                )["input_ids"][0]
                            )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        # Keep explicit tokenization to avoid transformers version default changes.
                        apply_kwargs.pop("tokenize", None)
                        apply_kwargs.pop("return_dict", None)
                        apply_kwargs.pop("return_tensors", None)

                        tokenized_prompt = tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, tokenize=True, **apply_kwargs
                        )
                        return len(normalize_token_ids(tokenized_prompt))
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict, key: str):
        """Replace multimodal placeholders in messages with structured content.

        This is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image} or {"type": "image", **image}
        - <video>: {"type": "video", "video": video} or {"type": "video", **video}
        - <audio>: {"type": "audio", "audio": audio} or {"type": "audio", **audio}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = example[key]
        # When concatenating multimodal datasets, get will return None for samples without a modality column.
        images = example.get(self.image_key, None) or []
        videos = example.get(self.video_key, None) or []
        audios = example.get(self.audio_key, None) or []

        # Resolve any remote CLIMB media handles ("climb://<relpath>") to local
        # cached paths before the placeholder loop, so the rest of this method
        # (and the downstream processor / rollout) sees ordinary local files.
        climb_cfg = getattr(self, "_climb_media_cfg", None)
        if climb_cfg is not None and (images or videos):
            from verl.utils.climb import has_climb_refs, resolve_climb_media

            if has_climb_refs(images, videos):
                images, videos = resolve_climb_media(images, videos, climb_cfg)
                example[self.image_key] = images
                example[self.video_key] = videos

        # Mixed text+multimodal FSDP safety: give text-only rows a dummy blank
        # image so the vision tower runs uniformly on every rank (a text-only
        # micro-batch otherwise skips the vision all-gather while an image one
        # performs it → collective desync → NCCL timeout). Idempotent: only
        # fires when an unbuilt (string-content) user message is present, so
        # re-accessing an already-built cached row is a no-op.
        if (
            climb_cfg is not None
            and getattr(climb_cfg, "force_multimodal", False)
            and not images
            and not videos
        ):
            target = next(
                (m for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)),
                None,
            )
            if target is not None:
                from verl.utils.climb import dummy_image

                images = [dummy_image()]
                example[self.image_key] = images
                target["content"] = "<image>\n" + target["content"]
                RLHFDataset._dummy_inject_count = getattr(RLHFDataset, "_dummy_inject_count", 0) + 1
                if RLHFDataset._dummy_inject_count <= 5 or RLHFDataset._dummy_inject_count % 200 == 0:
                    logger.warning(
                        "[climb force_multimodal] injected dummy image into text-only row "
                        "(data_source=%s, count=%d)",
                        example.get("data_source", "?"),
                        RLHFDataset._dummy_inject_count,
                    )

        # CLIMB image-only policy: flatten every <video> into N <image> frames so
        # the whole pipeline (train + eval) is uniformly image-format — no video
        # backend, and no per-rank video-token imbalance in FSDP. resolve_climb_media
        # above already turned each climb video ref into a list of extracted frame
        # paths; here we splice those frames into the image stream *in placeholder
        # order* and rewrite the <video> token to N consecutive <image> tokens. Each
        # frame carries a small max_pixels cap so a clip stays ~<2k vision tokens.
        if climb_cfg is not None and videos:
            vmp = getattr(climb_cfg, "video_max_pixels", None)

            def _frames_of(v):
                if isinstance(v, dict):
                    v = v.get("video", v)
                frames = v if isinstance(v, list) else [v]
                out = []
                for fr in frames:
                    fr = os.fspath(fr) if isinstance(fr, os.PathLike) else fr
                    out.append({"image": fr, "max_pixels": vmp} if vmp else {"image": fr})
                return out

            per_video_frames = [_frames_of(v) for v in videos]
            new_images: list = []
            ii = vi = 0
            for message in messages:
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                rebuilt = []
                for seg in re.split("(<image>|<video>|<audio>)", content):
                    if seg == "<image>":
                        if ii < len(images):
                            new_images.append(images[ii])
                            ii += 1
                        rebuilt.append("<image>")
                    elif seg == "<video>":
                        frames = per_video_frames[vi] if vi < len(per_video_frames) else []
                        vi += 1
                        new_images.extend(frames)
                        rebuilt.append("<image>" * len(frames))
                    else:
                        rebuilt.append(seg)
                message["content"] = "".join(rebuilt)
            if ii < len(images):  # defensive: keep any non-placeholder images
                new_images.extend(images[ii:])
            images = new_images
            videos = []
            example[self.image_key] = images
            example[self.video_key] = []
            RLHFDataset._video_flatten_count = getattr(RLHFDataset, "_video_flatten_count", 0) + 1
            if RLHFDataset._video_flatten_count <= 5 or RLHFDataset._video_flatten_count % 200 == 0:
                logger.warning(
                    "[climb video->image] flattened %d video(s) -> %d frame-images "
                    "(data_source=%s, count=%d)",
                    len(per_video_frames),
                    sum(len(f) for f in per_video_frames),
                    example.get("data_source", "?"),
                    RLHFDataset._video_flatten_count,
                )

        image_offset, video_offset, audio_offset = 0, 0, 0
        for message in messages:
            if not images and not videos and not audios:
                continue
            assert self.processor is not None, "processor is needed to process multimodal data"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>|<audio>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                        content_list.append({"type": "image", "image": image})
                    elif isinstance(image, dict):
                        if "bytes" in image:
                            image["image"] = Image.open(BytesIO(image["bytes"]))
                        elif "image" in image and isinstance(image["image"], str) and not os.path.isabs(image["image"]):
                            image = dict(image)
                            image["image"] = os.path.join(self.data_source_dir, image["image"])
                        content_list.append({"type": "image", **image})
                    elif isinstance(image, str | os.PathLike):
                        image = os.fspath(image)
                        if not os.path.isabs(image):
                            image = os.path.join(self.data_source_dir, image)
                        content_list.append({"type": "image", "image": image})
                    else:
                        raise TypeError(
                            f"image must be dict, PIL.Image, or path-like, unsupported image type: {type(image)}"
                        )
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    video = videos[video_offset]
                    if isinstance(video, dict):
                        video = dict(video)
                        if "video" in video and isinstance(video["video"], str | os.PathLike):
                            video["video"] = os.fspath(video["video"])
                            if not os.path.isabs(video["video"]):
                                video["video"] = os.path.join(self.data_source_dir, video["video"])
                        content_list.append({"type": "video", **video})
                    elif isinstance(video, str | os.PathLike):
                        video = os.fspath(video)
                        if not os.path.isabs(video):
                            video = os.path.join(self.data_source_dir, video)
                        content_list.append({"type": "video", "video": video})
                    elif isinstance(video, list):
                        video = [os.fspath(frame) if isinstance(frame, os.PathLike) else frame for frame in video]
                        content_list.append({"type": "video", "video": video})
                    else:
                        raise TypeError(
                            f"video must be dict, list, or path-like, unsupported video type: {type(video)}"
                        )
                    video_offset += 1
                elif segment == "<audio>":
                    assert audio_offset < len(audios), f"audio_offset {audio_offset} >= len(audios) {len(audios)}"
                    audio = audios[audio_offset]
                    if isinstance(audio, dict):
                        payload = dict(audio)
                        payload["type"] = "audio"
                        if "audio" not in payload and "audio_url" not in payload:
                            payload = {"type": "audio", "audio": audio}
                        content_list.append(payload)
                    else:
                        content_list.append({"type": "audio", "audio": audio})
                    audio_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        assert audio_offset == len(audios), f"audio_offset {audio_offset} != len(audios) {len(audios)}"
        return messages

    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict: dict = self.dataframe[item]
        row_dict["raw_prompt"] = self._build_messages(row_dict, key=self.prompt_key)

        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)
        row_dict.pop(self.audio_key, None)

        # TODO(wuxibin): We still need a dummy tensor to make sure DataProto.batch is not empty.
        # Remove this after deprecate DataProto by TensorDict.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.

        This method is called by AgentLoop (e.g SingleTurnAgentLoop) before apply_chat_template to
        the `raw_prompt` from dataset. User may customize RLHFDataset and override this method to
        support custom vision extraction.

        >>> messages = kwargs["raw_prompt"]
        >>> images, videos = RLHFDataset.process_vision_info(messages, image_patch_size)
        >>> videos, video_metadatas = zip(*videos)
        >>> raw_prompt = processor.apply_chat_template(messages, tokenize=False)
        >>> inputs = processor(text=[raw_prompt], images=images, videos=videos,
        ...                    video_metadata=video_metadatas, do_sample_frames=False)

        Args:
            messages: List of messages from dataset `raw_prompt`.
            image_patch_size: Image patch size for processor.
            config: Config for dataset.

        Returns:
            images: List of images.
            videos: List of videos, each video is a tuple of (video_tensor, video_metadata).
        """
        from qwen_vl_utils import process_vision_info

        # Pre-load image references that point to filesystem paths so that one
        # corrupt file cannot crash the rollout (qwen_vl_utils.fetch_image
        # otherwise propagates PIL's OSError "broken data stream"). Bad images
        # are replaced with a 224x224 black placeholder.
        cls._sanitize_image_messages(messages)

        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos

    @staticmethod
    def _sanitize_image_messages(messages: list[dict]) -> None:
        """In-place replace any unreadable image with a black PIL placeholder.

        Each replacement emits a WARNING so a flood of them is visible in the
        training log; that signal indicates a systemic data issue rather than
        the occasional corrupt file we expect to tolerate.
        """
        import logging as _logging
        from io import BytesIO
        from PIL import Image as _PILImage

        _log = _logging.getLogger(__name__)

        def _placeholder(image_ref: str, error: Exception):
            _log.warning(
                "[RLHFDataset.process_vision_info] using black placeholder for unreadable image '%s' (%s: %s)",
                image_ref,
                type(error).__name__,
                error,
            )
            return _PILImage.new("RGB", (224, 224), color=(0, 0, 0))

        def _try_load(image_field):
            # image_field is the value of an image content dict (str path,
            # dict with bytes/image, or already a PIL.Image).
            if isinstance(image_field, _PILImage.Image):
                try:
                    image_field.load()
                    return image_field
                except Exception as e:
                    return _placeholder("<PIL.Image instance>", e)
            if isinstance(image_field, str):
                path = image_field[len("file://"):] if image_field.startswith("file://") else image_field
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    img = _PILImage.open(BytesIO(data))
                    img.load()
                    return img
                except Exception as e:
                    return _placeholder(path, e)
            return image_field  # unknown type — leave for fetch_image to handle

        for msg in messages or []:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                # qwen_vl_utils accepts the image at item["image"] OR top-level dict keys.
                if "image" in item:
                    item["image"] = _try_load(item["image"])
                elif "url" in item:
                    item["image"] = _try_load(item["url"])
                    item.pop("url", None)

    @classmethod
    def _extract_audio_info(cls, messages: list[dict]) -> list[Any]:
        audios = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                if "audio" in item:
                    audios.append(item["audio"])
                elif "audio_url" in item:
                    audios.append(item["audio_url"])
                else:
                    audios.append({k: v for k, v in item.items() if k != "type"})
        return audios or None

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[Any], list[Any]]:
        has_visual = any(
            isinstance(message.get("content"), list)
            and any(isinstance(item, dict) and item.get("type") in {"image", "video"} for item in message["content"])
            for message in messages
        )
        if has_visual:
            from qwen_vl_utils import process_vision_info

            # Replace any unreadable image with a black placeholder before
            # qwen_vl_utils decodes it, so one corrupt file can't crash the
            # rollout's vision pipeline (the sibling process_vision_info() does
            # the same; this is the path the agent-loop rollout actually calls).
            cls._sanitize_image_messages(messages)
            images, videos = process_vision_info(
                messages, image_patch_size=image_patch_size, return_video_metadata=True
            )
        else:
            images, videos = None, None
        audios = cls._extract_audio_info(messages)
        return images, videos, audios

    @classmethod
    async def process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[Any], list[Any]]:
        return cls._process_multi_modal_info(messages, image_patch_size=image_patch_size, config=config)

    def split(self, num_splits: int):
        """
        split the dataset into num_splits sub-datasets
        Args:
            num_splits: specified number of splits
        Returns:
            List[RLHFDataset]: list of RLHFDataset splits
        Raises:
            ValueError: if num_splits is not a positive integer
        """
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        if not hasattr(self, "dataframe"):
            raise AttributeError(
                "dataframe not found in RLHFDataset\n"
                "reason: _read_files_and_tokenize() not called or Parquet file loading failed"
            )
        if self.dataframe is None:
            raise ValueError("RLHFDataset dataframe 为 None!")

        total_samples = len(self.dataframe)
        print(f"total_samples: {total_samples}")
        if total_samples == 0:
            raise ValueError("Cannot split an empty dataset")

        # Calculate effective sample count after dropping remainders if needed
        if total_samples % num_splits != 0:
            total_samples = total_samples - (total_samples % num_splits)
            logging.warning(f"Dropping {len(self.dataframe) % num_splits} samples, effective samples: {total_samples}")

        split_size = total_samples // num_splits
        splits = []

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else total_samples

            split_dataframe = self.dataframe.select(range(start_idx, end_idx))

            split_dataset = RLHFDataset(
                data_files=self.data_files,
                tokenizer=self.tokenizer,
                config=self.config,
                processor=self.processor,
                max_samples=self.max_samples,
            )
            split_dataset.dataframe = split_dataframe
            split_dataset.serialize_dataset = self.serialize_dataset
            split_dataset.original_data_files = self.original_data_files

            splits.append(split_dataset)

        return splits


def get_dataset_class(data_config: DictConfig):
    """Get RLHF dataset class.

    Args:
        data_config: The data config.

    Returns:
        dataset_cls: The dataset class.
    """

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls
