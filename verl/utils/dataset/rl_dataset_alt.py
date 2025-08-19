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
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from jinja2 import Template
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import warnings

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import time, os, math, warnings

logger = logging.getLogger(__name__)

def _tok_est_from_hw(H, W):
    # 28x28 -> 1 "visual token" heuristic
    return math.ceil(H/28) * math.ceil(W/28)

def _sec_from_array(arr, sr):
    try:
        return round(len(arr) / float(sr), 3)
    except Exception:
        return "?"

def _p99(xs):
    xs = sorted(xs)
    if not xs: return 0
    k = int(0.99*(len(xs)-1))
    return xs[k]


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    data_list = [d for d in data_list if d is not None]
    if not data_list:
        return None

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
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        
        # Essentially getting all the different keys.
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")

        # NOTE: SET AUDIO KEY AS AUDIOS
        self.audio_key = config.get("audio_key", "audios")

        # NOTE: SET MODALITIES, split the images and videos
        self.modalities = set(config.get("modalities", "images,videos").split(","))

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")

        # TODO: Check whether this is true
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        if isinstance(data_files, str):
            self.base_dir = os.path.dirname(os.path.abspath(data_files))
        else:
            self.base_dir = os.path.dirname(os.path.abspath(data_files[0]))

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        
        # Load format prompt from file if specified
        self.format_prompt_path = config.get("format_prompt", "examples/format_prompt/default.jinja")
        self.format_prompt = self._load_format_prompt()

        self._download()
        self._read_files_and_tokenize() # essentially this is prepared first before _getitem

    def _load_format_prompt(self) -> Optional[Template]:
        """Load format prompt from file if specified."""
        if self.format_prompt_path:
            with open(self.format_prompt_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            return Template(template_content)
        return None

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []

        features = datasets.Features({
            "problem": datasets.Value("string"),
            "answer":  datasets.Value("string"),
            "images":  datasets.Sequence(datasets.Value("string")),
            "videos":  datasets.Sequence(datasets.Value("string")),
            "audios":  datasets.Sequence(datasets.Value("string")),  # <- force list of strings
            "dataset": datasets.Value("string"),
            "texts":   datasets.Sequence(datasets.Value("string")),
        })

        for parquet_file in self.data_files:
            # read parquet files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file, features=features)["train"]
            elif parquet_file.endswith(".json") or parquet_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file, features=features)["train"]
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}. Only .parquet, .json, .jsonl are supported.")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # PROCESSING THE DATAFRAME for TRAINING
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # NOTE: filter out too long prompts, because the prompts can become very long
        # when the audio is appended.

        if self.filter_overlong_prompts:
            # NOTE: FILTER OUT THE LONG PROMPTS SO THAT THEY FIT THE LENGTH
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key
            audio_key = self.audio_key

            if processor is not None:
                # print(f"KEANE: PROCESSOR FOUND")
                from verl.utils.dataset.vision_utils import process_image, process_video
                from verl.utils.dataset.audio_utils import process_audio

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    processor_kwargs = {"text": [raw_prompt]}
                    
                    if "images" in self.modalities and image_key in doc and len(doc[image_key]) > 0:
                        images = [process_image(image) for image in doc[image_key]]
                        processor_kwargs["images"] = images

                    if "videos" in self.modalities and video_key in doc and len(doc[video_key]) > 0:    
                        videos = [process_video(video) for video in doc[video_key]]
                        processor_kwargs["videos"] = videos

                    if "audio" in self.modalities and audio_key in doc and doc.get(audio_key, None) is not None and len(doc[audio_key]) > 0:
                        # processing of audio
                        # print(f"KEANE: Processing audio within rl dataset file")
                        # audios = [process_audio(audio, processor) for audio in doc[audio_key]]
                        # processor_kwargs["audio"] = audios

                        # PATCH
                        audios = []
                        audio_tuples = []  # Keep tuples for multi_modal_data
                        for audio in doc.get(self.audio_key):
                            audio_path = os.path.join(self.base_dir, audio) if isinstance(audio, str) else audio
                            audio_data, sampling_rate = process_audio(audio_path, self.processor)
                            audio_tuples.append((audio_data, sampling_rate))
                            # audios.append(audio_data.numpy())  # Convert to numpy array for Whisper
                            audios.append(audio_data.detach().cpu().numpy().astype("float32"))

                        processor_kwargs["audio"] = audios  # Pass numpy arrays to processor
                    # TODO: cannot process the audio inputs
                    # print(f"KEANE: Processor class is {processor.__class__.__name__}")
                    # print(f"KEANE: Printing the processor_kwargs, {processor_kwargs}")
                    # Assume that all are in tensors already, hence there is no return_tensors = "pt"
                    return len(processor(**processor_kwargs)["input_ids"][0])

            else:
                # print(f"KEANE: PROCESSOR NOT FOUND")
                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

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

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        """
        This appears to be called twice, once during maybe_filter_out_long_prompts, and another time during getitems
        """
        messages: list = example.get(self.prompt_key)
        if isinstance(messages, str):
            messages = [messages]

        # NOTE: Before building, check if there is multimodal content
        has_multimodal = (
            ("images" in self.modalities and self.image_key in example) or
            ("videos" in self.modalities and self.video_key in example) or
            ("audio" in self.modalities and self.audio_key in example)
        )
        
        if has_multimodal:
            new_messages = []
            for message in messages:
                new_message = copy.deepcopy(message)
                if isinstance(new_message, str):
                    new_message = {"role": "user", "content": new_message}
                content = new_message["content"]
                
                # Apply format prompt to the entire content first if template is loaded
                if self.format_prompt:
                    content = self.format_prompt.render(content=content)

                image_count = len(example.get(self.image_key, []))
                video_count = len(example.get(self.video_key, []))
                audio_count = len(example.get(self.audio_key, []))
                image_tag_count = content.count("<image>")
                video_tag_count = content.count("<video>")
                audio_tag_count = content.count("<audio>")

                # NOTE: Apppending the <image>, <video>, <audio> tags when they are missing
                if image_tag_count < image_count:
                    content = "<image>" * (image_count - image_tag_count) + content
                    logger.warning("<image> tag count is less than image count, adding missing <image> tags."
                                   " content: %s", content)
                if video_tag_count < video_count:
                    content = "<video>" * (video_count - video_tag_count) + content
                    logger.warning("<video> tag count is less than video count, adding missing <video> tags."
                                 " content: %s", content)
                if audio_tag_count < audio_count:
                    content = "<audio>" * (audio_count - audio_tag_count) + content
                    logger.warning("<audio> tag count is less than audio count, adding missing <audio> tags."
                                   " content: %s", content)

                content_list = []
                # Build regex pattern based on enabled modalities
                tag_patterns = []
                if "images" in self.modalities:
                    tag_patterns.append("<image>")
                if "videos" in self.modalities:
                    tag_patterns.append("<video>")
                if "audio" in self.modalities:
                    tag_patterns.append("<audio>")
                
                # NOTE: Denote the different patterns based on the tag.
                # TODO: Double check what this does
                if tag_patterns:
                    pattern = "(" + "|".join(tag_patterns) + ")"
                    segments = re.split(pattern, content)
                    segments = [item for item in segments if item != ""]
                    for segment in segments:
                        if segment == "<image>" and "images" in self.modalities:
                            content_list.append({"type": "image"})
                        elif segment == "<video>" and "videos" in self.modalities:
                            content_list.append({"type": "video"})
                        elif segment == "<audio>" and "audio" in self.modalities:
                            content_list.append({"type": "audio"})
                        else:
                            content_list.append({"type": "text", "text": segment})
                else:
                    content_list.append({"type": "text", "text": content})
                new_message["content"] = content_list
                new_messages.append(new_message)
        else:
            new_messages = copy.deepcopy(messages)
            if isinstance(new_messages, str):
                new_messages = [{"role": "user", "content": new_messages}]
            elif isinstance(new_messages, list) and isinstance(new_messages[0], str):
                new_messages = [{"role": "user", "content": new_messages}]
            
            # Apply format prompt to text-only messages if template is loaded
            if self.format_prompt and len(new_messages) > 0:
                for i, msg in enumerate(new_messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            new_messages[i]["content"] = self.format_prompt.render(content=content)
        return new_messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        is_timeseries = False
        vision_path = row_dict['images'][0] if 'images' in row_dict and len(row_dict['images']) != 0 else None
        if vision_path is None:  # this may be video
            vision_path = row_dict['videos'][0] if 'videos' in row_dict and len(row_dict['videos']) != 0 else None
        if vision_path is None:  # this may be time series only
            vision_path = row_dict['time_series'][0] if 'time_series' in row_dict and len(
                row_dict['time_series']) != 0 else ''
            is_timeseries = True
        prompt_str = row_dict[self.prompt_key]

        if 'How long will the patient stay in the hospital?' in prompt_str:
            row_dict["data_source"] = "multimodal"
            row_dict["dataset"] = "los_prediction"
        elif 'Will the patient survive for at least 48 hours?' in prompt_str:
            row_dict["data_source"] = "multimodal"
            row_dict["dataset"] = "48_ihm"
        elif len(vision_path) != 0:
            try:
                row_dict["data_source"] = vision_path.split("/")[0]
                row_dict["dataset"] = vision_path.split("/")[1]
            except IndexError:
                row_dict["data_source"] = "unknown"
                row_dict["dataset"] = "unknown"
                print(
                    f"Failed to parse vision path: {vision_path}. The annotation is {row_dict}. Using default values.")
        elif is_timeseries:
            row_dict["data_source"] = "ecg"
            # dataset already set in json
        else:
            raise ValueError("No modality found.")

        if 'reward_model' not in row_dict:
            if 'answer' in row_dict:
                answer = row_dict['answer']
            elif 'ground_truth' in row_dict:
                answer = row_dict['ground_truth']
            else:
                raise ValueError("No answer or ground_truth found in the row_dict.")
            row_dict['reward_model'] = {'ground_truth': answer}

        for key, item in row_dict.items():
            if item is None:
                row_dict[key] = []

        # NOTE: BUILD_MESSAGES IS CALLED TWICE; 
        # NOTE: FIRST TIME IS TO GET THE LENGTH OF THE RAW PROMPT AND FILTER OUT 
        # NOTE: PROMPTS THAT DO NOT FIT THE LENGTH; 
        # NOTE: SECOND TIME IS TO BUILD THE MESSAGE TO BE PASSED INTO THE MODEL

        messages = self._build_messages(row_dict)

        if "audio" in self.modalities:
            # NOTE: Set the following prompt for qwen omni when we are training on audio
            messages.insert(0, {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                                             "capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            })

        model_inputs = {}
        dbg = True
        if dbg:
            print(f"[getitem] idx=? ds={row_dict.get('dataset')} src={row_dict.get('data_source')} modalities={self.modalities}")

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video
            from verl.utils.dataset.audio_utils import process_audio

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="System prompt modified")
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if dbg:
                print(f"[prompt] raw_prompt_chars={len(raw_prompt)}")

            # ---------- CHANGED: build kwargs with ONLY present modalities; no duplicates ----------
            processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}

            # Optional prompt gating helper
            def mentions(tag: str) -> bool:
                return (f"<{tag}>" in raw_prompt) or (f"<{tag}_0>" in raw_prompt)

            # IMAGES
            images = None
            if ("images" in self.modalities and self.image_key in row_dict
                and row_dict.get(self.image_key) and len(row_dict[self.image_key]) > 0):
                # Optional: uncomment to require prompt placeholders
                # if not mentions("image"): pass
                # else:
                images = []
                for image in row_dict[self.image_key]:
                    path = os.path.join(self.base_dir, image) if isinstance(image, str) else image
                    images.append(process_image(path))
                processor_kwargs["images"] = images
                if dbg:
                    print(f"[image] n={len(images)} shapes={[tuple(x.size()) if hasattr(x,'size') else 'np' for x in images]}")

            # VIDEOS
            videos = None
            if ("videos" in self.modalities and self.video_key in row_dict
                and row_dict.get(self.video_key) and len(row_dict[self.video_key]) > 0):
                # Optional: placeholder gate
                # if mentions("video"):
                videos = []
                for v in row_dict[self.video_key]:
                    path = os.path.join(self.base_dir, v) if isinstance(v, str) else v
                    t = process_video(path, debug=dbg, name_hint=os.path.basename(str(path)))
                    videos.append(t)  # [T,3,H,W] uint8 on CPU
                processor_kwargs["videos"] = videos
                if dbg:
                    shapes = [tuple(v.shape) for v in videos]
                    toks = []
                    for (T, C, H, W) in shapes:
                        toks.append(_tok_est_from_hw(H, W) * T)
                    print(f"[video] n={len(videos)} shapes={shapes} est_tokens={toks} sum_est_tokens={sum(toks)}")

            # AUDIO
            audios_np = None
            if ("audio" in self.modalities and self.audio_key in row_dict
                and row_dict.get(self.audio_key) and len(row_dict[self.audio_key]) > 0):
                # Optional: placeholder gate
                # if mentions("audio"):
                audios_np = []
                audio_secs = []
                for a in row_dict[self.audio_key]:
                    path = os.path.join(self.base_dir, a) if isinstance(a, str) else a
                    a_tensor, sr = process_audio(path, self.processor)  # clipped mono, 16k
                    arr = a_tensor.detach().cpu().numpy().astype("float32")
                    audios_np.append(arr)
                    audio_secs.append(round(len(arr)/float(sr), 3))
                processor_kwargs["audio"] = audios_np
                if dbg:
                    print(f"[audio] n={len(audios_np)} secs_each={audio_secs} total_secs≈{round(sum(audio_secs),3)}")

            # ---------- CHANGED: drop temporaries & do NOT stash media into row_dict ----------
            # (we intentionally do NOT create multi_modal_data or row_dict["multi_modal_*"])
            # Remove local references ASAP
            for _nm in ("images", "videos", "audios_np", "audio_secs"):
                if _nm in locals():
                    try: del locals()[_nm]
                    except Exception: pass

            # ---------- CHANGED: processor call; skip sample on error ----------
            try:
                t0 = time.time()
                model_inputs = self.processor(**processor_kwargs)   # stays CPU
                dt = (time.time() - t0) * 1000
                if dbg:
                    ids = model_inputs.get("input_ids")
                    lens = [len(x) for x in ids] if ids is not None else []
                    med = (sorted(lens)[len(lens)//2] if lens else "-")
                    print(f"[processor] ok in {dt:.1f}ms; input_ids lens={lens} min/med/max={ (min(lens) if lens else '-') }/{med}/{ (max(lens) if lens else '-') }")
            except Exception as e:
                print(f"[processor][ERROR] {type(e).__name__}: {e} — skipping sample")
                return None  # collate_fn should drop Nones

            # ---------- unchanged: extract ids/mask, postprocess ----------
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            model_inputs.pop("second_per_grid_ts", None)

            # >>> NEW: keep a PRUNED, CPU‑only media payload for rollout generation
            _mm_keep = {}
            for k in ("images", "videos", "audio", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
                if k in model_inputs and model_inputs[k] is not None:
                    v = model_inputs[k]
                    # ensure CPU + plain lists where possible (no CUDA!)
                    try:
                        _mm_keep[k] = v if isinstance(v, list) else v.tolist()
                    except Exception:
                        _mm_keep[k] = v  # e.g., list of CPU tensors / numpy arrays

            # Store ONLY this small payload; do NOT store frames twice, and don't keep model_inputs itself
            row_dict["multi_modal_inputs"] = _mm_keep

            # CHANGED: drop model_inputs entirely so nothing large lingers
            del model_inputs

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            toks = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = toks.pop("input_ids")
            attention_mask = toks.pop("attention_mask")
            del toks

        # postprocess (unchanged)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # position ids (unchanged logic)
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=None,   # not stored; processor will infer if needed
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        # set minimal required fields on row_dict
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # raw_prompt_ids (unchanged)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # tail (unchanged, small metadata only)
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict
    
    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
