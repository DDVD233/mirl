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

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
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
        for parquet_file in self.data_files:
            # read parquet files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json") or parquet_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
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
                print(f"KEANE: PROCESSOR FOUND")
                from verl.utils.dataset.vision_utils import process_image, process_video
                from verl.utils.dataset.audio_utils import process_audio

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    processor_kwargs = {"text": [raw_prompt]}
                    
                    if "images" in self.modalities and image_key in doc:
                        images = [process_image(image) for image in doc[image_key]]
                        processor_kwargs["images"] = images
                        
                    if "videos" in self.modalities and video_key in doc:
                        videos = [process_video(video) for video in doc[video_key]]
                        processor_kwargs["videos"] = videos
                        
                    if "audio" in self.modalities and audio_key in doc and doc.get(audio_key, None) is not None:
                        # TODO: make sure that this path is actually happening
                        # processing of audio
                        print(f"KEANE: Processing audio within rl dataset file")
                        audios = [process_audio(audio, processor) for audio in doc[audio_key]]
                        processor_kwargs["audio"] = audios
                    # TODO: cannot process the audio inputs
                    print(f"KEANE: Printing the processor_kwargs, {processor_kwargs}")
                    return len(processor(**processor_kwargs)["input_ids"][0])

            else:
                print(f"KEANE: PROCESSOR NOT FOUND")
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

        if self.processor is not None:
            # THIS CHUNK IS BASICALLY ABOUT PROCESSING ALL THE MODALITIES
            from verl.utils.dataset.vision_utils import process_image, process_video
            from verl.utils.dataset.audio_utils import process_audio

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="System prompt modified")
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}

            if "images" in self.modalities and self.image_key in row_dict and row_dict.get(self.image_key, None) is not None and len(row_dict[self.image_key]) > 0:
                images = []
                for image in row_dict.get(self.image_key):
                    image = os.path.join(self.base_dir, image) if isinstance(image, str) else image
                    images.append(process_image(image))

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images
                processor_kwargs["images"] = images

            if "videos" in self.modalities and self.video_key in row_dict and row_dict.get(self.video_key, None) is not None and len(row_dict[self.video_key]) > 0:
                videos = []
                for video in row_dict.get(self.video_key):
                    video = os.path.join(self.base_dir, video) if isinstance(video, str) else video
                    videos.append(process_video(video))

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]
                processor_kwargs["videos"] = videos

            # NOTE: PROCESSING OF THE AUDIO TUPLES
            if "audio" in self.modalities and self.audio_key in row_dict and row_dict.get(self.audio_key, None) is not None and len(row_dict[self.audio_key]) > 0:
                audios = []
                audio_tuples = []  # Keep tuples for multi_modal_data
                for audio in row_dict.get(self.audio_key):
                    audio_path = os.path.join(self.base_dir, audio) if isinstance(audio, str) else audio
                    audio_data, sampling_rate = process_audio(audio_path, self.processor)
                    audio_tuples.append((audio_data, sampling_rate))
                    audios.append(audio_data.numpy())  # Convert to numpy array for Whisper

                multi_modal_data["audio"] = audio_tuples  # Store tuples for reference
                processor_kwargs["audio"] = audios  # Pass numpy arrays to processor

            # TODO: Please check whether the model is processing the "audio" correctly, the processor that we are using is qwen 2.5 OMNI
            print(f"KEANE: Processing multimodal data with processor {self.processor.__class__.__name__} ")
            print(f"KEANE: Processor kwargs: {processor_kwargs}")
            model_inputs = self.processor(**processor_kwargs)

            # NOTE: all text should be processed by self.processor()
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index
            
            # NOTE: printing out whether this runs
            print("KEANE: Running getting the rope index of input ids")
            
            # NOTE: OBTAIN ROPE of rotary positional embeddings. ROPE encodes position by rotating components of query/key vectors
            # This is just for to get relative position in terms of angular differences etc.
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        # Essentially training with the different input ids etc.
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
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
