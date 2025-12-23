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
import inspect
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional,Dict, Any, List
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
from verl.utils.dataset.count_mm_tokens import compute_modality_token_breakdown

logger = logging.getLogger(__name__)


def processor_supports_video(processor: ProcessorMixin) -> bool:
    """
    Check if a processor supports video inputs by inspecting its __call__ signature.

    Args:
        processor: The processor to check

    Returns:
        True if the processor supports video parameter, False otherwise
    """
    return False
    # if processor is None:
    #     return False
    # else:
    #     return True

    # try:
    #     sig = inspect.signature(processor.__call__)
    #     params = sig.parameters
    #     # return false if it's Gemma3Processor, which doesn't support video
    #     if "Gemma3Processor" in processor.__class__.__name__:
    #         return False
    #
    #     # Check if 'videos' is a parameter
    #     if 'videos' in params:
    #         param = params['videos']
    #         # Verify it can be used as a keyword argument
    #         if param.kind in (inspect.Parameter.KEYWORD_ONLY,
    #                           inspect.Parameter.POSITIONAL_OR_KEYWORD):
    #             return True
    # except (ValueError, TypeError, AttributeError):
    #     logger.debug("Cannot inspect processor __call__ signature")
    #
    # return False



def processor_supports_video(processor: ProcessorMixin) -> bool:
    """
    Check if a processor supports video inputs by inspecting its __call__ signature.

    Args:
        processor: The processor to check

    Returns:
        True if the processor supports video parameter, False otherwise
    """
    return False
    # if processor is None:
    #     return False
    # else:
    #     return True

    # try:
    #     sig = inspect.signature(processor.__call__)
    #     params = sig.parameters
    #     # return false if it's Gemma3Processor, which doesn't support video
    #     if "Gemma3Processor" in processor.__class__.__name__:
    #         return False
    #
    #     # Check if 'videos' is a parameter
    #     if 'videos' in params:
    #         param = params['videos']
    #         # Verify it can be used as a keyword argument
    #         if param.kind in (inspect.Parameter.KEYWORD_ONLY,
    #                           inspect.Parameter.POSITIONAL_OR_KEYWORD):
    #             return True
    # except (ValueError, TypeError, AttributeError):
    #     logger.debug("Cannot inspect processor __call__ signature")
    #
    # return False

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

def assert_homogeneous(batch_list: List[Dict[str, Any]]):
    sigs = {b.get("modality_signature") for b in batch_list}
    if len(sigs) != 1:
        raise AssertionError(f"Non-homogeneous batch signatures: {sigs}")

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
    # data list is the batch list
    # NOTE: we assert homogeneous if the modality signatures are not homogeneous
    # assert_homogeneous(data_list) # assert if not homogeneous

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
        
        # Essentially getting all the different keys.
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")

        # NOTE: SET AUDIO KEY AS AUDIOS
        self.audio_key = config.get("audio_key", "audios")

        # NOTE: SET MODALITIES, split the images and videos
        self.modalities = set(config.get("modalities", "images,videos").split(","))

        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 4096)
        print("WARNING: max_prompt_length is set to", self.max_prompt_length)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")

        # TODO: Check whether this is true
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        if isinstance(data_files, str):
            self.base_dir = os.path.dirname(os.path.abspath(data_files))
        else:
            self.base_dir = os.path.dirname(os.path.abspath(data_files[0]))
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        
        # Load format prompt from file if specified
        self.format_prompt_path = config.get("format_prompt", "examples/format_prompt/default.jinja")
        self.format_prompt = self._load_format_prompt()
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self._download()
        self._read_files_and_tokenize() # essentially this is prepared first before _getitem

    def _load_format_prompt(self) -> Optional[Template]:
        """Load format prompt from file if specified."""
        if self.format_prompt_path:
            print("Loading format prompt from:", self.format_prompt_path)
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

        #TODO_TARPO
        features = datasets.Features({
            "problem": datasets.Value("string"),
            "answer":  datasets.Value("string"),
            "images":  datasets.Sequence(datasets.Value("string")),
            "videos":  datasets.Sequence(datasets.Value("string")),
            "audios":  datasets.Sequence(datasets.Value("string")),  # <- force list of strings
            "dataset": datasets.Value("string"),
            "task": datasets.Value("string"),
            "class_label": datasets.Value("string"),
            "texts":   datasets.Sequence(datasets.Value("string")),
            "modality_signature": datasets.Value("string"),
            "ext_video_feats": datasets.Sequence(datasets.Value("string")),  # <- optional, default []
            "ext_audio_feats": datasets.Sequence(datasets.Value("string")),  # <- optional, default []
        })

        # features = datasets.Features({
        #     "problem": datasets.Value("string"),
        #     "answer":  datasets.Value("string"),
        #     "images":  datasets.Sequence(datasets.Value("string")),
        #     "videos":  datasets.Sequence(datasets.Value("string")),
        #     "audios":  datasets.Sequence(datasets.Value("string")),  # <- force list of strings
        #     "dataset": datasets.Value("string"),
        #     "texts":   datasets.Sequence(datasets.Value("string")),
        #     "modality_signature": datasets.Value("string"),
        #     # TODO: THE BOTTOM TWO ARE JUST FOR DEBUGGING FOR NOW; remove when running the other scripts
        #     "ext_video_feats": datasets.Sequence(datasets.Value("string")),  # <- optional, default []
        #     "ext_audio_feats": datasets.Sequence(datasets.Value("string")),  # <- optional, default []
        # })

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

                    # pass tool schemas if available so the processor can format prompts
                    apply_kwargs = dict(**self.apply_chat_template_kwargs)
                    if self.tool_schemas is not None:
                        apply_kwargs["tools"] = self.tool_schemas

                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False,  **self.apply_chat_template_kwarg
                    )

                    processor_kwargs = {"text": [raw_prompt]}

                    if "images" in self.modalities and image_key in doc and len(doc[image_key]) > 0:
                        images = [process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]]
                        processor_kwargs["images"] = images
                    
                    else:
                        images = None
                    

                    if "videos" in self.modalities and video_key in doc and len(doc[video_key]) > 0:
                        videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                        videos = list(videos)
                        video_metadata = list(video_metadata)
                        videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                        processor_kwargs["videos"] = videos

                        # NOTE: verl updated
                        processor_kwargs["videos_kwargs"] = videos_kwargs
                    else:
                        videos = None
                        videos_kwargs = {}
                    
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
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

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

    def _build_messages(self, example: dict, convert_video_to_images: bool = False):
        """
        This appears to be called twice, once during maybe_filter_out_long_prompts, and another time during getitems
        """
        messages: list = example.get(self.prompt_key)
        if isinstance(messages, str):
            messages = [messages]

        format_prompt = ("You FIRST think about the reasoning process as an internal monologue and then "
                         "provide the final answer. The reasoning process MUST BE enclosed within <think> "
                         "</think> tags. The final answer MUST BE put in \\boxed{}.")

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

    def _process_demographic_info(self, demographic_info: str) -> str:
        """
        Process demographic information string to groups, separated by commas.
        Example input: "demo": "sex: Male, age: 68"
        Example output: "M,A3"
        Age is grouped into ranges: 0-25 (A1), 26-50 (A2), 51-75 (A3), 76+ (A4).
        """
        if not demographic_info:
            return ""

        groups = []
        for item in demographic_info.split(","):
            key, value = item.split(":")
            key = key.strip().lower()
            value = value.strip().lower()

            if key == 'sex':
                groups.append(value[0].upper())  # M or F
            elif key == 'age':
                try:
                    age = int(value)
                except ValueError:
                    try:
                        age = float(value)
                    except ValueError:
                        groups.append("UNK")
                        continue
                if age <= 25:
                    groups.append("A1")
                elif age <= 50:
                    groups.append("A2")
                elif age <= 75:
                    groups.append("A3")
                else:
                    groups.append("A4")
        return ",".join(groups)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        PROCESSING ONE ROW AT A TIME
        """
        row_dict: dict = self.dataframe[item]

        # NOTE: save the modality signature to this; 
        sig = row_dict.get("modality_signature", None)

        if not isinstance(sig, str) or len(sig.strip()) == 0:
            raise ValueError(
                f"[Dataset] Missing modality_signature for idx={item}. "
                "Preprocess your JSONL with signatures first."
            )
        
        # save the debug_prompts for debugging purposes
        row_dict["debug_prompts"] = row_dict[self.prompt_key]
        # TODO: Placeholder for now;
        row_dict["data_source"] = "unknown"

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

        # create the video_ext and audio ext paths:
        # TODO: This currently works for only the fist video within the .pt list
        # i.e. "ext_video_feats": [path1, path2, ...], we only take the first one
        video_rel_path = row_dict.get('ext_video_feats', None)
        if video_rel_path:
            video_rel_path = video_rel_path[0]
            row_dict["ext_video_feats_path"] = os.path.join(self.base_dir, video_rel_path)
        else:
            row_dict["ext_video_feats_path"] = None
        
        audio_rel_path = row_dict.get('ext_audio_feats', None)
        if audio_rel_path:
            audio_rel_path = audio_rel_path[0]
            row_dict["ext_audio_feats_path"] = os.path.join(self.base_dir, audio_rel_path)
        else:
            row_dict["ext_audio_feats_path"] = None

        # NOTE: BUILD_MESSAGES IS CALLED TWICE; 
        # NOTE: FIRST TIME IS TO GET THE LENGTH OF THE RAW PROMPT AND FILTER OUT 
        # NOTE: PROMPTS THAT DO NOT FIT THE LENGTH; 
        # NOTE: SECOND TIME IS TO BUILD THE MESSAGE TO BE PASSED INTO THE MODEL

        # Check if processor supports video to determine if we need to convert tags
        convert_video_to_images = False
        if self.processor is not None and self.video_key in row_dict and row_dict.get(self.video_key, None) is not None and len(row_dict[self.video_key]) > 0:
            convert_video_to_images = not processor_supports_video(self.processor)


        messages = self._build_messages(row_dict, convert_video_to_images=convert_video_to_images)

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
        
        # NOTE: DEBUGGING
        dbg = False
        if dbg:
            print(f"[getitem] idx=? ds={row_dict.get('dataset')} src={row_dict.get('data_source')} "
                f"modalities={self.modalities}")

        if self.processor is not None:
            # THIS CHUNK IS BASICALLY ABOUT PROCESSING ALL THE MODALITIES
            from verl.utils.dataset.vision_utils import process_image, process_video
            from verl.utils.dataset.audio_utils import process_audio

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="System prompt modified")
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
            multi_modal_data = {}
            # processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}

            images = None

            if "images" in self.modalities and self.image_key in row_dict and row_dict.get(self.image_key, None) is not None and len(row_dict[self.image_key]) > 0:
                images = []
                for image in row_dict.get(self.image_key):
                    image = os.path.join(self.base_dir, image) if isinstance(image, str) else image
                    images.append(process_image(image, image_patch_size=self.image_patch_size))

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            video_frames_as_images = None
            videos_kwargs = None
            if "videos" in self.modalities and self.video_key in row_dict and row_dict.get(self.video_key, None) is not None and len(row_dict[self.video_key]) > 0:
                row_dict_videos = row_dict.get(self.video_key)
                for video in row_dict_videos:
                    video = os.path.join(self.base_dir, video) if isinstance(video, str) else video
                    video, video_metadata = process_video(video,
                                                          image_patch_size=self.image_patch_size,
                                                          return_video_metadata=True)
                    if videos is None:
                        videos = [video]
                    else:
                        videos.append(video)

                    if videos_kwargs is None:
                        videos_kwargs = {"video_metadata": [video_metadata], "do_sample_frames": False}
                    else:
                        videos_kwargs['video_metadata'].append(video_metadata)
                
                if processor_supports_video(self.processor):
                    # Processor supports video, use it directly
                    # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                    # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                    multi_modal_data["video"] = [(video.numpy(), metadata) for video, metadata in zip(videos, videos_kwargs['video_metadata'], strict=True)]
                else:
                    # NOTE: Updated VERL qwen3 vl utilizes these lines instead of processing the videos directly
                    # Processor doesn't support video, convert to images
                    video_frames_as_images = []
                    for video_tensor in videos:
                        # video_tensor is shape [n_frames, 3, H, W]
                        # Convert each frame to PIL Image
                        for frame_idx in range(video_tensor.shape[0]):
                            frame = video_tensor[frame_idx]  # [3, H, W]
                            # Convert from tensor to PIL Image
                            # Assuming the tensor is in uint8 format [0, 255]
                            frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, 3]
                            from PIL import Image
                            frame_image = Image.fromarray(frame_np.astype('uint8'), 'RGB')
                            video_frames_as_images.append(frame_image)
                    
                 # Append video frames to existing images
                    if images is None:
                        images = video_frames_as_images
                    else:
                        images.extend(video_frames_as_images)

                    # Update multi_modal_data with the combined images
                    multi_modal_data["image"] = images

                    # Clear videos since we've converted them to images
                    videos = None
                
            if (
                "audio" in self.modalities
                and self.audio_key in row_dict
                and row_dict.get(self.audio_key)
                and len(row_dict[self.audio_key]) > 0
            ):
                audios_np = []
                audios_np_sr = []
                audio_secs = []

                for audio in row_dict[self.audio_key]:
                    audio_path = os.path.join(self.base_dir, audio) if isinstance(audio, str) else audio
                    audio_tensor, sr = process_audio(audio_path, self.processor)

                    # What BOTH HF and vLLM need:
                    arr = audio_tensor.detach().cpu().numpy().astype("float32")
                    audios_np.append(arr)
                    audios_np_sr.append((arr, int(sr)))
                    audio_secs.append(_sec_from_array(arr, sr))

                # HF (Whisper / Omni processor) path
                multi_modal_data["audio"] = audios_np_sr  # Store numpy arrays (it should not accept tuples)

                # processor_kwargs["audio"] = audios_np  # Pass numpy arrays to processor

            try:
                if processor_supports_video(self.processor):
                    # Build kwargs dict and filter out None values
                    kwargs = {
                        "text": [raw_prompt],
                        "images": images,
                        "videos": videos,
                        "videos_kwargs": videos_kwargs,
                        "return_tensors": "pt"
                    }
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    model_inputs = self.processor(**kwargs)
                else:
                    # Only pass images parameter if processor doesn't support video
                    # Build kwargs dict and filter out None values
                    kwargs = {
                        "text": [raw_prompt],
                        "images": images,
                        "return_tensors": "pt"
                    }
                    kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    model_inputs = self.processor(**kwargs)
            except Exception as e:
                logger.error("Error processing multi-modal data for item %d: %s", item, e)
                traceback.print_exc()
                # Process as text only - remove multimodal tags and data from row_dict
                original_prompt = row_dict.get(self.prompt_key)
                if isinstance(original_prompt, str):
                    # Remove <image> and <video> tags from string prompt
                    row_dict[self.prompt_key] = re.sub(r'<image>|<video>', '', original_prompt)
                elif isinstance(original_prompt, list):
                    # Handle list of messages - remove tags from content
                    cleaned_prompt = []
                    for msg in original_prompt:
                        if isinstance(msg, dict):
                            new_msg = copy.deepcopy(msg)
                            if isinstance(new_msg.get("content"), str):
                                new_msg["content"] = re.sub(r'<image>|<video>', '', new_msg["content"])
                            cleaned_prompt.append(new_msg)
                        elif isinstance(msg, str):
                            cleaned_prompt.append(re.sub(r'<image>|<video>', '', msg))
                        else:
                            cleaned_prompt.append(msg)
                    row_dict[self.prompt_key] = cleaned_prompt
                # Clear multimodal data
                row_dict[self.image_key] = []
                row_dict[self.video_key] = []
                # Rebuild messages without multimodal content
                messages = self._build_messages(row_dict, convert_video_to_images=False)
                text_only_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                model_inputs = self.tokenizer(text_only_prompt, return_tensors="pt", add_special_tokens=False)
                # drop multi_modal_data
                multi_modal_data = {}

            # model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            # model_inputs = self.processor(
            #     text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            # )

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
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        
        #TODO_DEBUG double check the prompt length here as it may lead to error
        #TODO_DEBUG it may need to be set higher, to 4096 or something
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
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
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()

        # dump all keys in row_dict that has numerical or string values or list of strings in extra info
        for key, value in row_dict.items():
            if key not in row_dict["extra_info"]:
                if isinstance(value, (int, float, str)):
                    row_dict["extra_info"][key] = value
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    row_dict["extra_info"][key] = value

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