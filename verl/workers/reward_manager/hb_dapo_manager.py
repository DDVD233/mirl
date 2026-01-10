# Copyright 2025 Individual Contributor
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

from collections import defaultdict
from typing import Callable

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("hb_dapo")
class HBDAPORewardManager(AbstractRewardManager):
    """
    DAPO-style reward manager for human behavior tasks.

    Supports CLS (exact match) and QA (cosine similarity) tasks with
    extracted boxed content from responses.

    Args:
        tokenizer: The tokenizer to use for decoding the responses.
        num_examine: The number of responses to examine per data source.
        compute_score: The function to compute the rewards (batch function).
        reward_fn_key: The key to use for the reward function (default: "data_source").
        max_resp_len: Maximum response length (in tokens).
        overlong_buffer_cfg: Configuration for overlong penalty (optional).
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score: Callable,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Process data and compute rewards, following DAPO pattern."""

        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # First, collect all data for batch processing
        batch_data_sources = []
        batch_solution_strs = []
        batch_ground_truths = []
        batch_extra_infos = []
        batch_task_ids = []
        batch_valid_response_lengths = []
        batch_prompt_strs = []  # Store for printing

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            task_id = data_item.non_tensor_batch.get("task", None)

            batch_data_sources.append(data_source)
            batch_solution_strs.append(response_str)
            batch_ground_truths.append(ground_truth)
            batch_extra_infos.append(extra_info)
            batch_task_ids.append(task_id)
            batch_valid_response_lengths.append(valid_response_length)
            batch_prompt_strs.append(prompt_str)

        # Compute all scores at once using batch function
        batch_results = self.compute_score(
            data_sources=batch_data_sources,
            solution_strs=batch_solution_strs,
            ground_truths=batch_ground_truths,
            extra_infos=batch_extra_infos,
            task_ids=batch_task_ids,
        )

        # Now process each item with the computed scores
        for i in range(len(data)):
            data_item = data[i]
            result = batch_results[i]
            valid_response_length = batch_valid_response_lengths[i]
            data_source = batch_data_sources[i]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score

            # Add overlong penalty (following dapo.py pattern)
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            # Print examination outputs
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

                prompt_str = batch_prompt_strs[i]
                response_str = batch_solution_strs[i]
                ground_truth = batch_ground_truths[i]

                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
