# Copyright 2025 Individual Contributor: Mert Unsal
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

import inspect
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase


@register("batch")
class BatchRewardLoopManager(RewardLoopManagerBase):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        config (dict): Configuration for the reward manager.
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        compute_score (callable): The function to compute the rewards.
        reward_router_address (str): The address of the reward router.
        reward_model_tokenizer (Tokenizer): The tokenizer for the reward model.
        reward_fn_key (str): The key to use for the reward function.
        num_examine (int): The number of responses to examine.
        **reward_kwargs: Additional keyword arguments to pass to the reward function.
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        reward_fn_key="data_source",
        num_examine=10,
        **reward_kwargs
    ):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score) if compute_score else False
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.reward_fn_key = reward_fn_key
        self.num_examine = num_examine
        self.reward_kwargs = reward_kwargs
        self.already_printed = {}

    async def compute_batch_scores(self, data: DataProto):
        """Compute scores for a batch of data."""
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = await self.loop.run_in_executor(
                None, lambda ids=valid_response_ids: self.tokenizer.decode(ids, skip_special_tokens=True)
            )
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

        # Call compute_score with batch parameters
        if self.is_async_reward_score:
            scores = await self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
                reward_router_address=self.reward_router_address,
                reward_model_tokenizer=self.reward_model_tokenizer,
                **self.reward_kwargs,
            )
        else:
            scores = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_sources=data_sources,
                    solution_strs=responses_str,
                    ground_truths=ground_truths,
                    extra_infos=extras,
                    reward_router_address=self.reward_router_address,
                    reward_model_tokenizer=self.reward_model_tokenizer,
                    **self.reward_kwargs,
                ),
            )

        return scores, valid_response_lengths

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item or batch."""
        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            return {"reward_score": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}

        # Check if this is actually a single item or a batch
        is_single = len(data) == 1

        if is_single:
            # Handle single item processing similar to naive.py
            data_item = data[0]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
            if tool_extra_fields is not None:
                extra_info.update(tool_extra_fields.items())

            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            response_str = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            )

            # Call compute_score with singular parameters for single item
            if self.compute_score:
                if self.is_async_reward_score:
                    result = await self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        reward_router_address=self.reward_router_address,
                        reward_model_tokenizer=self.reward_model_tokenizer,
                        **self.reward_kwargs,
                    )
                else:
                    result = await self.loop.run_in_executor(
                        None,
                        lambda: self.compute_score(
                            data_source=data_source,
                            solution_str=response_str,
                            ground_truth=ground_truth,
                            extra_info=extra_info,
                            reward_router_address=self.reward_router_address,
                            reward_model_tokenizer=self.reward_model_tokenizer,
                            **self.reward_kwargs,
                        ),
                    )
            else:
                # Default score if no compute_score function
                result = 0.0

            reward_extra_info = {}
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key] = value
            else:
                score = result
                reward_extra_info["acc"] = score

            return {"reward_score": score, "reward_extra_info": reward_extra_info}
        else:
            # Handle batch processing
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)

            scores, valid_response_lengths = await self.compute_batch_scores(data)

            prompt_ids = data.batch["prompts"]
            data_sources = data.non_tensor_batch[self.reward_fn_key]
            rewards = []

            for i in range(len(data)):
                length = valid_response_lengths[i].item()
                score = scores[i] if scores else 0.0

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                rewards.append(reward)
                reward_tensor[i, length - 1] = reward

                # Print sample outputs for debugging
                data_source = data_sources[i]
                if self.already_printed.get(data_source, 0) < self.num_examine:
                    response_str = await self.loop.run_in_executor(
                        None,
                        lambda ids=data.batch["responses"][i][:length]: self.tokenizer.decode(
                            ids, skip_special_tokens=True
                        ),
                    )
                    prompt_str = await self.loop.run_in_executor(
                        None,
                        lambda ids=data.batch["prompts"][i]: self.tokenizer.decode(ids, skip_special_tokens=True),
                    )
                    ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", scores[i] if scores else 0.0)
                    self.already_printed[data_source] = self.already_printed.get(data_source, 0) + 1

            # Store accuracy scores
            data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

            return {"reward_score": reward_tensor, "reward_extra_info": dict(reward_extra_info)}