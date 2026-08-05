# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("dapo")
class DAPORewardManager(RewardManagerBase):
    """DAPO Reward Manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        overlong_buffer_cfg = config.reward.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0, (
                "overlong_buffer.len must be positive when overlong penalty is enabled,"
                f"but got {self.overlong_buffer_cfg.len}."
                "To disable the overlong penalty, set overlong_buffer.enable = False"
            )

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        # Copy so we don't mutate the stored dict; thread the train/val flag through to
        # compute_score (reward-evolution mode is training-only and reads _is_validation).
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        extra_info["_is_validation"] = bool(data.meta_info.get("validate", False))
        # The trained trajectory is not always the one worth grading. The retrieval
        # agent loop trains a fraction of rollouts on the phase-1 retrieval DECISION,
        # whose tokens are a tool call, and passes the answer that decision led to via
        # `graded_answer`. Without this, those rollouts get graded on the tool call —
        # empty answer, format_ok=0, score pinned at the 0/N constant — which teaches
        # the model that searching is always wrong, the exact opposite of the intent.
        #
        # NOTE the lookup path. agent_loop.py builds the reward DataProto with a FIXED
        # key set and packs every AgentLoopOutput.extra_fields into the single column
        # `tool_extra_fields`; a top-level "graded_answer" column never exists. Reading
        # only the top level silently no-ops, which is exactly how this bug survived a
        # previous "fix" and left ~25% of each batch ungraded for a whole run.
        #
        # The retrieval telemetry below (queries issued, the evidence the model
        # actually saw, per-rollout counters) reaches the reward through the SAME
        # column, and would hit the same trap read any other way. The reward needs
        # `retrieval_context` in particular because it is deliberately stripped out
        # of `solution_str` before grading — the answer grader must not see the
        # passages, but the retrieval-coverage grader must.
        tef = data_item.non_tensor_batch.get("tool_extra_fields")
        tef = tef if isinstance(tef, dict) else {}

        graded_answer = data_item.non_tensor_batch.get("graded_answer")
        if not isinstance(graded_answer, str) or not graded_answer.strip():
            graded_answer = tef.get("graded_answer")
        if isinstance(graded_answer, str) and graded_answer.strip():
            extra_info["graded_answer"] = graded_answer

        for _k in (
            "retrieval_context", "search_queries", "n_search", "n_queries",
            "retrieval_hits", "retrieval_error", "retrieval_truncated",
            "answer_rescued", "budget_exhausted",
        ):
            _v = tef.get(_k)
            if _v is not None:
                extra_info.setdefault(_k, _v)

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
