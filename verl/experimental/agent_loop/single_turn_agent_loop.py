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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.think_budget import generate_with_think_budget
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Thinking-budget forcing (opt-in via VERL_THINK_BUDGET_TOKENS > 0): phase 1
        # decodes at most think_budget tokens; if the reasoning channel is still open
        # (no "</think>") we INJECT "</think>\n\n" — loss-masked like a tool response —
        # and phase 2 decodes the answer with the remaining budget. Runaway thinking
        # then cannot eat the whole response window: every rollout yields a gradeable
        # answer instead of a zero-reward truncation cliff (and GRPO groups keep
        # reward variance instead of collapsing to all-zero).
        self.think_budget = int(os.getenv("VERL_THINK_BUDGET_TOKENS", "0"))
        self.think_close_ids: list[int] = (
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False) if self.think_budget > 0 else []
        )

    async def _generate_with_think_budget(
        self, vllm_prompt_ids: list[int], sampling_params: dict[str, Any], gen_kwargs: dict[str, Any]
    ) -> tuple[list[int], list[int], list[float] | None, TokenOutput]:
        """Two-phase generation with a hard cap on the thinking channel.

        Returns (response_ids, response_mask, response_logprobs, last TokenOutput).
        """
        return await generate_with_think_budget(
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            think_budget=self.think_budget,
            think_close_ids=self.think_close_ids,
            response_length=self.response_length,
            prompt_ids=vllm_prompt_ids,
            sampling_params=sampling_params,
            gen_kwargs=gen_kwargs,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        # For video inputs with processor, use tokenizer-style prompt_ids for vLLM.
        # processor.apply_chat_template() creates per-frame video placeholders that
        # conflict with vLLM's own video expansion (causes IndexError in MRoPE).
        # The tokenizer generates a single placeholder per video that vLLM can expand.
        if videos is not None and self.processor is not None:
            vllm_prompt_ids = await self.apply_chat_template_for_vllm(messages)
        else:
            vllm_prompt_ids = prompt_ids

        # 3. generate sequences
        metrics = {}
        gen_kwargs = {
            "image_data": images,
            "video_data": videos,
            "audio_data": audios,
            "mm_processor_kwargs": mm_processor_kwargs,
        }
        with simple_timer("generate_sequences", metrics):
            if self.think_budget > 0:
                response_ids, response_mask, response_logprobs, output = await self._generate_with_think_budget(
                    vllm_prompt_ids, sampling_params, gen_kwargs
                )
            else:
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=vllm_prompt_ids,
                    sampling_params=sampling_params,
                    **gen_kwargs,
                )
                response_ids = list(output.token_ids)
                response_mask = [1] * len(response_ids)
                response_logprobs = list(output.log_probs) if output.log_probs else None
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
