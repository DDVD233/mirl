# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Retrieval-augmented agent loop for the self-evolving HealthBench RL run.

A ``ToolAgentLoop`` specialization that enforces a **mandatory 2-turn** structure:

    turn 1 (assistant): reason + call ``search_medical_kb``
    tool  turn         : retrieved passages (loss-masked)
    turn 2 (assistant): reason + final answer  → terminate

Two additions over the base ``ToolAgentLoop``:

1. **Thinking-budget forcing** on every assistant turn (base loop lacks it): the
   reasoning channel is capped and force-closed so runaway thinking cannot eat
   the response window — this is what stabilized the HealthBench v4–v6 runs.
2. **Guaranteed retrieval**: if the model's first turn produces no parseable tool
   call, a ``search_medical_kb`` call is synthesized from the raw user question,
   so retrieval happens even before the policy reliably emits tool calls.

Set via ``actor_rollout_ref.rollout.agent.default_agent_loop=retrieval_tool_agent``
with ``multi_turn.enable=True``, ``max_assistant_turns=2``, ``format=hermes``, and
a ``tool_config_path`` exposing ``search_medical_kb`` (MedicalRetrievalTool).
"""

import json
import logging
import os

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.think_budget import generate_with_think_budget
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Name of the retrieval tool; the mandatory-retrieval fallback synthesizes a call
# to it. Must match the tool_schema.function.name in the tool config yaml.
RETRIEVAL_TOOL_NAME = os.getenv("RETRIEVAL_TOOL_NAME", "search_medical_kb")


def _last_user_text(messages: list[dict]) -> str:
    """Extract the last user turn's text (handles str or multimodal-list content)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
    return ""


@register("retrieval_tool_agent")
class RetrievalToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.think_budget = int(os.getenv("VERL_THINK_BUDGET_TOKENS", "0"))
        self.think_close_ids: list[int] = (
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False) if self.think_budget > 0 else []
        )

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict, ignore_termination: bool = False
    ) -> AgentState:
        """Generate one assistant turn (with think budget), then decide next state.

        Mirrors ``ToolAgentLoop._handle_generating_state`` but (a) routes generation
        through the thinking-budget forcing and (b) synthesizes a retrieval call on
        the first turn if the model produced none, so retrieval is mandatory.
        """
        # Inject tool parser stop tokens so generation halts after each tool call.
        if self.tool_parser.stop_token_ids:
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

        gen_kwargs = {
            "image_data": agent_data.image_data,
            "video_data": agent_data.video_data,
            "audio_data": agent_data.audio_data,
            "mm_processor_kwargs": agent_data.mm_processor_kwargs,
        }
        remaining = self.response_length - len(agent_data.response_mask)
        with simple_timer("generate_sequences", agent_data.metrics):
            if self.think_budget > 0:
                response_ids, response_mask, response_logprobs, output = await generate_with_think_budget(
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    think_budget=self.think_budget,
                    think_close_ids=self.think_close_ids,
                    response_length=self.response_length,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    gen_kwargs=gen_kwargs,
                    max_new_tokens=remaining,
                )
            else:
                output: TokenOutput = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    **gen_kwargs,
                )
                response_ids = list(output.token_ids)
                response_mask = [1] * len(response_ids)
                response_logprobs = list(output.log_probs) if output.log_probs else None

        # num_preempted bookkeeping (matches base loop).
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields or {})
        elif output.extra_fields:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += response_mask
        if response_logprobs:
            agent_data.response_logprobs += response_logprobs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Termination checks (identical to base loop).
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed).
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        # Mandatory retrieval: on the FIRST assistant turn, if the model produced no
        # usable retrieval call, synthesize one from the raw user question so every
        # rollout retrieves before answering.
        if agent_data.assistant_turns == 1 and RETRIEVAL_TOOL_NAME in active_tools:
            has_retrieval = any(tc.name == RETRIEVAL_TOOL_NAME for tc in agent_data.tool_calls)
            if not has_retrieval:
                query = _last_user_text(agent_data.messages)[:1000]
                agent_data.tool_calls = [
                    FunctionCall(name=RETRIEVAL_TOOL_NAME, arguments=json.dumps({"query": query}))
                ]

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED
