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

Structure enforced per rollout:

    turn 1 (assistant): reason + call ``search_medical_kb``   (MANDATORY)
    tool  turn         : retrieved passages (loss-masked)
    [ up to ``max_searches`` more search rounds, budget permitting ]
    FINAL turn (assistant): reason + answer, tools disabled     (GUARANTEED)

Why a guaranteed final answer turn: reasoning models naturally want to issue a
*second* refined search after seeing the first passages. A hard "turn 2 = answer"
cap made those rollouts end on a dangling tool call with NO answer (empty → zero
reward — this sank the first v7 val to 0.18). So we let the model search up to
``max_searches`` times, then force one answer turn where tools are disabled and a
budget reserve guarantees room to actually respond.

Additions over the base ``ToolAgentLoop``:
1. **Thinking-budget forcing** on every assistant turn (base loop lacks it) — caps
   runaway reasoning; search turns get a smaller budget than the answer turn.
2. **Guaranteed retrieval**: if turn 1 has no parseable tool call, one is
   synthesized from the raw user question.
3. **Answer-budget reserve**: a search is only taken if enough response budget
   would remain for a full answer turn; otherwise we answer now.

Config: ``agent.default_agent_loop=retrieval_tool_agent``, ``multi_turn.enable=True``,
``multi_turn.format=qwen3_coder`` (the model's native tool-call XML), and a
``tool_config_path`` exposing ``search_medical_kb`` (MedicalRetrievalTool). Tunables
via env: ``VERL_MAX_SEARCHES`` (2), ``VERL_SEARCH_THINK_BUDGET`` (2048),
``VERL_ANSWER_RESERVE_TOKENS`` (3500).
"""

import json
import logging
import os
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.think_budget import generate_with_think_budget
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

RETRIEVAL_TOOL_NAME = os.getenv("RETRIEVAL_TOOL_NAME", "search_medical_kb")

RETRIEVE_FIRST_INSTRUCTION = (
    f"You have a medical knowledge search tool `{RETRIEVAL_TOOL_NAME}`. Before "
    "answering ANY clinical question, you MUST first call it to retrieve supporting "
    "evidence, then ground your final answer in the returned passages. Write a "
    "focused query naming the key clinical entities and the specific fact you need "
    "(drug, dose, contraindication, threshold, guideline). You may search again to "
    "refine, but once you have enough evidence, give your final answer."
)

ANSWER_NOW_INSTRUCTION = (
    "You now have enough retrieved evidence. Provide your FINAL answer to the user's "
    "question, grounded in the passages above. Do NOT call any tools."
)


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
        self.search_think_budget = int(os.getenv("VERL_SEARCH_THINK_BUDGET", "2048"))
        self.max_searches = int(os.getenv("VERL_MAX_SEARCHES", "2"))
        self.answer_reserve = int(os.getenv("VERL_ANSWER_RESERVE_TOKENS", "3500"))
        self.think_close_ids: list[int] = (
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False) if self.think_budget > 0 else []
        )

    def _inject_instruction(self, agent_data: AgentData) -> None:
        msgs = agent_data.messages
        if msgs and msgs[0].get("role") == "system":
            c = msgs[0].get("content")
            if isinstance(c, list):
                msgs[0] = {**msgs[0], "content": list(c) + [{"type": "text", "text": "\n\n" + RETRIEVE_FIRST_INSTRUCTION}]}
            else:
                msgs[0] = {**msgs[0], "content": f"{c}\n\n{RETRIEVE_FIRST_INSTRUCTION}"}
        else:
            msgs.insert(0, {"role": "system", "content": RETRIEVE_FIRST_INSTRUCTION})

    async def _generate_turn(
        self, agent_data: AgentData, sampling_params: dict, think_budget: int, allow_tools: bool
    ) -> None:
        """Generate one assistant turn (with think-budget forcing) and append it."""
        sp = sampling_params
        if allow_tools and self.tool_parser.stop_token_ids:
            stop = list(set((sp.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sp = {**sp, "stop_token_ids": stop}

        gen_kwargs = {
            "image_data": agent_data.image_data,
            "video_data": agent_data.video_data,
            "audio_data": agent_data.audio_data,
            "mm_processor_kwargs": agent_data.mm_processor_kwargs,
        }
        remaining = self.response_length - len(agent_data.response_mask)
        with simple_timer("generate_sequences", agent_data.metrics):
            if think_budget > 0:
                response_ids, response_mask, response_logprobs, output = await generate_with_think_budget(
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    think_budget=think_budget,
                    think_close_ids=self.think_close_ids,
                    response_length=self.response_length,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sp,
                    gen_kwargs=gen_kwargs,
                    max_new_tokens=remaining,
                )
            else:
                output: TokenOutput = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sp,
                    **gen_kwargs,
                )
                response_ids = list(output.token_ids)
                response_mask = [1] * len(response_ids)
                response_logprobs = list(output.log_probs) if output.log_probs else None

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0
        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields or {})
        elif output.extra_fields:
            mgs = output.extra_fields.get("max_global_steps", None)
            if mgs:
                agent_data.extra_fields["max_global_steps"] = mgs

        agent_data.assistant_turns += 1
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += response_mask
        if response_logprobs:
            agent_data.response_logprobs += response_logprobs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

    async def _append_masked(self, agent_data: AgentData, messages: list[dict]) -> None:
        """Tokenize non-assistant turns (tool responses / instructions) and append
        them loss-masked (the qwen3_coder / generic tool-response path)."""
        agent_data.messages.extend(messages)
        response_ids = await self.apply_chat_template(messages, images=None, videos=None, remove_system_prompt=True)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

    async def _run_tool_calls(self, agent_data: AgentData, tool_calls) -> None:
        """Execute retrieval calls and append their responses (loss-masked)."""
        import asyncio

        tasks = [self._call_tool(tc, agent_data.tools_kwargs, agent_data) for tc in tool_calls[: self.max_parallel_calls]]
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        add_messages = []
        for tool_response, tool_reward, _ in responses:
            add_messages.append({"role": "tool", "content": tool_response.text or ""})
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
        await self._append_masked(agent_data, add_messages)
        agent_data.user_turns += 1

    @rollout_trace_op
    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics={},
            request_id=uuid4().hex,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = self.tool_schemas

        self._inject_instruction(agent_data)
        agent_data.prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=agent_data._active_tool_schemas,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        tool_schemas = [t.tool_schema for t in agent_data._active_tools.values()]
        n_search = 0
        # Cost of one more search round we must keep budget for: its think + query + tool response.
        search_cost = self.search_think_budget + 900
        while True:
            remaining = self.response_length - len(agent_data.response_mask)
            if remaining <= 0:
                break
            budget_for_search = remaining - search_cost >= self.answer_reserve
            answer_turn = (n_search >= self.max_searches) or (n_search >= 1 and not budget_for_search)

            think_budget = self.think_budget if answer_turn else min(self.think_budget, self.search_think_budget) if self.think_budget > 0 else 0
            await self._generate_turn(agent_data, sampling_params, think_budget, allow_tools=not answer_turn)
            if answer_turn or len(agent_data.response_mask) >= self.response_length:
                break

            _, calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tool_schemas)
            search_calls = [c for c in calls if c.name == RETRIEVAL_TOOL_NAME]
            # Mandatory first retrieval: synthesize from the raw question if none.
            if n_search == 0 and not search_calls and RETRIEVAL_TOOL_NAME in agent_data._active_tools:
                from verl.experimental.agent_loop.tool_parser import FunctionCall

                q = _last_user_text(agent_data.messages)[:1000]
                search_calls = [FunctionCall(name=RETRIEVAL_TOOL_NAME, arguments=json.dumps({"query": q}))]

            if search_calls:
                await self._run_tool_calls(agent_data, search_calls)
                n_search += 1
                # If the next turn will be the forced answer, tell the model so.
                nxt_remaining = self.response_length - len(agent_data.response_mask)
                if n_search >= self.max_searches or (nxt_remaining - search_cost < self.answer_reserve):
                    await self._append_masked(agent_data, [{"role": "user", "content": ANSWER_NOW_INSTRUCTION}])
                continue
            # No search call -> the model has answered.
            break

        # Finalize (mirrors ToolAgentLoop.run).
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        mm_data = {}
        if agent_data.image_data is not None:
            mm_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            mm_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            mm_data["audios"] = agent_data.audio_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            multi_modal_data=mm_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output
