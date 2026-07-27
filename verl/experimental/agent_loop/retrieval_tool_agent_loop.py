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
import re
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

# Retrieval is OPTIONAL and passages AUGMENT (do not bound) the model's knowledge.
# This fixes the two dominant regression modes found in the 90-case analysis:
# over-anchoring (30%) — the model suppressed correct parametric knowledge because
# it wasn't in the passages — and unnecessary retrieval distracting non-factual
# tasks (ethics/formatting/translation). See healthbench-v7-retrieval memory.
RETRIEVE_FIRST_INSTRUCTION = (
    f"You are an expert physician. ANSWER DIRECTLY from your own knowledge by default — "
    f"you have a `{RETRIEVAL_TOOL_NAME}` tool but it is a RARE last resort. The vast "
    "majority of requests (writing or formatting notes/letters, explanations, ethics or "
    "refusal decisions, translation, general management, anything you know) must be "
    "answered directly WITHOUT searching. Search ONLY when you cannot recall a single "
    "specific fact (exact dose, threshold, contraindication, code, current guideline) "
    "AND getting it wrong would change the answer. When you do search, the passages "
    "AUGMENT your knowledge, they do not limit it: still state well-established facts "
    "you know even if absent from the passages, ask for missing context when ambiguous, "
    "and refuse unsafe requests. Never say 'the retrieved evidence does not contain...' "
    "about something you actually know."
)

ANSWER_NOW_INSTRUCTION = (
    "Now write your COMPLETE final answer directly — no meta-commentary about how you "
    "will format it, just the answer itself. Use the retrieved passages where helpful, "
    "but also state all relevant facts you know (not only what the passages mention). "
    "Do NOT call any tools."
)

# Rescue path: ~13% of retrieval rollouts end EMPTY because on the forced answer turn the
# tool schema is still in the prompt, so the (eager) model emits ANOTHER <tool_call> and
# ends its turn expecting results — never writing an answer. allow_tools=False only stops
# PARSING, not generation. Fix: re-render the prompt WITHOUT tools (question + retrieved
# passages as context) and generate a clean prose answer. This alone recovers ~46 of the
# 134 stock+retrieval val regressions (their no-retrieval baseline averaged 0.858).
RESCUE_SYSTEM = (
    "You are an expert physician. Write a COMPLETE, clinically sound, well-structured answer "
    "to the user's request. Preserve appropriate diagnostic uncertainty and ask for missing "
    "context when the request is ambiguous, include safety / red-flag guidance and "
    "contraindications, and be as comprehensive as a thorough expert answer requires. The "
    "reference passages AUGMENT your knowledge — they do not limit it; state well-established "
    "facts you know even if the passages omit them, and never say 'the evidence does not contain'."
)
HARD_ANSWER_INSTRUCTION = (
    "The search tool is now CLOSED and will return nothing further. Do NOT search and do NOT "
    "output any tool call. Write your COMPLETE final answer to the request above now, as plain "
    "prose, using the reference passages where helpful plus your own medical knowledge."
)

_TOOL_SPAN_RE = re.compile(r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>", re.DOTALL | re.IGNORECASE)


def _final_answer_of(text: str) -> str:
    """The graded answer: strip tool spans, then take text after the last </think>."""
    t = _TOOL_SPAN_RE.sub("", text or "")
    t = t.rsplit("</think>", 1)[-1] if "</think>" in t else t
    return t.strip()


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

    async def _generate_clean_answer(self, agent_data: AgentData, sampling_params: dict, orig_messages: list[dict]) -> None:
        """Generate the FINAL graded answer from a clean, TOOL-FREE prompt: the original
        question + the retrieved passages (if any) as context + an answer instruction.
        This replaces the rollout response, so the graded answer never carries tool-schema
        overhead and can never be an empty/dangling-tool-call turn. Used for every rollout
        (both direct and post-retrieval)."""
        passages = "\n\n".join(
            (m.get("content") or "") for m in agent_data.messages if m.get("role") == "tool"
        ).strip()
        question = _last_user_text(orig_messages)
        ctx = f"\n\nReference passages retrieved for this question:\n{passages}" if passages else ""
        answer_messages = [
            {"role": "system", "content": RESCUE_SYSTEM},
            {"role": "user", "content": f"{question}{ctx}\n\n{HARD_ANSWER_INSTRUCTION}"},
        ]
        prompt_ids = await self.apply_chat_template(
            answer_messages,
            tools=None,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
        # Reset the rollout to this clean single-turn answer and regenerate (no tools).
        agent_data.prompt_ids = prompt_ids
        agent_data.response_mask = []
        if agent_data.response_logprobs:
            agent_data.response_logprobs = []
        agent_data.metrics["retrieval_used"] = 1 if passages else 0
        await self._generate_turn(agent_data, sampling_params, self.think_budget, allow_tools=False)

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
        # ---- Phase 1: retrieval DECISION ----
        # Let the model optionally search (up to max_searches). This turn's answer text
        # is discarded; the graded answer is ALWAYS generated in phase 2 from a clean,
        # tool-free prompt. This is the root-cause fix for the two dominant regression
        # modes found in the stock+retrieval val (vs no-retrieval 0.559):
        #   * empty answers (~13% of rollouts): the forced answer turn still had the tool
        #     schema in-prompt, so the eager model emitted ANOTHER <tool_call> and ended
        #     with no answer. Generating the answer tool-free makes empties impossible.
        #   * tool-prompt overhead: the tool schema + retrieve instruction made even the
        #     DIRECT answers terser / more over-confident / drop safety+hedging (78 of 86
        #     regressions in the rare-retrieval config were direct answers). A clean
        #     tool-free answer prompt removes that overhead.
        n_search = 0
        while n_search < self.max_searches:
            if self.response_length - len(agent_data.response_mask) < self.answer_reserve:
                break
            await self._generate_turn(agent_data, sampling_params, self.search_think_budget, allow_tools=True)
            _, calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tool_schemas)
            search_calls = [c for c in calls if c.name == RETRIEVAL_TOOL_NAME]
            if not search_calls:
                break  # model chose not to (further) search
            await self._run_tool_calls(agent_data, search_calls)
            n_search += 1

        # ---- Phase 2: always generate the final answer from a clean tool-free prompt ----
        await self._generate_clean_answer(agent_data, sampling_params, messages)

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
