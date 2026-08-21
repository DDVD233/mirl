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
"""Retrieval-augmented agent loop: ONE continuous trajectory, queries included.

Emitted trajectory (a single sequence, no rebuilds):

    [system + question + tool schema]
    assistant : think + <tool_call>{queries:[...]}</tool_call>      mask=1  TRAINED
    tool      : summarized evidence brief                           mask=0
    [ up to ``max_searches`` such pairs ]
    user      : "the tool is CLOSED, answer now"                    mask=0
    assistant : think + final answer                                mask=1  TRAINED

WHY THIS SHAPE. The previous design generated the graded answer from a REBUILT,
tool-free prompt and emitted only that, resetting ``response_mask`` in the process.
The query tokens were therefore deleted from the trajectory before it ever reached
the optimizer: retrieval quality had no gradient at all and could only drift as a
side effect of other updates. Since the whole point is to learn WHAT TO RETRIEVE as
well as how to answer, the trajectory has to stay continuous. Tool responses are
loss-masked (the model did not generate them) but still conditioned on, and they
consume the response budget — which is exactly why /retrieve summarizes.

THE FAILURE THIS MUST NOT REGRESS. The rebuild existed for a reason: ~13% of v7
rollouts ended EMPTY because the forced answer turn still had the tool schema in
the prompt, so the eager model emitted ANOTHER tool call and stopped without ever
writing an answer (that sank the first v7 val to 0.18). Removing the rebuild means
that guard has to be rebuilt in-trajectory, which is what ``_generate_answer_turn``
does: stop on the tool-call marker, trim the fragment, inject a think-wrapped
masked notice, and regenerate with the marker banned outright so a second failure
is structurally impossible. The notice is think-wrapped so the reward's
"text after the last </think>" rule excludes it from the graded answer for free.

RETRIEVAL BUDGET. Optional, 0..``max_searches`` (default 2). A call beyond the
budget is answered with an explicit error tool message telling the model it must
answer now — the model sees a real refusal rather than silence.

Config: ``agent.default_agent_loop=retrieval_tool_agent``, ``multi_turn.enable=True``,
``multi_turn.format=qwen3_coder``, and a ``tool_config_path`` exposing
``search_medical_kb`` (MedicalRetrievalTool). Env tunables: ``VERL_MAX_SEARCHES`` (2),
``VERL_SEARCH_THINK_BUDGET`` (1024), ``VERL_THINK_BUDGET_TOKENS`` (3072),
``VERL_ANSWER_RESERVE_TOKENS`` (3500), ``VERL_MIN_ANSWER_TOKENS`` (1536).
"""

import asyncio
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
# Optional second tool (WebSearchTool -> Serper). Only ACTIVE when the tool config
# actually registers it; the name here just tells the loop which calls to execute
# and count. KB and web calls share ONE search budget: the budget teaches "look
# things up sparingly", and two separate allowances would double the spend the
# moment the model learns to alternate tools.
WEB_TOOL_NAME = os.getenv("WEB_SEARCH_TOOL_NAME", "web_search")

# Markers that mean "the model is starting a tool call". Used as stop strings on the
# answer turn (detector) and as bad_words on the retry (structural prevention).
TOOL_MARKERS = ("<tool_call>", "<function=")

# Retrieval is OPTIONAL and passages AUGMENT (do not bound) the model's knowledge.
# This fixes the two dominant regression modes found in the 90-case analysis:
# over-anchoring (30%) — the model suppressed correct parametric knowledge because
# it wasn't in the passages — and unnecessary retrieval distracting non-factual
# tasks (ethics/formatting/translation). See healthbench-v7-retrieval memory.
def _domain_rebrand():
    """``domains.rebrand`` from scripts/self_evolving/domains.py (env SE_DOMAIN).

    Located relative to the repo root; identity when the file is missing so a medical
    run can never fail on it. An UNKNOWN SE_DOMAIN still raises (SystemExit from the
    bundle) -- silently running medical prompts under a misspelled domain is worse.
    """
    try:
        import importlib.util
        from pathlib import Path

        import verl

        path = Path(verl.__file__).resolve().parents[1] / "scripts" / "self_evolving" / "domains.py"
        if not path.is_file():
            return lambda t: t
        spec = importlib.util.spec_from_file_location("se_domains", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.rebrand
    except Exception:  # noqa: BLE001
        return lambda t: t


_rebrand = _domain_rebrand()

# Tool names are substituted AFTER rebranding: the default KB tool is called
# `search_medical_kb`, and the bundle's "medical" rule must not rewrite an identifier.
RETRIEVE_INSTRUCTION = _rebrand(
    "You are an expert physician. You may look facts up with the `[[TOOL]]` "
    "tool before answering, at most {max_searches} time(s) in total. Search when the "
    "request turns on a specific fact you cannot recall with confidence — an exact dose, "
    "threshold, contraindication, code, or current guideline — and getting it wrong would "
    "change the answer. Answer directly, without searching, for requests you already know "
    "or that are not factual lookups (writing or formatting notes and letters, "
    "explanations, ethics or refusal decisions, translation, general management). "
    "When you do search, pass SEVERAL sub-queries covering the DIFFERENT facts the answer "
    "needs — one query returns one fact, and a thorough answer usually needs more. The "
    "retrieved passages AUGMENT your knowledge, they do not limit it: still state "
    "well-established facts you know even if absent from the passages, ask for missing "
    "context when ambiguous, and refuse unsafe requests. Never say 'the retrieved evidence "
    "does not contain...' about something you actually know."
).replace("[[TOOL]]", RETRIEVAL_TOOL_NAME)

# Appended to RETRIEVE_INSTRUCTION only when the web tool is registered. The KB is
# embedded and ends in 2019; the web tool is the route to anything newer -- but it
# returns raw search results (titles, links, snippets), not passages, so the model
# must judge source reliability itself.
WEB_INSTRUCTION = _rebrand(
    " You also have a `[[WEB]]` tool (one query per call) that searches the "
    "live web and returns raw results — title, link, snippet. Use it instead of "
    "`[[TOOL]]` for CURRENT information: guidelines revised or drugs "
    "approved in the last few years, recalls, epidemiology, or anything the knowledge "
    "base failed to find. Results are unfiltered web content: weigh each by its "
    "source, prefer guidelines, journals and regulators, and never treat a snippet "
    "from a low-quality site as fact. Both tools draw from the same total search "
    "budget of {max_searches}."
).replace("[[WEB]]", WEB_TOOL_NAME).replace("[[TOOL]]", RETRIEVAL_TOOL_NAME)

# WEB-ONLY arms (the non-medical domains register just `web_search`, no KB tool):
# the same search policy as RETRIEVE_INSTRUCTION, with the web tool as the only tool.
# Used INSTEAD of RETRIEVE_INSTRUCTION + WEB_INSTRUCTION when the KB tool is absent.
WEB_ONLY_INSTRUCTION = _rebrand(
    "You are an expert physician. You may look facts up with the `[[WEB]]` tool before "
    "answering, at most {max_searches} time(s) in total, one query per call. Search when "
    "the request turns on a specific fact you cannot recall with confidence — an exact "
    "dose, threshold, contraindication, code, or current guideline — and getting it wrong "
    "would change the answer. Answer directly, without searching, for requests you already "
    "know or that are not factual lookups (writing or formatting notes and letters, "
    "explanations, ethics or refusal decisions, translation, general management). The "
    "tool returns raw, unfiltered web results — title, link, snippet: weigh each by its "
    "source, prefer primary and authoritative sources (guidelines, journals and "
    "regulators), and never treat a snippet from a low-quality site as fact. Retrieved "
    "material AUGMENTS your knowledge, it does not limit it: still state well-established "
    "facts you know even if absent from the results, ask for missing context when "
    "ambiguous, and refuse unsafe requests. Never say 'the retrieved evidence does not "
    "contain...' about something you actually know."
).replace("[[WEB]]", WEB_TOOL_NAME)


def _web_only_variant(text: str) -> str:
    """The masked-turn texts speak of KB 'passages'; the web tool returns results."""
    return text.replace("passages", "results").replace("passage", "result")

# Delivered as a masked user turn after the last tool response.
#
# The attribution sentence is the last link in a chain that is useless without it. The KB
# now carries article titles and assembled references, and the evidence brief ends in a
# Sources block -- but nothing here previously told the policy to NAME a source, so a
# rubric criterion like "mentions the 2022 ACG guideline" or "references the 2000 NEJM
# trial by Lau et al." was unreachable even with the citation sitting in the context. The
# same omission was found and fixed one layer up in the summarizer; this is the layer that
# actually writes the graded answer.
#
# WORDED AGAINST FABRICATION, deliberately. "Cite your sources" invites a model to invent
# plausible references, which in a clinical answer is worse than citing nothing and is also
# a reward-hacking route: a judge may well credit a confident "per the 2020 ACC/AHA
# guideline" that no passage supports. Hence copy-only, verbatim, and an explicit
# instruction to attribute nothing when the passages name nothing.
HARD_ANSWER_INSTRUCTION = _rebrand(
    "The search tool is now CLOSED and will return nothing further. Do NOT search and do "
    "NOT output any tool call. Write your COMPLETE final answer to the request above now, "
    "as plain prose: clinically sound and well-structured, preserving appropriate "
    "diagnostic uncertainty, asking for missing context where the request is ambiguous, "
    "and including safety / red-flag guidance and contraindications. Use the retrieved "
    "passages where helpful PLUS your own medical knowledge — never say 'the evidence does "
    "not contain'. "
    "When a passage NAMES its source — an issuing organisation, a guideline and its year, "
    "an article title, a journal, an author — name it in your answer next to the claim it "
    "supports, copying it EXACTLY as given. Do NOT invent, guess or reconstruct a "
    "reference: no made-up trial names, years, journals or authors, and no citation for a "
    "claim that came from your own knowledge rather than a passage. If the passages name "
    "no source, state the fact without attributing it."
)

# Returned as a tool message when the model calls the tool past its budget. The model
# gets an explicit refusal rather than silence, so the behaviour is learnable.
BUDGET_EXHAUSTED_ERROR = _rebrand(
    "ERROR: retrieval budget exhausted ({n}/{n} searches used). The search tool is now "
    "CLOSED and will return nothing further. You must answer now from the passages already "
    "retrieved plus your own medical knowledge. Do NOT output another tool call."
)

# Injected INSIDE the assistant turn (loss-masked) when the answer turn tries to call
# the tool. Think-wrapped on purpose: the reward grades only the text after the LAST
# </think>, so this never reaches the graded answer or the length penalty.
CLOSED_NOTICE = _rebrand(
    "\n\n<think>\nThe retrieval tool is closed and a further call returns nothing. "
    "I will write the complete final answer now, using the passages above plus my own "
    "medical knowledge.\n</think>\n\n"
)

# Web-only counterparts of the three masked texts (see WEB_ONLY_INSTRUCTION).
WEB_ONLY_HARD_ANSWER_INSTRUCTION = _web_only_variant(HARD_ANSWER_INSTRUCTION)
WEB_ONLY_BUDGET_EXHAUSTED_ERROR = _web_only_variant(BUDGET_EXHAUSTED_ERROR)
WEB_ONLY_CLOSED_NOTICE = _web_only_variant(CLOSED_NOTICE)

# Closing tag is optional so a span truncated at the response cap is still stripped.
_TOOL_SPAN_RE = re.compile(
    r"<tool_call>.*?(?:</tool_call>|\Z)|<tool_response>.*?(?:</tool_response>|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _final_answer_of(text: str) -> str:
    """The graded answer: strip tool spans, then take text after the last </think>."""
    t = _TOOL_SPAN_RE.sub("", text or "")
    t = t.rsplit("</think>", 1)[-1] if "</think>" in t else t
    return t.strip()


@register("retrieval_tool_agent")
class RetrievalToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Which tool calls count as "a search" for budget/telemetry. The web tool
        # joins only when the tool config registered it, so arms without it are
        # byte-identical to before.
        tools = getattr(self, "tools", {}) or {}
        # WEB-ONLY: the tool config registered the web tool and NOT the KB tool (the
        # non-medical arms). The KB name then stays out of the search set, the
        # instruction names only the web tool, and the masked turns speak of results
        # rather than passages. Any other configuration is byte-identical to before.
        self.web_only = WEB_TOOL_NAME in tools and RETRIEVAL_TOOL_NAME not in tools
        self.search_tool_names = set() if self.web_only else {RETRIEVAL_TOOL_NAME}
        if WEB_TOOL_NAME in tools:
            self.search_tool_names.add(WEB_TOOL_NAME)
        self.hard_answer_instruction = (
            WEB_ONLY_HARD_ANSWER_INSTRUCTION if self.web_only else HARD_ANSWER_INSTRUCTION
        )
        self.budget_exhausted_error = (
            WEB_ONLY_BUDGET_EXHAUSTED_ERROR if self.web_only else BUDGET_EXHAUSTED_ERROR
        )
        self.answer_think_budget = int(os.getenv("VERL_THINK_BUDGET_TOKENS", "3072"))
        self.search_think_budget = int(os.getenv("VERL_SEARCH_THINK_BUDGET", "1024"))
        self.max_searches = int(os.getenv("VERL_MAX_SEARCHES", "2"))
        self.answer_reserve = int(os.getenv("VERL_ANSWER_RESERVE_TOKENS", "3500"))
        # Floor on the tokens left for the answer TEXT after its thinking budget.
        self.min_answer_tokens = int(os.getenv("VERL_MIN_ANSWER_TOKENS", "1536"))
        # Rough tokens/char for budget projection (clinical English, Qwen tokenizer).
        self.chars_per_token = float(os.getenv("VERL_CHARS_PER_TOKEN", "3.3"))
        # Cost estimates for the reserve projection: the XML tool call itself, and the
        # chat-template scaffolding around a tool turn + the close instruction.
        self.tool_call_est = int(os.getenv("VERL_TOOL_CALL_EST_TOKENS", "200"))
        self.close_est = int(os.getenv("VERL_CLOSE_EST_TOKENS", "120"))
        # Computed unconditionally: this used to be gated on the ANSWER think budget
        # while search turns pass the SEARCH budget, so with the answer budget at 0 the
        # forced close was an empty list and generate_with_think_budget silently
        # appended nothing — the reasoning channel was never actually closed.
        self.think_close_ids: list[int] = self.tokenizer.encode(
            "</think>\n\n", add_special_tokens=False
        )
        self.closed_notice_ids: list[int] = self.tokenizer.encode(
            WEB_ONLY_CLOSED_NOTICE if self.web_only else CLOSED_NOTICE, add_special_tokens=False
        )
        self.tool_marker_ids: list[int] = []
        for m in TOOL_MARKERS:
            ids = self.tokenizer.encode(m, add_special_tokens=False)
            if len(ids) == 1:  # only single-token markers work as stop_token_ids
                self.tool_marker_ids.append(ids[0])

    # ------------------------------------------------------------------ budget
    def _budget_left(self, agent_data: AgentData) -> int:
        return self.response_length - len(agent_data.response_mask)

    def _tool_span_estimate(self) -> int:
        max_chars = getattr(self, "max_tool_response_length", 0) or 4000
        return int(max_chars / self.chars_per_token) + 40

    def _can_afford_search(self, agent_data: AgentData) -> bool:
        """Reserve room for the answer turn BEFORE committing to a search round.

        Projects the whole round (assistant turn + tool span + close instruction),
        not just the current position. The old check compared the raw remaining
        budget against the reserve, which protected nothing once phase 2 reset the
        mask — and in a continuous trajectory the tool span is the biggest term.
        """
        projected = (
            self.search_think_budget + self.tool_call_est
            + self._tool_span_estimate() + self.close_est
        )
        return self._budget_left(agent_data) - projected >= self.answer_reserve

    # ------------------------------------------------------- trajectory edits
    def _append_ids_masked(self, agent_data: AgentData, ids: list[int]) -> None:
        """Append tokens the model did NOT generate (mask 0)."""
        agent_data.prompt_ids += ids
        agent_data.response_mask += [0] * len(ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(ids)

    def _rollback_to(self, agent_data: AgentData, mark: int) -> None:
        """Undo everything appended since ``len(response_mask) == mark``.

        Safe because every append to ``prompt_ids`` after the initial prompt is
        matched 1:1 by an append to ``response_mask`` (both generation branches and
        ``_append_masked``), so dropping d mask entries drops exactly the last d
        prompt_ids.
        """
        drop = len(agent_data.response_mask) - mark
        if drop <= 0:
            return
        del agent_data.prompt_ids[-drop:]
        del agent_data.response_mask[mark:]
        if agent_data.response_logprobs:
            del agent_data.response_logprobs[mark:]
        agent_data.response_ids = []

    def _trim_tool_marker(self, agent_data: AgentData, floor: int, max_drop: int = 12) -> bool:
        """Pop trailing tokens until the tail no longer shows a tool marker.

        A stop STRING truncates only vLLM's `output_text`, never `token_ids`, so the
        marker tokens are still in the trajectory and would otherwise land in the
        graded answer. `floor` keeps this from eating into an earlier, legitimate
        tool span.
        """
        dropped = 0
        while dropped < max_drop and len(agent_data.response_mask) > floor:
            tail = self.tokenizer.decode(agent_data.prompt_ids[-24:])
            if not any(m in tail for m in TOOL_MARKERS):
                break
            agent_data.prompt_ids.pop()
            agent_data.response_mask.pop()
            if agent_data.response_logprobs:
                agent_data.response_logprobs.pop()
            dropped += 1
        return dropped > 0

    def _guarded_params(self, sampling_params: dict, ban: bool) -> dict:
        """Stop on / ban the tool-call markers for the final answer turn."""
        sp = dict(sampling_params)
        if self.tool_marker_ids:
            sp["stop_token_ids"] = list(
                {*(sp.get("stop_token_ids") or []), *self.tool_marker_ids}
            )
        sp["stop"] = list({*(sp.get("stop") or []), *TOOL_MARKERS})
        if ban:
            # Masks the marker's logit whenever its prefix matches, so the retry
            # cannot emit a tool call at all.
            sp["bad_words"] = list(TOOL_MARKERS)
        return sp

    # ------------------------------------------------------------- generation
    def _inject_instruction(self, agent_data: AgentData) -> None:
        if self.web_only:
            text = WEB_ONLY_INSTRUCTION.format(max_searches=self.max_searches)
        else:
            text = RETRIEVE_INSTRUCTION.format(max_searches=self.max_searches)
            if WEB_TOOL_NAME in self.search_tool_names:
                text += WEB_INSTRUCTION.format(max_searches=self.max_searches)
        msgs = agent_data.messages
        if msgs and msgs[0].get("role") == "system":
            c = msgs[0].get("content")
            if isinstance(c, list):
                msgs[0] = {**msgs[0], "content": list(c) + [{"type": "text", "text": "\n\n" + text}]}
            else:
                msgs[0] = {**msgs[0], "content": f"{c}\n\n{text}"}
        else:
            msgs.insert(0, {"role": "system", "content": text})

    async def _generate_turn(self, agent_data: AgentData, sampling_params: dict,
                             think_budget: int) -> None:
        """Generate one assistant turn (think-budget forced) and append it, mask=1."""
        gen_kwargs = {
            "image_data": agent_data.image_data,
            "video_data": agent_data.video_data,
            "audio_data": agent_data.audio_data,
            "mm_processor_kwargs": agent_data.mm_processor_kwargs,
        }
        remaining = max(0, self._budget_left(agent_data))
        if remaining <= 0:
            return
        with simple_timer("generate_sequences", agent_data.metrics):
            if think_budget > 0:
                response_ids, response_mask, response_logprobs, output = await generate_with_think_budget(
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    think_budget=think_budget,
                    think_close_ids=self.think_close_ids,
                    response_length=self.response_length,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    gen_kwargs=gen_kwargs,
                    max_new_tokens=remaining,
                    request_id=agent_data.request_id,
                )
            else:
                output: TokenOutput = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    # max_tokens was previously omitted here, so the server fell back
                    # to (prompt_length + response_length - len(prompt)) and a turn
                    # could overrun the response window and be sliced off at the end.
                    sampling_params={**sampling_params, "max_tokens": remaining},
                    **gen_kwargs,
                )
                response_ids = list(output.token_ids)
                response_mask = [1] * len(response_ids)
                response_logprobs = list(output.log_probs) if output.log_probs else None

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0
        # Absorb the engine's extra_fields (global_steps / spec-decode counters). Keyed
        # explicitly rather than "only if empty", because retrieval keys are written
        # into extra_fields too and would otherwise block this.
        for k, v in (output.extra_fields or {}).items():
            if k not in agent_data.extra_fields or k == "max_global_steps":
                agent_data.extra_fields[k] = v

        agent_data.assistant_turns += 1
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += response_mask
        if response_logprobs:
            agent_data.response_logprobs += response_logprobs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

    async def _append_masked(self, agent_data: AgentData, messages: list[dict]) -> None:
        """Render non-assistant turns through the chat template, append loss-masked.

        MUST be called with ALL the messages of one boundary at once: the template
        adds a generation prompt, so two calls would leave a dangling assistant
        header in the middle of the sequence.
        """
        agent_data.messages.extend(messages)
        response_ids = await self.apply_chat_template(
            messages, images=None, videos=None, remove_system_prompt=True
        )
        self._append_ids_masked(agent_data, response_ids)

    async def _run_tool_calls(self, agent_data: AgentData, tool_calls, *, close: bool,
                              stats: dict, contexts: list, queries: list) -> None:
        """Execute retrieval calls, record their telemetry, append masked."""
        tasks = [
            self._call_tool(tc, agent_data.tools_kwargs, agent_data)
            for tc in tool_calls[: self.max_parallel_calls]
        ]
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        add_messages = []
        for tool_response, tool_reward, res in responses:
            text = tool_response.text or ""
            # Fit the span inside the answer reserve. It is loss-masked, so trimming
            # it costs no gradient — unlike letting it crowd out the answer turn.
            room = self._budget_left(agent_data) - self.answer_reserve - self.close_est
            max_chars = max(0, int(room * self.chars_per_token))
            if max_chars and len(text) > max_chars:
                text = text[:max_chars] + "\n...(truncated)"
                stats["retrieval_truncated"] = 1
            res = res if isinstance(res, dict) else {}
            contexts.append(text)
            queries.extend(res.get("queries") or [])
            stats["retrieval_hits"] += int(res.get("retrieval_hits", 0) or 0)
            stats["retrieval_error"] += int(res.get("retrieval_error", 0) or 0)
            add_messages.append({"role": "tool", "content": text})
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if close:
            add_messages.append({"role": "user", "content": self.hard_answer_instruction})
        await self._append_masked(agent_data, add_messages)
        agent_data.user_turns += 1

    async def _generate_answer_turn(self, agent_data: AgentData, sampling_params: dict,
                                    stats: dict) -> int:
        """Generate the final answer, guaranteeing it is an ANSWER and not a tool call.

        Returns the prompt_ids offset at which the answer turn starts.
        """
        start = len(agent_data.prompt_ids)
        mark = len(agent_data.response_mask)
        left = self._budget_left(agent_data)
        think = max(0, min(self.answer_think_budget, left - self.min_answer_tokens))
        await self._generate_turn(agent_data, self._guarded_params(sampling_params, ban=False), think)

        text = self.tokenizer.decode(agent_data.prompt_ids[start:])
        if _final_answer_of(text).strip() and not any(m in text for m in TOOL_MARKERS):
            return start

        # The turn produced no answer (it started another tool call, or nothing).
        # Keep the reasoning already generated — it is usually good — trim just the
        # marker, tell the model the tool is closed, and regenerate with the marker
        # banned so this cannot happen twice.
        stats["answer_rescued"] = 1
        self._trim_tool_marker(agent_data, floor=mark)
        self._append_ids_masked(agent_data, self.closed_notice_ids)
        if self._budget_left(agent_data) > 0:
            await self._generate_turn(
                agent_data, self._guarded_params(sampling_params, ban=True), 0
            )
        return start

    # ------------------------------------------------------------------- run
    @rollout_trace_op
    async def run(self, sampling_params: dict, validate: bool = False, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=self._get_mm_processor_kwargs(audios),
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
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
        prompt_len0 = len(agent_data.prompt_ids)
        tool_schemas = [t.tool_schema for t in agent_data._active_tools.values()]

        stats = {
            "n_search": 0, "n_queries": 0, "retrieval_hits": 0, "retrieval_error": 0,
            "retrieval_truncated": 0, "answer_rescued": 0, "budget_exhausted": 0,
            "n_web": 0,
        }
        contexts: list[str] = []
        queries: list[str] = []
        answer_start = prompt_len0
        answered = False

        # ---- interleaved search rounds, all in ONE trajectory ----
        # `tool_open` is the budget state the MODEL is told about. Once it flips, the
        # close instruction has already been delivered in a masked turn, so the next
        # turn is normally the answer — but if the model calls the tool anyway it gets
        # an explicit error message rather than silence, which is what makes
        # "at most N searches" a learnable rule instead of an invisible one.
        tool_open = True
        while True:
            if self._budget_left(agent_data) <= self.min_answer_tokens:
                break
            if tool_open:
                think = self.search_think_budget
                sp = sampling_params
            else:
                # The tool is closed: this turn should be the answer, so give it the
                # answer's thinking budget and stop it early if it starts a tool call.
                think = max(0, min(self.answer_think_budget,
                                   self._budget_left(agent_data) - self.min_answer_tokens))
                sp = self._guarded_params(sampling_params, ban=False)

            turn_start = len(agent_data.prompt_ids)
            await self._generate_turn(agent_data, sp, think)
            text = self.tokenizer.decode(agent_data.prompt_ids[turn_start:])
            _, calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tool_schemas)
            search_calls = [c for c in calls if c.name in self.search_tool_names]
            # Textual fallback: a call cut short by the stop marker (or a bare
            # `<function=` with no `<tool_call>` wrapper) does not parse, but it is
            # still an attempt to search and still not an answer.
            attempted = bool(search_calls) or any(m in text for m in TOOL_MARKERS)

            if not attempted:
                # This turn IS the answer — provided it actually contains one.
                if _final_answer_of(text).strip():
                    answer_start, answered = turn_start, True
                break

            if not tool_open:
                # Over budget: refuse in words, then force the answer turn.
                stats["budget_exhausted"] = 1
                await self._append_masked(agent_data, [
                    {"role": "tool",
                     "content": self.budget_exhausted_error.format(n=self.max_searches)},
                    {"role": "user", "content": self.hard_answer_instruction},
                ])
                agent_data.user_turns += 1
                break

            if not search_calls:  # unparseable call while the tool is still open
                break

            stats["n_search"] += 1
            stats["n_web"] += sum(
                1 for c in search_calls[: self.max_parallel_calls] if c.name == WEB_TOOL_NAME
            )
            tool_open = (stats["n_search"] < self.max_searches
                         and self._can_afford_search(agent_data))
            await self._run_tool_calls(
                agent_data, search_calls, close=not tool_open, stats=stats,
                contexts=contexts, queries=queries,
            )

        if not answered:
            answer_start = await self._generate_answer_turn(agent_data, sampling_params, stats)

        stats["n_queries"] = len(queries)
        graded_answer = _final_answer_of(
            self.tokenizer.decode(agent_data.prompt_ids[answer_start:], skip_special_tokens=True)
        )
        agent_data.metrics["searched"] = 1 if stats["n_search"] else 0

        # Retrieval telemetry + the evidence text reach the reward through
        # extra_fields -> the `tool_extra_fields` object column. Every key is set on
        # every rollout: the trainer snapshots the reward key set from sample 0, and
        # DataProto.concat unions keys per worker, so a missing key silently drops a
        # metric batch-wide or raises.
        agent_data.extra_fields.update({
            "graded_answer": graded_answer,
            "retrieval_context": "\n\n".join(c for c in contexts if c).strip(),
            "search_queries": list(queries),
            "retrieval_used": 1 if contexts else 0,
            **stats,
        })

        # Finalize. NEVER slice with [-0:] — that returns the WHOLE list, so a rollout
        # that produced no tokens would emit its prompt as the response and train on it.
        n = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[len(agent_data.prompt_ids) - n:] if n else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - n]
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
            # routed_experts holds only the LAST turn's tensor, which cannot be aligned
            # with a multi-turn sequence. Drop it rather than emit a misaligned slice;
            # it is only consumed when MoE rollout-routing replay is enabled.
            routed_experts=None,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores,
                                    "tool_rewards": agent_data.tool_rewards})
        return output
