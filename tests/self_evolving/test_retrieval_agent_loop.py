"""Mechanics of the continuous-trajectory retrieval agent loop.

These are the invariants that, when broken, fail SILENTLY in a training run —
mis-masked query tokens, a prompt trained as its own response, an answer turn that
is really a tool call, a tool span that eats the answer's budget. Each has cost a
run before, so each gets a test that does not need a GPU.

    pytest tests/self_evolving/test_retrieval_agent_loop.py -q
"""

import asyncio
import types

import pytest

from verl.experimental.agent_loop import retrieval_tool_agent_loop as R
from verl.experimental.agent_loop.tool_agent_loop import AgentData

TOOL_CALL = (
    "<tool_call>\n<function=search_medical_kb>\n<parameter=queries>\n"
    '["metformin eGFR threshold", "lactic acidosis risk"]\n'
    "</parameter>\n</function>\n</tool_call>"
)


class FakeTok:
    """Character-level tokenizer: one token per character, so token arithmetic in
    the test is exactly string arithmetic."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kw):
        return "".join(f"<|{m['role']}|>{m.get('content', '')}" for m in messages)


class FakeOutput:
    def __init__(self, ids):
        self.token_ids = ids
        self.log_probs = None
        self.num_preempted = None
        self.extra_fields = {}
        self.routed_experts = None
        self.stop_reason = "stop"


class FakeServer:
    """Returns scripted turns in order; records the sampling params it saw."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    async def generate(self, request_id, prompt_ids, sampling_params, **kw):
        self.seen.append(sampling_params)
        text = self.turns.pop(0) if self.turns else "<think>t</think>fallback answer"
        return FakeOutput([ord(c) for c in text])


class FakeParser:
    stop_token_ids: list[int] = []

    async def extract_tool_calls(self, response_ids, schemas):
        text = "".join(chr(i) for i in response_ids)
        if "<tool_call>" in text:
            return text, [types.SimpleNamespace(name=R.RETRIEVAL_TOOL_NAME, arguments="{}")]
        return text, []


def make_loop(turns, *, max_searches=2, response_length=4000, tool_text="RETRIEVED EVIDENCE BRIEF."):
    loop = R.RetrievalToolAgentLoop.__new__(R.RetrievalToolAgentLoop)
    loop.tokenizer = FakeTok()
    loop.server_manager = FakeServer(turns)
    loop.tool_parser = FakeParser()
    loop.response_length = response_length
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 400
    loop.tools = {}
    loop.tool_schemas = []
    loop.answer_think_budget = 0          # 0 => single-shot turns, easier to script
    loop.search_think_budget = 0
    loop.max_searches = max_searches
    loop.answer_reserve = 300
    loop.min_answer_tokens = 100
    loop.chars_per_token = 1.0            # char tokenizer
    loop.tool_call_est = 10
    loop.close_est = 10
    loop.think_close_ids = loop.tokenizer.encode("</think>\n\n")
    loop.closed_notice_ids = loop.tokenizer.encode(R.CLOSED_NOTICE)
    loop.tool_marker_ids = []

    async def apply_chat_template(messages, tools=None, **kw):
        return loop.tokenizer.encode(loop.tokenizer.apply_chat_template(messages))

    async def process_multi_modal_info(messages):
        return {}

    async def call_tool(tc, tools_kwargs, agent_data):
        from verl.tools.schemas import ToolResponse
        return ToolResponse(text=tool_text), 0.0, {
            "retrieval_hits": 7, "retrieval_error": 0,
            "queries": ["metformin eGFR threshold", "lactic acidosis risk"],
            "retrieval_text": tool_text,
        }

    loop.apply_chat_template = apply_chat_template
    loop.process_multi_modal_info = process_multi_modal_info
    loop._call_tool = call_tool
    loop._get_mm_processor_kwargs = lambda audios: None
    return loop


def run(loop, **kw):
    return asyncio.run(loop.run({"temperature": 1.0}, validate=False,
                                raw_prompt=[{"role": "user", "content": "eGFR 38 on metformin?"}],
                                **kw))


def decoded(out, loop):
    return loop.tokenizer.decode(out.response_ids)


def test_query_tokens_are_trained_and_tool_spans_are_masked():
    """The whole point of the rewrite: the <tool_call> must be IN the response and
    carry mask=1, while the tool's evidence carries mask=0."""
    loop = make_loop([f"<think>need a lookup</think>{TOOL_CALL}",
                      "<think>now answer</think>The final answer text."])
    out = run(loop)
    text = decoded(out, loop)
    assert "<tool_call>" in text, "query tokens must be inside the trained response"

    trained = "".join(chr(i) for i, m in zip(out.response_ids, out.response_mask) if m)
    masked = "".join(chr(i) for i, m in zip(out.response_ids, out.response_mask) if not m)
    assert "<tool_call>" in trained, "the query must receive gradient"
    assert "RETRIEVED EVIDENCE BRIEF." in masked, "tool evidence must be loss-masked"
    assert "RETRIEVED EVIDENCE BRIEF." not in trained
    assert "The final answer text." in trained, "the answer must receive gradient"
    assert 0 < sum(out.response_mask) < len(out.response_mask)
    assert out.extra_fields["graded_answer"] == "The final answer text."
    assert out.extra_fields["n_search"] == 1
    assert out.extra_fields["retrieval_used"] == 1
    assert out.extra_fields["n_queries"] == 2
    assert out.extra_fields["retrieval_hits"] == 7


def test_lengths_stay_consistent():
    """len(prompt_ids) == prompt_len0 + len(response_mask) is what makes rollback and
    every budget computation sound."""
    loop = make_loop([f"<think>a</think>{TOOL_CALL}", "<think>b</think>Answer."])
    out = run(loop)
    assert len(out.response_ids) == len(out.response_mask)
    assert len(out.prompt_ids) > 0


def test_budget_exhausted_returns_an_explicit_error():
    """A third call must be REFUSED in words, not silently ignored."""
    loop = make_loop([f"<think>1</think>{TOOL_CALL}",
                      f"<think>2</think>{TOOL_CALL}",
                      f"<think>3</think>{TOOL_CALL}",      # over budget
                      "<think>ok</think>Forced answer."],
                     max_searches=2)
    out = run(loop)
    text = decoded(out, loop)
    assert out.extra_fields["n_search"] == 2, "must not exceed the budget"
    assert out.extra_fields["budget_exhausted"] == 1
    assert "retrieval budget exhausted" in text
    assert out.extra_fields["graded_answer"] == "Forced answer."


def test_over_budget_call_gets_the_error_not_a_silent_drop():
    """With the budget spent, a further call is answered with the error message and
    the rollout still produces an answer."""
    loop = make_loop([f"<think>1</think>{TOOL_CALL}",
                      f"<think>still searching</think>{TOOL_CALL}",   # post-close attempt
                      "Answer after the refusal."],
                     max_searches=1)
    out = run(loop)
    assert out.extra_fields["n_search"] == 1
    assert out.extra_fields["budget_exhausted"] == 1
    assert "retrieval budget exhausted" in decoded(out, loop)
    assert out.extra_fields["graded_answer"] == "Answer after the refusal."


def test_empty_answer_is_rescued():
    """The v7 failure mode: even after the explicit error, the forced answer turn
    emits yet another tool call and no answer. The in-trajectory rescue must still
    produce a gradeable answer (this is what the prompt rebuild used to guarantee)."""
    loop = make_loop([f"<think>1</think>{TOOL_CALL}",          # search 1 -> tool closes
                      f"<think>2</think>{TOOL_CALL}",          # post-close attempt -> error
                      f"<think>3</think>{TOOL_CALL}",          # answer turn: ANOTHER tool call
                      "Rescued real answer."],                 # retry with markers banned
                     max_searches=1)
    out = run(loop)
    assert out.extra_fields["budget_exhausted"] == 1
    assert out.extra_fields["answer_rescued"] == 1
    assert out.extra_fields["graded_answer"] == "Rescued real answer."
    assert "<tool_call>" not in out.extra_fields["graded_answer"]


def test_no_search_path_is_clean():
    """Answering directly must not fabricate retrieval telemetry."""
    loop = make_loop(["<think>I know this</think>Direct answer without searching."])
    out = run(loop)
    assert out.extra_fields["n_search"] == 0
    assert out.extra_fields["retrieval_used"] == 0
    assert out.extra_fields["retrieval_context"] == ""
    assert out.extra_fields["graded_answer"] == "Direct answer without searching."
    assert all(out.response_mask), "a no-tool rollout has no masked spans"


def test_tool_span_is_trimmed_to_protect_the_answer_reserve():
    """A huge tool payload must be cut rather than crowd out the answer turn."""
    loop = make_loop([f"<think>a</think>{TOOL_CALL}", "<think>b</think>Answer."],
                     response_length=1200, tool_text="X" * 5000)
    out = run(loop)
    assert out.extra_fields["retrieval_truncated"] == 1
    assert len(out.response_ids) <= loop.response_length
    assert out.extra_fields["graded_answer"] == "Answer."


def test_empty_mask_never_emits_the_prompt_as_the_response():
    """prompt_ids[-0:] returns the WHOLE list — that would train the prompt."""
    loop = make_loop([""])
    loop.response_length = 0            # no budget at all
    out = run(loop)
    assert out.response_ids == []
    assert out.response_mask == []
    assert len(out.prompt_ids) > 0


def test_all_telemetry_keys_present_on_every_path():
    """A key missing on one rollout drops the metric batch-wide or raises."""
    keys = {"graded_answer", "retrieval_context", "search_queries", "retrieval_used",
            "n_search", "n_queries", "retrieval_hits", "retrieval_error",
            "retrieval_truncated", "answer_rescued", "budget_exhausted"}
    for turns, ms in [
        (["<think>x</think>Direct."], 2),
        ([f"<think>a</think>{TOOL_CALL}", "<think>b</think>Answered."], 2),
        ([f"<think>a</think>{TOOL_CALL}", f"<think>b</think>{TOOL_CALL}",
          f"<think>c</think>{TOOL_CALL}", "<think>d</think>Forced."], 2),
    ]:
        out = run(make_loop(turns, max_searches=ms))
        assert keys <= set(out.extra_fields), keys - set(out.extra_fields)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
