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
"""A summarizer outage must cost latency, not the evidence brief.

/retrieve summarizes retrieved passages on the GENERATION critical path, and on failure it
serves the RAW passages instead. That degradation loses no rollout, which is why it was
written that way -- but it is invisible in the only place it matters: the rollout still
trains and still scores, and the substitution shows up nowhere except a counter. Under
RETRIEVAL=1 the retrieval arm makes ~256-800 summarizer calls per step in one burst, so a
browned-out endpoint silently changes what the policy learns to retrieve while every log
still reports retrieval as on.

Hence a second endpoint, tried once before the raw-passage fallback: primary is a
dedicated multi-GPU box, fallback is the small shared one. These tests pin all four
outcomes, including that leaving the fallback unset reproduces the previous behaviour
exactly.
"""

import asyncio
import contextlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "scripts", "self_evolving"))
import generation_server as G  # noqa: E402

BRIEF = "x" * 400          # comfortably over RETRIEVE_SUMMARY_MIN_CHARS
RAW = "RAW PASSAGES"


@contextlib.asynccontextmanager
async def _timed(_s, _label):
    yield


def _state(fallback="http://fb/v1"):
    s = types.SimpleNamespace()
    # Endpoint only. The summarizer is always the frozen self-model, so there is no
    # per-fallback model/key/provider to get out of step with the primary.
    s.args = types.SimpleNamespace(
        summarizer_api_base="http://primary/v1", summarizer_api_key="K",
        summarizer_model="Qwen/Qwen3.5-9B", summarizer_provider="vllm",
        summarizer_fallback_api_base=fallback,
    )
    s.stats = {}
    s.summarizer_warned = False
    s.summary_sem = asyncio.Semaphore(4)
    return s


def _run(monkeypatch, *, primary, fallback, state=None):
    """Drive _summarize_passages with scripted per-endpoint behaviour.

    `primary`/`fallback` are either "ok", a short string to return (degenerate), or an
    Exception to raise. Returns (text, summarized, reason, endpoints_called, stats).
    """
    called = []

    async def fake_api_call(_s, _sys, _user, **kw):
        called.append(kw["api_base"])
        beh = primary if kw["api_base"] == "http://primary/v1" else fallback
        if isinstance(beh, Exception):
            raise beh
        return BRIEF if beh == "ok" else beh

    monkeypatch.setattr(G, "_api_call", fake_api_call)
    monkeypatch.setattr(G, "timed", _timed)
    s = state or _state()
    text, ok, reason = asyncio.run(G._summarize_passages(s, "q", [], RAW))
    return text, ok, reason, called, s.stats


def test_healthy_primary_never_touches_the_fallback(monkeypatch):
    text, ok, reason, called, stats = _run(monkeypatch, primary="ok", fallback="ok")
    assert (text, ok, reason) == (BRIEF, True, None)
    assert called == ["http://primary/v1"], "the fallback must cost nothing when unused"
    assert "summarizer_fail" not in stats


def test_dead_primary_is_rescued_by_the_fallback(monkeypatch):
    text, ok, _, called, stats = _run(
        monkeypatch, primary=RuntimeError("connection refused"), fallback="ok")
    assert (text, ok) == (BRIEF, True), "a brief was available and must be used"
    assert called == ["http://primary/v1", "http://fb/v1"]
    # Both counters move: the primary's failure stays visible even though the brief
    # was served, so a browning-out primary cannot hide behind a working fallback.
    assert stats["summarizer_fail"] == 1
    assert stats["summarizer_fallback_ok"] == 1


def test_a_degenerate_primary_reply_also_routes_to_the_fallback(monkeypatch):
    """Truncated/empty is a failure. An endpoint under load returns short, not nothing."""
    text, ok, _, called, stats = _run(monkeypatch, primary="tiny", fallback="ok")
    assert (text, ok) == (BRIEF, True)
    assert len(called) == 2
    assert stats["summarizer_fallback_ok"] == 1


def test_both_endpoints_down_degrades_to_raw_and_never_raises(monkeypatch):
    """The rollout must survive. This is the pre-existing contract, unchanged."""
    text, ok, reason, _, stats = _run(
        monkeypatch, primary=RuntimeError("boom"), fallback=RuntimeError("also boom"))
    assert (text, ok, reason) == (RAW, False, "RuntimeError")
    assert stats["summarizer_fail"] == 1
    assert stats["summarizer_fallback_fail"] == 1


def test_no_fallback_configured_reproduces_the_previous_behaviour(monkeypatch):
    """Arms that do not set one must be bit-identical to before this change."""
    text, ok, reason, called, stats = _run(
        monkeypatch, primary=RuntimeError("boom"), fallback="ok",
        state=_state(fallback=""))
    assert (text, ok, reason) == (RAW, False, "RuntimeError")
    assert called == ["http://primary/v1"], "no second endpoint may be dialled"
    assert "summarizer_fallback_ok" not in stats
    assert "summarizer_fallback_fail" not in stats


def test_both_endpoints_are_called_with_the_one_frozen_model(monkeypatch):
    """Two hosts, one model. Briefs must be reproducible across a primary outage.

    If the fallback could name its own model, a hiccup would silently switch which model
    wrote the brief -- the exact invariant the frozen summarizer exists to hold.
    """
    seen = {}

    async def fake_api_call(_s, _sys, _user, **kw):
        seen[kw["api_base"]] = (kw["model_name"], kw["api_key"], kw["provider_override"])
        if kw["api_base"] == "http://primary/v1":
            raise RuntimeError("boom")
        return BRIEF

    monkeypatch.setattr(G, "_api_call", fake_api_call)
    monkeypatch.setattr(G, "timed", _timed)
    s = _state()
    asyncio.run(G._summarize_passages(s, "q", [], RAW))
    expected = (s.args.summarizer_model, s.args.summarizer_api_key,
                s.args.summarizer_provider)
    assert seen["http://primary/v1"] == expected
    assert seen["http://fb/v1"] == expected, "the fallback must not diverge from the primary"


def test_the_fallback_gets_its_own_timeout_budget(monkeypatch):
    """A saturated primary must not consume the deadline the fallback needs.

    The failure this guards against is subtle: with one shared budget the second attempt
    exists in the code and never completes in the exact scenario it was added for.
    """
    monkeypatch.setenv("RETRIEVE_SUMMARY_TIMEOUT", "0.05")
    monkeypatch.setenv("RETRIEVE_SUMMARY_FALLBACK_TIMEOUT", "10")
    monkeypatch.setattr(G, "timed", _timed)

    async def fake_api_call(_s, _sys, _user, **kw):
        # The primary hangs past its (tiny) deadline; the fallback is merely slower than
        # that deadline, and must still be allowed to finish.
        await asyncio.sleep(0.5 if kw["api_base"] == "http://primary/v1" else 0.2)
        return BRIEF

    monkeypatch.setattr(G, "_api_call", fake_api_call)
    s = _state()
    text, ok, _ = asyncio.run(G._summarize_passages(s, "q", [], RAW))
    assert (text, ok) == (BRIEF, True)
    assert s.stats["summarizer_fallback_ok"] == 1


def test_the_launcher_passes_exactly_one_fallback_flag():
    """The flag the launcher emits must exist, or RETRIEVAL=1 dies at startup.

    And it must be the ONLY one: a per-fallback model/key/provider is what would let the
    two endpoints drift apart, so their absence is the guarantee, not an omission.
    """
    src = open(G.__file__).read()
    assert '"--summarizer_fallback_api_base"' in src
    for gone in ("--summarizer_fallback_model", "--summarizer_fallback_api_key",
                 "--summarizer_fallback_provider"):
        assert gone not in src, f"{gone} reintroduces a second model"

    run_sh = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                          "self_evolving", "train", "run_9b_hb_gen.sh")
    assert "--summarizer_fallback_api_base" in open(run_sh).read()
