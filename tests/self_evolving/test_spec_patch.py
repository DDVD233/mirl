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
"""Spec-patch acceptance, the admission-probe arithmetic, and the memo gate.

The invariant every test here defends: **no reward-specification edit is accepted
on an LLM's opinion.** Each LLM call produces a candidate; every acceptance is
arithmetic. That is what the leniency drift of run v5 cost, and it is why the
rejection paths matter more than the happy path.

No server, no network: `_grade_items` is stubbed with scripted verdicts.
"""

import asyncio
import os
import sys
import types
from collections import OrderedDict, deque

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "scripts", "self_evolving"))
import generation_server as G  # noqa: E402

ITEMS = [{"criterion_text": "Mentions the eGFR threshold for metformin", "points": 8.0},
         {"criterion_text": "Advises renal monitoring interval", "points": 8.0}]
CASE = {
    "question_id": "q1", "task": "Clinician asks about metformin in CKD.",
    "rubric_items": ITEMS,
    "top_response": "farmed answer that gestures at renal function generally",
    "better_response": "honest answer stating the eGFR 30 cutoff",
    "top_score": 0.95, "better_score": 0.50, "referee_margin": 0.45,
}
GOOD_CRIT = "Answers only in generic terms without stating any eGFR threshold for metformin"


def _state(**kw):
    s = types.SimpleNamespace()
    s.args = types.SimpleNamespace(log_dir="/tmp", prompt_dir="/tmp", api_base="", api_key="",
                                   probe_model_name="", teacher_retries=0, teacher_max_tokens=512)
    s.stats = {}
    s.history = deque(maxlen=100)
    s.history_counters = {}
    s.replay_buffer = deque()
    s.log_lock = asyncio.Lock()
    s.gen_step = 0
    s.__dict__.update(kw)
    return s


def _stub_grade(verdicts):
    """Scripted `_grade_items`: `verdicts` maps an answer substring -> met verdict."""
    async def fake(state, task, answer, items, label="", strict=False, votes=1):
        for needle, met in verdicts.items():
            if needle in answer:
                return [(float(items[0]["points"]), met)]
        return [(float(items[0]["points"]), False)]
    return fake


def _validate(monkeypatch, crit, verdicts, items=ITEMS, points=-8.0, state=None, case=None):
    monkeypatch.setattr(G, "_grade_items", _stub_grade(verdicts))
    cand = {"mode": "hedge_namedrop", "criterion_text": crit, "points": points}
    return asyncio.run(G.validate_patch_criterion(state or _state(), case or CASE, cand, items))


# ----------------------------------------------------------------------
# The separation test: the decisive check
# ----------------------------------------------------------------------
def test_accepts_a_criterion_that_separates_the_pair(monkeypatch):
    ok, reason, ev = _validate(monkeypatch, GOOD_CRIT, {"farmed": True, "honest": False})
    assert ok and reason == "ok"
    assert ev["met_top"] is True and ev["met_better"] is False
    assert ev["gap_drop"] == pytest.approx(8.0 / 16.0)


def test_rejects_a_criterion_that_does_not_fire_on_the_exploiter(monkeypatch):
    ok, reason, _ = _validate(monkeypatch, GOOD_CRIT, {"farmed": False, "honest": False})
    assert not ok and reason == "no_fire"


def test_rejects_a_criterion_that_also_fires_on_the_better_answer(monkeypatch):
    # Fires on both == it penalises the good answer too. This is the check that
    # keeps the patcher from turning the rubric into a copy of the referee.
    ok, reason, _ = _validate(monkeypatch, GOOD_CRIT, {"farmed": True, "honest": True})
    assert not ok and reason == "fires_better"


def test_rejects_an_ungradable_verdict_rather_than_treating_it_as_false(monkeypatch):
    # met=None must never be coerced: a rubric edit justified by an ungradable
    # verdict is a rubric edit justified by nothing.
    ok, reason, _ = _validate(monkeypatch, GOOD_CRIT, {"farmed": None, "honest": False})
    assert not ok and reason == "ungradable"
    ok, reason, _ = _validate(monkeypatch, GOOD_CRIT, {"farmed": True, "honest": None})
    assert not ok and reason == "ungradable"


# ----------------------------------------------------------------------
# The sign-inversion trap, and why the regexes are shared with the generator
# ----------------------------------------------------------------------
def test_a_sign_inverted_negative_is_caught_by_MEASUREMENT_not_by_phrasing(monkeypatch):
    """The trap the generator's phrasing heuristic exists for, caught arithmetically.

    "Does not recommend X" is marked MET whenever an answer correctly steers clear,
    so it fires on the exploiter AND on the better answer -> fires_better. No text
    check is needed or wanted: the measurement is what defines the defect.
    """
    crit = "Does not recommend continuing metformin at an eGFR below 30 in this patient"
    ok, reason, _ = _validate(monkeypatch, crit, {"farmed": True, "honest": True})
    assert not ok and reason == "fires_better"


def test_an_absence_phrasing_that_genuinely_separates_is_ACCEPTED(monkeypatch):
    """Where measuring beats matching: the regex would have thrown this away.

    A criterion may open with "Does not" and still be a correct, discriminating
    negative. Rejecting it on phrasing loses a valid repair; grading it does not.
    """
    crit = "Does not state any eGFR threshold before advising to continue metformin here"
    ok, reason, _ = _validate(monkeypatch, crit, {"farmed": True, "honest": False})
    assert ok and reason == "ok"


def test_a_grader_instruction_is_accepted_only_if_it_actually_separates(monkeypatch):
    # Same principle: "Penalise ..." is bad drafting, but acceptance is decided by
    # whether it discriminates, not by how it opens.
    crit = "Penalise the answer for omitting the eGFR threshold for metformin in CKD"
    ok, reason, _ = _validate(monkeypatch, crit, {"farmed": True, "honest": True})
    assert not ok and reason == "fires_better"


def _code_of(func_name: str, until: str) -> str:
    """Executable lines of one function: comments and docstrings stripped.

    The invariant is about what the code DOES, so prose explaining why a heuristic
    was removed must not trip the guard.
    """
    src = open(G.__file__).read()
    body = src[src.index(func_name):src.index(until)]
    lines, in_doc = [], False
    for ln in body.splitlines():
        stripped = ln.strip()
        if stripped.startswith(('"""', "'''")) or (in_doc and stripped.endswith(('"""', "'''"))):
            in_doc = not in_doc if stripped.count('"""') % 2 else in_doc
            continue
        if in_doc or stripped.startswith("#"):
            continue
        lines.append(ln.split("  #")[0])
    return "\n".join(lines)


def test_the_patch_path_contains_no_text_matching_at_all():
    """The invariant: every acceptance is arithmetic.

    Guards against someone reintroducing a phrasing gate. The generator still has
    heuristics — it has no answer pair to measure against — but
    validate_patch_criterion decides only on graded verdicts and arithmetic.
    """
    code = _code_of("async def validate_patch_criterion", "async def _mint_negative")
    for banned in ("_INVERTED", "_GRADER_META", "re.compile", ".match(", ".search(", "jaccard"):
        assert banned not in code, f"text matching crept back into the patch path: {banned}"


def test_the_memo_gate_contains_no_content_matching():
    code = _code_of("def validate_hack_memo", "HACK_MEMO_SYSTEM =")
    for banned in ("re.compile", ".match(", ".search(", "LENIENCY", "INVARIANT_RE"):
        assert banned not in code, f"content matching crept back into the memo gate: {banned}"


# ----------------------------------------------------------------------
# Structural and shape gates
# ----------------------------------------------------------------------
@pytest.mark.parametrize("crit,points", [
    ("too short", -8.0),                                   # < 40 chars
    ("x" * 240, -8.0),                                     # > 220 chars
    (GOOD_CRIT, -2.0),                                     # weaker than -5
    (GOOD_CRIT, -12.0),                                    # beyond -10
    (GOOD_CRIT, 8.0),                                      # positive: not a negative
    (GOOD_CRIT, None),                                     # non-numeric
])
def test_structural_rejections(monkeypatch, crit, points):
    ok, reason, _ = _validate(monkeypatch, crit, {"farmed": True, "honest": False},
                              points=points)
    assert not ok and reason == "structural"


def test_a_full_rubric_cannot_be_extended(monkeypatch):
    six = [{"criterion_text": f"criterion number {i} about renal dosing", "points": 8.0}
           for i in range(6)]
    ok, reason, _ = _validate(monkeypatch, GOOD_CRIT, {"farmed": True, "honest": False},
                              items=six)
    assert not ok and reason == "full"


# ----------------------------------------------------------------------
# The honest-reference guard: the anchor the referee cannot contaminate
# ----------------------------------------------------------------------
def test_a_criterion_firing_on_a_rubric_blind_honest_answer_is_rejected(monkeypatch):
    st = _state(probe_answers={"q1": {"honest": "PROBE rubric-blind answer"}})
    ok, reason, ev = _validate(monkeypatch, GOOD_CRIT,
                               {"farmed": True, "honest": False, "PROBE": True}, state=st)
    assert not ok and reason == "fires_honest"
    assert ev["met_honest"] is True


def test_the_honest_guard_passes_when_the_criterion_leaves_it_alone(monkeypatch):
    st = _state(probe_answers={"q1": {"honest": "PROBE rubric-blind answer"}})
    ok, reason, _ = _validate(monkeypatch, GOOD_CRIT,
                              {"farmed": True, "honest": False, "PROBE": False}, state=st)
    assert ok and reason == "ok"


def test_the_honest_guard_is_skipped_when_no_probe_answer_exists(monkeypatch):
    ok, _, ev = _validate(monkeypatch, GOOD_CRIT, {"farmed": True, "honest": False},
                          state=_state(probe_answers={}))
    assert ok and "met_honest" not in ev


# ----------------------------------------------------------------------
# Applying a patch
# ----------------------------------------------------------------------
def _entry(qid="q1", items=None):
    return {"extra_info": {"question_id": qid,
                           "rubric_items": list(items if items is not None else ITEMS)}}


def test_apply_hardens_the_same_dict_the_pool_holds():
    # _rubric_iteration puts the SAME object in history, the pool and spec_index, so
    # an in-place append hardens the pooled copy for free.
    e = _entry()
    st = _state(spec_index=OrderedDict({"q1": e}))
    st.history.append(e)
    new = {"criterion_text": GOOD_CRIT, "points": -8.0, "patched": True}
    assert asyncio.run(G._apply_spec_patch(st, "q1", new, {"mode": "hedge_namedrop"})) == "ok"
    assert e["extra_info"]["rubric_items"][-1]["patched"] is True
    assert e["extra_info"]["rubric_version"] == 1
    assert e["extra_info"]["patched_modes"] == ["hedge_namedrop"]
    assert list(st.replay_buffer) == [e]      # re-served, else the patch never trains


def test_apply_unevicts_and_resets_the_report_window():
    # A hacked rubric scores HIGH, so /report's evictor kills it before the exploit
    # report arrives. Without the discard the re-served task is silently dropped;
    # without the window reset the pre-patch reports re-evict it immediately.
    e = _entry()
    st = _state(spec_index=OrderedDict({"q1": e}), dead_qids={"q1"},
                report_accs={"q1": [0.9, 0.9, 0.95, 0.92]})
    st.history_counters["q1"] = {"num_reports": 4, "num_correct": 4}
    asyncio.run(G._apply_spec_patch(st, "q1", {"criterion_text": GOOD_CRIT, "points": -8.0}, {}))
    assert "q1" not in st.dead_qids
    assert st.report_accs["q1"] == []
    assert st.history_counters["q1"] == {"num_reports": 0, "num_correct": 0}
    assert any((h.get("extra_info") or {}).get("question_id") == "q1" for h in st.history)


def test_per_qid_lifetime_cap_stops_monotonic_accretion():
    e = _entry()
    st = _state(spec_index=OrderedDict({"q1": e}))
    new = {"criterion_text": GOOD_CRIT, "points": -8.0}
    assert asyncio.run(G._apply_spec_patch(st, "q1", new, {})) == "ok"
    assert asyncio.run(G._apply_spec_patch(st, "q1", new, {})) == "per_qid_cap"
    assert len(e["extra_info"]["rubric_items"]) == len(ITEMS) + 1


def test_replay_buffer_guard_protects_fresh_generation(monkeypatch):
    # The buffer is unbounded and /sample drains it FIRST, so an unguarded patcher
    # would starve fresh generation for many steps.
    monkeypatch.setenv("HB_PATCH_REPLAY_MAX", "2")
    e = _entry()
    st = _state(spec_index=OrderedDict({"q1": e}))
    st.replay_buffer.extend([{}, {}])
    assert asyncio.run(G._apply_spec_patch(st, "q1", {"criterion_text": GOOD_CRIT,
                                                      "points": -8.0}, {})) == "ok"
    assert len(st.replay_buffer) == 2
    assert st.stats["patch_replay_skipped_full"] == 1


def test_apply_to_an_unknown_qid_is_reported_not_crashed():
    st = _state(spec_index=OrderedDict())
    assert asyncio.run(G._apply_spec_patch(st, "ghost", {}, {})) == "unknown_qid"


def test_spec_index_is_fifo_bounded(monkeypatch):
    monkeypatch.setenv("HB_SPEC_INDEX_MAX", "3")
    st = _state()
    for i in range(5):
        G._spec_index_put(st, _entry(qid=f"q{i}"))
    assert list(st.spec_index) == ["q2", "q3", "q4"]


# ----------------------------------------------------------------------
# The memo gate
# ----------------------------------------------------------------------
def test_memo_gate_does_not_judge_content():
    """v5's "Eliminate all negative deductions" is NOT caught here, on purpose.

    A phrasing gate is the wrong instrument: an optimizer restates the instruction
    in a form no pattern anticipates. What holds this loop is that farm_win_rate
    RISES when rubrics get easier (so the leniency direction is self-punishing) and
    that a regressing version is rolled back by comparing numbers.
    """
    assert G.validate_hack_memo("Eliminate all negative deductions.") == ""
    assert G.validate_hack_memo("Use three criteria per rubric.") == ""


def test_memo_gate_accepts_a_tightening_instruction():
    good = ("Do not write criteria that can be satisfied by naming a topic. Require a "
            "specific checkable value: a threshold, an interval, or a named agent.")
    assert G.validate_hack_memo(good) == ""


def test_memo_gate_refuses_to_tighten_when_honest_answers_already_fail():
    # A memo can only ADD restrictions, so its sole failure direction is
    # over-restriction. This is the counter-gate, using the probe's own number.
    good = "Require a specific checkable value in every criterion."
    assert G.validate_hack_memo(good, 0.50) == ""
    assert "honest reference" in G.validate_hack_memo(good, 0.10)


def test_memo_gate_rejects_empty_and_oversized(monkeypatch):
    assert G.validate_hack_memo("") == "empty"
    monkeypatch.setenv("HB_MEMO_MAX_CHARS", "50")
    assert "too long" in G.validate_hack_memo("Require specific checkable values. " * 10)


def test_memo_renders_nothing_when_disabled_or_empty(monkeypatch):
    store = types.SimpleNamespace(get=lambda k: "some memo text")
    st = _state(prompt_store=store)
    monkeypatch.delenv("HB_HACK_MEMO", raising=False)
    assert G._render_hack_memo(st) == ""
    monkeypatch.setenv("HB_HACK_MEMO", "1")
    assert "KNOWN EXPLOITS" in G._render_hack_memo(st)
    st2 = _state(prompt_store=types.SimpleNamespace(get=lambda k: ""))
    assert G._render_hack_memo(st2) == ""      # no dangling header


def test_memo_file_name_must_end_in_guidance():
    # PromptStore re-syncs any non-guidance file from the code default on every
    # startup, which would silently wipe an accumulating memo.
    src = open(G.__file__).read()
    assert '"hack_memo_guidance": ""' in src
    assert G._render_hack_memo.__doc__


# ----------------------------------------------------------------------
# Mode vocabulary
# ----------------------------------------------------------------------
def test_exploit_modes_are_a_closed_vocabulary():
    assert "none" in G.HACK_MODES
    assert len(set(G.HACK_MODES)) == len(G.HACK_MODES)
    for m in G.HACK_MODES:
        assert m.islower() and " " not in m


def test_out_of_vocabulary_mode_drops_the_candidate(monkeypatch):
    async def fake_api(state, sysp, user, **kw):
        return '{"mode": "invented_new_mode", "criterion_text": "' + GOOD_CRIT + '", "points": -8}'
    monkeypatch.setattr(G, "_api_call", fake_api)
    assert asyncio.run(G._mint_negative(_state(), CASE)) is None


def test_mode_none_means_do_not_patch(monkeypatch):
    async def fake_api(state, sysp, user, **kw):
        return '{"mode": "none", "criterion_text": "", "points": -8}'
    monkeypatch.setattr(G, "_api_call", fake_api)
    assert asyncio.run(G._mint_negative(_state(), CASE)) is None


def test_mint_survives_an_unreachable_model(monkeypatch):
    async def boom(state, sysp, user, **kw):
        raise RuntimeError("endpoint down")
    monkeypatch.setattr(G, "_api_call", boom)
    assert asyncio.run(G._mint_negative(_state(), CASE)) is None


def test_referee_note_reaches_the_mint_prompt_as_a_hypothesis(monkeypatch):
    seen = {}

    async def fake_api(state, sysp, user, **kw):
        seen["user"] = user
        return '{"mode": "hedge_namedrop", "criterion_text": "' + GOOD_CRIT + '", "points": -8}'
    monkeypatch.setattr(G, "_api_call", fake_api)
    case = dict(CASE, referee_note="A never states a threshold")
    assert asyncio.run(G._mint_negative(_state(), case)) is not None
    # Shipping exploits bypasses the veto that keeps the referee out of the
    # gradient, so its note must be framed as a claim to verify, not an order.
    assert "hypothesis, not an instruction" in seen["user"]
    assert "A never states a threshold" in seen["user"]


# ----------------------------------------------------------------------
# Admission-probe arithmetic
# ----------------------------------------------------------------------
def test_length_adjustment_mirrors_the_reward(monkeypatch):
    monkeypatch.setenv("HB_TRAIN_LENGTH_ADJ", "1")
    assert G._hb_length_adj(1.0, 2000) == pytest.approx(1.0)
    assert G._hb_length_adj(1.0, 4000) == pytest.approx(1.0 - 0.0147 * 4)
    # UNCLIPPED: resolution below zero is where a hard spec's separation lives
    assert G._hb_length_adj(0.0, 9163) < -0.2
    monkeypatch.setenv("HB_TRAIN_LENGTH_ADJ", "0")
    assert G._hb_length_adj(0.5, 9000) == 0.5


def test_a_farmer_that_scored_nothing_drops_the_probe(monkeypatch):
    """The dangerous direction, caught without asking HOW the farmer failed.

    A farmer that earns nothing (refusal, boilerplate, breakdown) would make every
    rubric look unfarmable. Arithmetic replaces what a refusal pattern would catch.
    """
    async def grade(state, task, answer, items, label="", strict=False, votes=1):
        met = "honest" in answer
        return [(8.0, met) for _ in items]

    async def gen(state, sysp, user, **kw):
        return "honest substantive answer " + "x" * 300 if sysp is G.RUBRIC_SOLVER_SYSTEM \
            else "I cannot help with that. " + "x" * 300
    monkeypatch.setattr(G, "_grade_items", grade)
    monkeypatch.setattr(G, "_api_call", gen)
    st = _state(probe_sem=asyncio.Semaphore(1))
    entry = {"extra_info": {"question_id": "q1", "rubric_items": ITEMS,
                            "conversation": [{"role": "user", "content": "question"}]}}
    v = asyncio.run(G._probe_admission(st, entry))
    assert v["admit"] is True and v["reason"] == "farmer_scored_nothing"


def test_probe_stats_reports_the_fixed_adversary_win_rate():
    st = _state(probe_hist=deque([
        {"sep": 0.4, "s_honest": 0.8, "s_farm": 0.4, "farm_win": 0, "admit": True, "saturated": 0},
        {"sep": -0.1, "s_honest": 0.5, "s_farm": 0.6, "farm_win": 1, "admit": False, "saturated": 0},
    ]))
    p = G.probe_stats(st)
    assert p["n"] == 2
    assert p["farm_win_rate"] == 0.5
    assert p["admit_rate"] == 0.5
    assert p["gap_mean"] == pytest.approx(-0.15)


def test_probe_stats_is_empty_before_any_probe():
    assert G.probe_stats(_state()) == {}


def test_note_probe_does_not_pollute_the_difficulty_controllers():
    # accuracy_history drives [[RECENT_SCORE]] and the marginal controllers; probe
    # scores are not samples from the policy's distribution.
    st = _state()
    st.accuracy_history = deque(maxlen=64)
    v = {"reason": "ok", "sep": 0.3, "s_honest": 0.7, "s_farm": 0.4, "admit": True,
         "honest_answer": "honest text"}
    G._note_probe(st, _entry(), v)
    assert len(st.accuracy_history) == 0
    assert st.stats["probe_n"] == 1
    assert st.probe_answers["q1"]["honest"] == "honest text"


def test_hack_stats_counts_modes_and_outcomes_in_code():
    st = _state(hack_ledger=deque([
        {"mode": "hedge_namedrop", "accepted": True, "reason": "ok", "gap_drop": 0.33},
        {"mode": "hedge_namedrop", "accepted": False, "reason": "fires_better", "gap_drop": 0.0},
        {"mode": "breadth_padding", "accepted": True, "reason": "ok", "gap_drop": 0.25},
    ]))
    h = G.hack_stats(st)
    assert h["n"] == 3 and h["n_accepted"] == 2
    assert h["modes"] == {"hedge_namedrop": 2, "breadth_padding": 1}
    assert h["reasons"]["fires_better"] == 1
    assert h["mean_gap_drop"] == pytest.approx(0.29)
