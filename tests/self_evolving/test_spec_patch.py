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
import copy
import json
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
    # "other" keeps the vocabulary closed for counting without making it a gate.
    assert "other" in G.HACK_MODES
    assert len(set(G.HACK_MODES)) == len(G.HACK_MODES)
    for m in G.HACK_MODES:
        assert m.islower() and " " not in m


def test_out_of_vocabulary_mode_is_relabelled_not_discarded(monkeypatch):
    """An unnameable exploit is still a repairable one.

    The taxonomy is accounting; the validator decides acceptance. Discarding a
    criterion because its LABEL was unknown let an LLM's classification opinion veto a
    candidate before any arithmetic ran -- inverting the module's core invariant, and
    costing ~60% of farmable specs their repair.
    """
    async def fake_api(state, sysp, user, **kw):
        return '{"mode": "invented_new_mode", "criterion_text": "' + GOOD_CRIT + '", "points": -8}'
    monkeypatch.setattr(G, "_api_call", fake_api)
    st = _state()
    cand = asyncio.run(G._mint_negative(st, CASE))
    assert cand is not None
    assert cand["mode"] == "other"
    assert cand["criterion_text"] == GOOD_CRIT
    assert st.stats["mint_oov_mode"] == 1


def test_mode_none_means_do_not_patch(monkeypatch):
    async def fake_api(state, sysp, user, **kw):
        return '{"mode": "none", "criterion_text": "", "points": -8}'
    monkeypatch.setattr(G, "_api_call", fake_api)
    st = _state()
    assert asyncio.run(G._mint_negative(st, CASE)) is None
    # A decline describes the SPECS; it must not be tallied with broken calls.
    assert st.stats["mint_declined"] == 1
    assert "mint_call_failed" not in st.stats
    assert "mint_unparsed" not in st.stats


def test_a_classified_exploit_with_no_criterion_is_its_own_outcome(monkeypatch):
    """Asserting a trap exists but writing nothing is neither a decline nor a bug.

    It gets a separate counter because a rise here means the prompt's "STILL WRITE THE
    CRITERION" clause stopped landing -- invisible if folded into either neighbour.
    """
    async def fake_api(state, sysp, user, **kw):
        return '{"mode": "other", "criterion_text": "   ", "points": -8}'
    monkeypatch.setattr(G, "_api_call", fake_api)
    st = _state()
    assert asyncio.run(G._mint_negative(st, CASE)) is None
    assert st.stats["mint_empty_criterion"] == 1
    assert "mint_declined" not in st.stats


def test_mint_outcomes_never_share_a_counter(monkeypatch):
    """Each distinct cause lands in exactly one bucket, and success is counted too."""
    cases = {
        "minted": '{"mode": "hedge_namedrop", "criterion_text": "' + GOOD_CRIT + '", "points": -8}',
        "unparsed": "not json at all {{{",
        "declined": '{"mode": "none", "criterion_text": ""}',
    }
    for expect, payload in cases.items():
        async def fake_api(state, sysp, user, _p=payload, **kw):
            return _p
        monkeypatch.setattr(G, "_api_call", fake_api)
        st = _state()
        asyncio.run(G._mint_negative(st, CASE))
        got = {k for k in st.stats if k.startswith("mint_")}
        assert got == {f"mint_{expect}"}, f"{expect}: {got}"


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


# ----------------------------------------------------------------------
# The refinement loop: repair the specification before the solver sees it
# ----------------------------------------------------------------------
def _refine_state(**kw):
    st = _state(probe_sem=asyncio.Semaphore(4), **kw)
    st.accuracy_history = deque(maxlen=64)
    return st


def _refine_entry(items=None):
    return {"extra_info": {
        "question_id": "q1",
        "rubric_items": list(items if items is not None else ITEMS),
        "conversation": [{"role": "user", "content": "Metformin in CKD, continue?"}]}}


def _stub_probe_and_mint(monkeypatch, farm_wins_until: int, mint_ok: bool = True,
                         patch_separates: bool = True):
    """Farmer wins the first `farm_wins_until` probes, then loses.

    Models the thing the loop exists to do: a rubric that starts farmable stops being
    farmable once a validated negative criterion is attached.
    """
    calls = {"probe": 0, "mint": 0}

    async def probe(state, entry):
        i = calls["probe"]
        calls["probe"] += 1
        farm_wins = i < farm_wins_until
        s_h, s_f = (0.40, 0.70) if farm_wins else (0.75, 0.30)
        return {"admit": not farm_wins, "reason": "farmable" if farm_wins else "ok",
                "s_honest": s_h, "s_farm": s_f, "sep": s_h - s_f, "gap": s_f - s_h,
                "honest_answer": "honest text " + "x" * 300,
                "farm_answer": "farmed text " + "x" * 300}

    async def mint(state, case, feedback=""):
        calls["mint"] += 1
        if not mint_ok:
            return None
        return {"mode": "hedge_namedrop", "points": -8.0,
                "criterion_text": GOOD_CRIT + f" (round {calls['mint']})"}

    async def grade(state, task, answer, items, label="", strict=False, votes=1):
        met = ("farmed" in answer) if patch_separates else True
        return [(float(items[0]["points"]), met)]

    monkeypatch.setattr(G, "_probe_admission", probe)
    monkeypatch.setattr(G, "_mint_negative", mint)
    monkeypatch.setattr(G, "_grade_items", grade)
    # These tests are about the MINT-a-negative mechanism, which is now the non-default
    # repair mode ("patch"). Pin it explicitly: without this they silently exercise the
    # rewrite path instead and assert against a mechanism they are not driving.
    monkeypatch.setenv("HB_REFINE_MODE", "patch")
    return calls


def test_a_farmable_rubric_is_repaired_and_then_served(monkeypatch):
    monkeypatch.setenv("HB_REFINE_ROUNDS", "1")
    calls = _stub_probe_and_mint(monkeypatch, farm_wins_until=1)
    st, entry = _refine_state(), _refine_entry()
    assert asyncio.run(G._refine_spec(st, entry)) is True
    assert calls["probe"] == 2 and calls["mint"] == 1        # probe, patch, re-probe
    items = entry["extra_info"]["rubric_items"]
    assert len(items) == len(ITEMS) + 1 and items[-1]["patched"] is True
    assert entry["extra_info"]["rubric_version"] == 1
    assert st.stats["refine_patched"] == 1


def test_a_clean_rubric_is_served_untouched_after_one_probe(monkeypatch):
    calls = _stub_probe_and_mint(monkeypatch, farm_wins_until=0)
    st, entry = _refine_state(), _refine_entry()
    assert asyncio.run(G._refine_spec(st, entry)) is True
    assert calls["probe"] == 1 and calls["mint"] == 0
    assert len(entry["extra_info"]["rubric_items"]) == len(ITEMS)
    assert "rubric_version" not in entry["extra_info"]


def test_refinement_is_bounded_and_drops_what_it_cannot_fix(monkeypatch):
    # Farmer always wins; after the budget the spec is dropped in gate mode.
    monkeypatch.setenv("HB_REFINE_ROUNDS", "2")
    monkeypatch.setenv("HB_PROBE", "1")
    monkeypatch.setenv("HB_PROBE_MODE", "gate")
    calls = _stub_probe_and_mint(monkeypatch, farm_wins_until=99)
    st, entry = _refine_state(), _refine_entry()
    assert asyncio.run(G._refine_spec(st, entry)) is False
    assert calls["probe"] == 3          # rounds+1, never unbounded
    assert st.stats["refine_unfixed"] == 1


def test_log_mode_measures_but_never_drops(monkeypatch):
    monkeypatch.setenv("HB_REFINE_ROUNDS", "1")
    monkeypatch.setenv("HB_PROBE", "1")
    monkeypatch.setenv("HB_PROBE_MODE", "log")
    _stub_probe_and_mint(monkeypatch, farm_wins_until=99)
    st, entry = _refine_state(), _refine_entry()
    assert asyncio.run(G._refine_spec(st, entry)) is True
    assert st.stats["refine_unfixed"] == 1


def test_a_patch_that_does_not_separate_is_not_attached(monkeypatch):
    monkeypatch.setenv("HB_REFINE_ROUNDS", "1")
    monkeypatch.setenv("HB_PROBE", "1")
    monkeypatch.setenv("HB_PROBE_MODE", "gate")
    _stub_probe_and_mint(monkeypatch, farm_wins_until=99, patch_separates=False)
    st, entry = _refine_state(), _refine_entry()
    assert asyncio.run(G._refine_spec(st, entry)) is False
    assert len(entry["extra_info"]["rubric_items"]) == len(ITEMS)   # nothing attached
    assert st.stats.get("refine_fires_better") == 1


def test_an_unmeasurable_probe_serves_the_spec_rather_than_dropping_it(monkeypatch):
    # A probe outage must degrade the measurement, never the curriculum's throughput.
    async def probe(state, entry):
        return {"admit": True, "reason": "probe_failed"}
    monkeypatch.setattr(G, "_probe_admission", probe)
    monkeypatch.setenv("HB_PROBE_MODE", "gate")
    st, entry = _refine_state(), _refine_entry()
    assert asyncio.run(G._refine_spec(st, entry)) is True
    assert len(entry["extra_info"]["rubric_items"]) == len(ITEMS)


def test_dropping_stops_when_it_would_starve_the_pool(monkeypatch):
    # Below the admit-rate floor the probe has become a difficulty filter, which is a
    # confound rather than a treatment.
    monkeypatch.setenv("HB_PROBE", "1")
    monkeypatch.setenv("HB_PROBE_MODE", "gate")
    monkeypatch.setenv("HB_PROBE_MIN_ADMIT_RATE", "0.25")
    monkeypatch.setenv("HB_REFINE_ROUNDS", "0")
    _stub_probe_and_mint(monkeypatch, farm_wins_until=99)
    hist = deque({"sep": -0.3, "s_honest": 0.4, "s_farm": 0.7, "farm_win": 1,
                  "admit": False, "saturated": 0, "n_patched": 0} for _ in range(50))
    st, entry = _refine_state(probe_hist=hist), _refine_entry()
    assert G._refine_starving(st) is True
    assert asyncio.run(G._refine_spec(st, entry)) is True     # measured, not dropped


def test_probe_stats_separates_fresh_rubrics_from_refined_ones():
    # farm_win_rate must describe the rubric AS WRITTEN, or refinement would flatter
    # the number it is being judged by.
    hist = deque([
        {"sep": -0.2, "s_honest": 0.4, "s_farm": 0.6, "farm_win": 1, "admit": False,
         "saturated": 0, "n_patched": 0},
        {"sep": 0.3, "s_honest": 0.7, "s_farm": 0.4, "farm_win": 0, "admit": True,
         "saturated": 0, "n_patched": 2},
    ])
    p = G.probe_stats(_state(probe_hist=hist))
    assert p["n"] == 2 and p["n_fresh"] == 1
    assert p["farm_win_rate"] == 1.0        # the one unrefined rubric was farmable
    assert p["farm_win_rate_all"] == 0.5
    assert p["refine_depth_mean"] == 1.0


# ----------------------------------------------------------------------
# Memo objective: leniency must be self-punishing
# ----------------------------------------------------------------------
def _obs(gap, honest, n=3):
    """n rounds at a fixed (gap, honest). farm is implied: farm = gap + honest."""
    return [{"step": 10 * i, "gap_mean": gap, "honest_mean": honest,
             "farm_mean": gap + honest, "farm_win_rate": 0.99} for i in range(1, n + 1)]


def test_unpenalised_gap_is_exactly_a_leniency_objective():
    """Why the guard is needed at all, in one assertion.

    With the farmer pinned at its ceiling, gap = farm - honest is an affine image of
    honest, so raising honest lowers gap by exactly as much. Unpenalised, the memo
    scores easier rubrics as progress -- which is what the live run did for 20 rounds.
    """
    lenient = G._memo_objective_score(_obs(gap=0.19, honest=0.80), None, penalty=0.0)
    strict = G._memo_objective_score(_obs(gap=0.26, honest=0.73), None, penalty=0.0)
    assert lenient < strict            # "improved", purely by making rubrics easier
    # ...and the farmer never got worse: both have farm pinned at ~0.99.
    assert abs((0.19 + 0.80) - (0.26 + 0.73)) < 0.01


def test_penalty_makes_leniency_score_worse_not_better():
    """The live numbers: gap 0.256->0.192 bought entirely by honest 0.731->0.800."""
    h0 = 0.731
    before = G._memo_objective_score(_obs(gap=0.256, honest=0.731), h0)
    after = G._memo_objective_score(_obs(gap=0.192, honest=0.800), h0)
    assert after > before, "leniency must not be rewarded"


def test_penalty_still_rewards_a_real_repair():
    """farm falls, honest steady -> gap falls, no penalty, score improves."""
    h0 = 0.75
    before = G._memo_objective_score(_obs(gap=0.24, honest=0.75), h0)
    after = G._memo_objective_score(_obs(gap=0.10, honest=0.75), h0)
    assert after < before


def test_small_honest_drift_is_tolerated():
    h0 = 0.75
    s = G._memo_objective_score(_obs(gap=0.20, honest=0.76), h0, tol=0.02)
    raw = G._memo_objective_score(_obs(gap=0.20, honest=0.76), h0, penalty=0.0)
    assert abs(s - raw) < 1e-9, "drift inside tolerance must not be penalised"


def test_baseline_is_the_first_version_not_the_previous_one():
    """Against a rolling baseline the memo drifts arbitrarily far, a hair per round."""
    h0 = 0.70
    creep = G._memo_objective_score(_obs(gap=0.15, honest=0.84), h0)
    assert creep > G._memo_objective_score(_obs(gap=0.15, honest=0.70), h0)


def test_insufficient_evidence_scores_none():
    assert G._memo_objective_score(_obs(0.2, 0.75, n=1), 0.75, min_obs=2) is None
    assert G._memo_objective_score([], 0.75) is None


# ----------------------------------------------------------------------
# Root-cause repair: rewrite the (task, rubric) pair, accept on measurement
# ----------------------------------------------------------------------
REWRITE_OK = {
    "question": "A patient on cisplatin has a serum magnesium of 1.4 mg/dL and no symptoms. "
                "State which prospective studies measured repletion thresholds, what they "
                "actually reported, and what follows for this patient under a 6-week course.",
    "rubric_items": [
        {"criterion_text": "Names at least one prospective cohort and states what it measured",
         "points": 9.0},
        {"criterion_text": "Gives the numeric magnesium range at which the cited work acted",
         "points": 8.0},
        {"criterion_text": "States trials established a validated cutoff", "points": -8.0},
    ],
    "why_farmer_fails": "cannot name a cohort or a number",
    "why_honest_passes": "supplies both",
}


def _rewrite_state(**kw):
    st = _state(**kw)
    st.prompt_store = None
    return st


def _fake_rewrite(monkeypatch, payload):
    async def fake_api(state, sysp, user, **kwargs):
        return payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(G, "_api_call", fake_api)


def test_rewrite_returns_a_candidate_and_never_mutates_the_original(monkeypatch):
    """The original spec must survive a rewrite ATTEMPT untouched.

    Acceptance depends on re-probing the candidate, so a rejected rewrite has to cost
    nothing -- mutating in place before measuring would corrupt the spec on every miss.
    """
    _fake_rewrite(monkeypatch, REWRITE_OK)
    entry = {"extra_info": {"question_id": "q1", "rubric_items": copy.deepcopy(ITEMS),
                            "conversation": [{"role": "user", "content": "original question"}]}}
    before = copy.deepcopy(entry)
    cand = asyncio.run(G._rewrite_spec(_rewrite_state(), entry,
                                       {"s_farm": 1.0, "s_honest": 0.36, "sep": -0.64,
                                        "gap": 0.64, "met_farm": [True, True],
                                        "met_honest": [True, False],
                                        "farm_answer": "f" * 300, "honest_answer": "h" * 300}))
    assert cand is not None and len(cand["rubric_items"]) == 3
    assert entry == before, "the original entry must not change until the rewrite is measured"


def test_candidate_entry_replaces_the_last_user_turn_only():
    """Multi-turn cases keep their structure; only the question text changes."""
    entry = {"extra_info": {"question_id": "q1", "rubric_items": copy.deepcopy(ITEMS),
                            "conversation": [{"role": "user", "content": "first"},
                                             {"role": "assistant", "content": "reply"},
                                             {"role": "user", "content": "second"}]}}
    cand = {"question": "rewritten", "rubric_items": copy.deepcopy(ITEMS)}
    new = G._candidate_entry(entry, cand)
    conv = new["extra_info"]["conversation"]
    assert [t["role"] for t in conv] == ["user", "assistant", "user"]
    assert conv[0]["content"] == "first" and conv[2]["content"] == "rewritten"
    assert new["extra_info"]["question"] == "rewritten"
    # and the source is still untouched
    assert entry["extra_info"]["conversation"][2]["content"] == "second"


@pytest.mark.parametrize("bad,reason", [
    ({**REWRITE_OK, "question": "too short"}, "question below the length floor"),
    ({**REWRITE_OK, "rubric_items": [{"criterion_text": "only one positive", "points": 9.0}]},
     "fewer than two positives"),
    ({**REWRITE_OK, "rubric_items": REWRITE_OK["rubric_items"] + [
        {"criterion_text": "second negative", "points": -6.0}]}, "more than one negative"),
    ({**REWRITE_OK, "rubric_items": [
        {"criterion_text": "positive out of point range", "points": 25.0},
        {"criterion_text": "another positive", "points": 8.0}]}, "points outside [5,10]"),
])
def test_rewrite_structural_gate_rejects_malformed_candidates(monkeypatch, bad, reason):
    """Shape invariants are enforced BEFORE any grading is paid for."""
    _fake_rewrite(monkeypatch, bad)
    st = _rewrite_state()
    entry = {"extra_info": {"question_id": "q1", "rubric_items": copy.deepcopy(ITEMS),
                            "conversation": [{"role": "user", "content": "q"}]}}
    assert asyncio.run(G._rewrite_spec(st, entry, {"met_farm": [], "met_honest": []})) is None, reason
    assert st.stats.get("mint_rewrite_structural", 0) >= 1


def test_rewrite_mode_is_the_default_and_patch_is_still_selectable(monkeypatch):
    monkeypatch.delenv("HB_REFINE_MODE", raising=False)
    assert G._refine_mode() == "rewrite"
    monkeypatch.setenv("HB_REFINE_MODE", "patch")
    assert G._refine_mode() == "patch"
    monkeypatch.setenv("HB_REFINE_MODE", "nonsense")
    assert G._refine_mode() == "rewrite", "an unknown mode must fall back to the safe default"


def _rewrite_loop(monkeypatch, sep_before, sep_after, s_honest_after=0.55):
    """Drive _refine_spec in rewrite mode with scripted probe separations."""
    calls = {"probe": 0, "rewrite": 0}

    async def probe(state, entry):
        calls["probe"] += 1
        sep = sep_before if calls["probe"] == 1 else sep_after
        s_h = 0.40 if calls["probe"] == 1 else s_honest_after
        return {"admit": sep >= 0.20, "reason": "ok" if sep >= 0.20 else "farmable",
                "sep": sep, "gap": -sep, "s_honest": s_h, "s_farm": s_h - sep,
                "met_farm": [True, True], "met_honest": [True, False],
                "farm_answer": "f" * 300, "honest_answer": "h" * 300}

    async def rewrite(state, entry, verdict):
        calls["rewrite"] += 1
        return {"question": REWRITE_OK["question"],
                "rubric_items": copy.deepcopy(REWRITE_OK["rubric_items"])}

    monkeypatch.setattr(G, "_probe_admission", probe)
    monkeypatch.setattr(G, "_rewrite_spec", rewrite)
    monkeypatch.setenv("HB_REFINE_MODE", "rewrite")
    monkeypatch.setenv("HB_REFINE_ROUNDS", "1")
    return calls


def test_a_rewrite_that_raises_the_honest_answer_is_accepted(monkeypatch):
    """The two-sided acceptance the negative-only patch could never express."""
    calls = _rewrite_loop(monkeypatch, sep_before=-0.64, sep_after=0.25)
    st, entry = _refine_state(), _refine_entry()
    assert asyncio.run(G._refine_spec(st, entry)) is True
    ex = entry["extra_info"]
    assert ex["question"] == REWRITE_OK["question"], "task must be rewritten at the root"
    assert len(ex["rubric_items"]) == 3 and ex["rubric_version"] == 1
    rec = ex["rewrites"][-1]
    assert rec["sep_before"] == pytest.approx(-0.64) and rec["sep_after"] == pytest.approx(0.25)
    assert st.stats.get("refine_rewritten") == 1


def test_a_rewrite_that_does_not_improve_separation_is_discarded(monkeypatch):
    """No gain, no change: the original spec is kept byte-for-byte."""
    calls = _rewrite_loop(monkeypatch, sep_before=-0.64, sep_after=-0.60)   # +0.04 < 0.15
    st, entry = _refine_state(), _refine_entry()
    before = copy.deepcopy(entry)
    asyncio.run(G._refine_spec(st, entry))
    assert entry["extra_info"]["rubric_items"] == before["extra_info"]["rubric_items"]
    assert "rewrites" not in entry["extra_info"]
    assert st.stats.get("mint_rewrite_no_gain") == 1


def test_a_rewrite_that_buys_separation_by_making_the_task_trivial_is_refused(monkeypatch):
    """Separation is necessary but not sufficient.

    A rewrite can always win the probe by making the question so easy that the honest
    answer aces it. That is difficulty drift, not repair, so the honest score must stay
    inside the band -- otherwise the loop would quietly flatten the curriculum, which is
    the same leniency failure the memo objective had.
    """
    calls = _rewrite_loop(monkeypatch, sep_before=-0.64, sep_after=0.40, s_honest_after=0.99)
    st, entry = _refine_state(), _refine_entry()
    before = copy.deepcopy(entry)
    asyncio.run(G._refine_spec(st, entry))
    assert entry["extra_info"]["rubric_items"] == before["extra_info"]["rubric_items"]
    assert st.stats.get("mint_rewrite_out_of_band") == 1


def test_an_unmeasurable_rewrite_is_never_accepted(monkeypatch):
    """Probe outage -> keep the original. Unmeasured must never mean approved."""
    async def probe(state, entry):
        if not hasattr(probe, "n"):
            probe.n = 0
        probe.n += 1
        if probe.n == 1:
            return {"admit": False, "reason": "farmable", "sep": -0.5, "gap": 0.5,
                    "s_honest": 0.4, "s_farm": 0.9, "met_farm": [], "met_honest": [],
                    "farm_answer": "f" * 300, "honest_answer": "h" * 300}
        return {"admit": True, "reason": "probe_failed"}          # no "sep"

    async def rewrite(state, entry, verdict):
        return {"question": REWRITE_OK["question"],
                "rubric_items": copy.deepcopy(REWRITE_OK["rubric_items"])}

    monkeypatch.setattr(G, "_probe_admission", probe)
    monkeypatch.setattr(G, "_rewrite_spec", rewrite)
    monkeypatch.setenv("HB_REFINE_MODE", "rewrite")
    st, entry = _refine_state(), _refine_entry()
    before = copy.deepcopy(entry)
    asyncio.run(G._refine_spec(st, entry))
    assert entry["extra_info"]["rubric_items"] == before["extra_info"]["rubric_items"]
    assert st.stats.get("mint_rewrite_unmeasured") == 1


# ----------------------------------------------------------------------
# Patch evidence: the answer pair must be persisted at acceptance time
# ----------------------------------------------------------------------
def test_patch_evidence_records_the_pair_and_the_pre_patch_rubric(tmp_path, monkeypatch):
    """Every accepted patch is a worked exploit-and-repair example; keep the proof.

    Acceptance REQUIRES met(exploit)=True and met(honest)=False, so the pair written
    here is guaranteed to demonstrate the mechanism. Re-deriving it later does not
    work: the farmer resamples different answers against an evolved rubric, and the
    guarantee is lost.
    """
    st = _state()
    st.args.log_dir = str(tmp_path)
    orig = [dict(it) for it in ITEMS]
    cand = {"mode": "breadth_padding", "criterion_text": GOOD_CRIT, "points": -8.0,
            "why_fires_on_A": "generic", "why_not_on_B": "specific"}
    ev = {"met_top": True, "met_better": False, "gap_drop": 0.5}

    asyncio.run(G._log_patch_evidence(
        st, qid="q1", step=7, task="Clinician asks about metformin in CKD.",
        original_items=orig, candidate=cand, evidence=ev,
        exploit_answer="farmed text", honest_answer="honest text",
        scores={"exploit_score_original": 0.98, "honest_score_original": 0.34},
        source="refine_loop"))

    files = list(tmp_path.glob("server_patch_evidence_*.jsonl"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text().strip())
    assert rec["question_id"] == "q1" and rec["source"] == "refine_loop"
    assert rec["exploit_answer"] == "farmed text"
    assert rec["honest_answer"] == "honest text"
    # the recorded rubric is the PRE-patch one: the contrast is the whole point
    assert len(rec["original_rubric"]) == len(ITEMS)
    assert all(it["criterion_text"] != GOOD_CRIT for it in rec["original_rubric"])
    assert rec["minted_criterion"]["criterion_text"] == GOOD_CRIT
    # self-verifying: the acceptance test is stored with the row
    assert rec["acceptance"]["met_on_exploit"] is True
    assert rec["acceptance"]["met_on_honest"] is False


def test_patch_evidence_snapshot_is_immune_to_later_mutation(tmp_path):
    """The caller holds a LIVE list; appending the patch must not edit the record.

    _refine_spec appends the new criterion to the same list it passed in. Storing a
    reference rather than a copy would make every recorded "original rubric" already
    contain its own patch -- destroying the before/after contrast silently, since the
    file would still look well-formed.
    """
    st = _state()
    st.args.log_dir = str(tmp_path)
    live = [dict(it) for it in ITEMS]
    asyncio.run(G._log_patch_evidence(
        st, qid="q2", step=1, task="t", original_items=[dict(it) for it in live],
        candidate={"mode": "x", "criterion_text": "C", "points": -6.0},
        evidence={"met_top": True, "met_better": False, "gap_drop": 0.1},
        exploit_answer="a", honest_answer="b", scores={}, source="refine_loop"))
    live.append({"criterion_text": "C", "points": -6.0, "patched": True})

    rec = json.loads(list(tmp_path.glob("server_patch_evidence_*.jsonl"))[0]
                     .read_text().strip())
    assert len(rec["original_rubric"]) == len(ITEMS)


def test_patch_evidence_never_breaks_the_loop_on_io_failure(tmp_path):
    """A logging outage must not cost a repair."""
    st = _state()
    st.args.log_dir = "/nonexistent-dir-xyz"
    asyncio.run(G._log_patch_evidence(
        st, qid="q3", step=1, task="t", original_items=[],
        candidate={"mode": "x", "criterion_text": "C", "points": -6.0},
        evidence={}, exploit_answer="a", honest_answer="b", scores={},
        source="refine_loop"))


def test_patch_evidence_caps_runaway_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("HB_PATCH_EVIDENCE_CHARS", "50")
    st = _state()
    st.args.log_dir = str(tmp_path)
    asyncio.run(G._log_patch_evidence(
        st, qid="q4", step=1, task="t" * 500, original_items=[],
        candidate={"mode": "x", "criterion_text": "C", "points": -6.0},
        evidence={}, exploit_answer="a" * 5000, honest_answer="b" * 5000,
        scores={}, source="refine_loop"))
    rec = json.loads(list(tmp_path.glob("server_patch_evidence_*.jsonl"))[0]
                     .read_text().strip())
    assert len(rec["exploit_answer"]) == 50 and len(rec["honest_answer"]) == 50
