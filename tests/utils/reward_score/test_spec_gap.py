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
"""Pure-logic tests for the specification-gap measurement. No network, no trainer.

These cover the properties the mechanism's correctness rests on, in the order they
would bite: the parser must never fabricate an ordering, the pair counting must
exclude ties on both sides, and the advantage weighting must preserve GRPO's
mean-zero property for every group it touches.
"""

import numpy as np
import pytest
import torch

from verl.utils.reward_score.spec_gap import (
    GroupStats,
    _parse_tiers,
    group_stats,
    pick_exploit,
    shrink_weights,
)

LABELS3 = ["A", "B", "C"]


# ----------------------------------------------------------------------
# _parse_tiers: rejects, never repairs
# ----------------------------------------------------------------------
def test_parse_tiers_accepts_clean_fenced_and_prose_wrapped():
    body = '{"tiers": [["A"], ["B", "C"]], "notes": "A commits to a dose"}'
    for raw in (body, f"```json\n{body}\n```", f"Here is my verdict:\n{body}\nDone."):
        parsed = _parse_tiers(raw, LABELS3)
        assert parsed is not None, raw
        tiers, notes = parsed
        assert tiers == [["A"], ["B", "C"]]
        assert "dose" in notes


def test_parse_tiers_is_case_and_space_insensitive_on_labels():
    parsed = _parse_tiers('{"tiers": [[" a "], ["b"], ["c"]]}', LABELS3)
    assert parsed is not None
    assert parsed[0] == [["A"], ["B"], ["C"]]


@pytest.mark.parametrize(
    "raw",
    [
        '{"tiers": [["A"], ["B"]]}',                 # missing C: NOT repaired
        '{"tiers": [["A", "A"], ["B", "C"]]}',       # duplicate
        '{"tiers": [["A"], ["B"], ["C"], ["D"]]}',   # unknown label
        '{"tiers": [["A"], [], ["B", "C"]]}',        # empty tier
        '{"tiers": []}',                             # no tiers
        '{"notes": "no tiers key"}',
        "not json at all",
        "",
    ],
)
def test_parse_tiers_rejects_non_partitions(raw):
    # Appending a missing label to the bottom tier would fabricate an ordering,
    # which is exactly what the tier format exists to prevent.
    assert _parse_tiers(raw, LABELS3) is None


# ----------------------------------------------------------------------
# group_stats: H / C / D and the health counters
# ----------------------------------------------------------------------
def _mk(scores, tiers, lens=None, slots=None, **kw):
    rows = sorted(scores)
    lens = lens or {r: 1000 for r in rows}
    slots = slots or {r: i for i, r in enumerate(rows)}
    return group_stats(rows, scores, lens, slots, tiers, **kw)


def test_perfect_agreement_gives_h_zero():
    st = _mk({0: 0.9, 1: 0.5, 2: 0.1}, {0: 0, 1: 1, 2: 2}, min_pairs=3)
    assert st.n_dec == 3 and st.n_disc == 0
    assert st.h == 0.0 and st.c == 1.0


def test_perfect_inversion_gives_h_one_and_zero_weight():
    st = _mk({0: 0.1, 1: 0.5, 2: 0.9}, {0: 0, 1: 1, 2: 2}, min_pairs=3)
    assert st.h == 1.0
    w, _ = shrink_weights({"u": st}, prior_pairs=0.0, mode="soft")
    assert w["u"] == 0.0


def test_single_tier_is_unmeasured_not_h_zero():
    # A referee that will not differentiate has abstained. Treating that as
    # agreement would silently disable the treatment while logging success.
    st = _mk({0: 0.9, 1: 0.5, 2: 0.1}, {0: 0, 1: 0, 2: 0})
    assert st.n_ref_dec == 0 and st.n_dec == 0
    assert st.h is None and not st.measured


def test_flat_rubric_scores_are_unmeasured():
    # The zero-variance case the DAPO filter also catches: no rubric ordering
    # exists, so there is nothing for the referee to contradict.
    st = _mk({0: 0.5, 1: 0.5, 2: 0.5}, {0: 0, 1: 1, 2: 2})
    assert st.n_ref_dec == 3 and st.n_dec == 0
    assert st.h is None


def test_margin_gate_excludes_sub_threshold_pairs():
    # tier 0 is BEST, so agreement means the higher-scoring row sits in tier 0.
    st = _mk({0: 0.54, 1: 0.50}, {0: 0, 1: 1}, margin=0.05, min_pairs=1)
    assert st.n_ref_dec == 1 and st.n_dec == 0 and st.h is None
    st2 = _mk({0: 0.56, 1: 0.50}, {0: 0, 1: 1}, margin=0.05, min_pairs=1)
    assert st2.n_dec == 1 and st2.h == 0.0


def test_min_pairs_leaves_thin_groups_unmeasured():
    st = _mk({0: 0.9, 1: 0.1}, {0: 0, 1: 1}, min_pairs=3)
    assert st.n_dec == 1 and st.h is None


def test_swap_inconsistent_pairs_are_dropped_not_counted():
    scores = {0: 0.9, 1: 0.1}
    st = group_stats([0, 1], scores, {0: 1, 1: 1}, {0: 0, 1: 1},
                     {0: 0, 1: 1}, tier_of_swap={0: 1, 1: 0}, min_pairs=1)
    assert st.n_unstable == 1
    assert st.n_dec == 0 and st.n_ref_dec == 0 and st.h is None


def test_length_preference_is_measured_on_referee_decisive_pairs():
    # Referee ranks strictly by length -> longer_pref must read 1.0. This is the
    # hard pre-GPU gate: the known pathology of these runs is answers growing
    # 3.4k -> 9.2k chars, and a length-proxy referee would attenuate exactly the
    # groups where the rubric correctly punished verbosity.
    scores = {0: 0.9, 1: 0.5, 2: 0.1}
    lens = {0: 9000, 1: 5000, 2: 1000}
    st = group_stats([0, 1, 2], scores, lens, {0: 0, 1: 1, 2: 2},
                     {0: 0, 1: 1, 2: 2}, min_pairs=3)
    _, m = shrink_weights({"u": st}, prior_pairs=0.0)
    assert m["spec_gap/referee/longer_pref"] == 1.0
    assert m["spec_gap/referee/pos_pref"] == 1.0


def test_health_counters_ignore_the_rubric_margin_gate():
    # A pair excluded by the rubric margin must still count toward the referee's
    # length/position estimate, or the bias estimate is conditioned on the
    # instrument under test.
    st = _mk({0: 0.50, 1: 0.51}, {0: 0, 1: 1}, lens={0: 9000, 1: 100},
             margin=0.05, min_pairs=1)
    assert st.n_dec == 0
    assert st.n_ref_dec == 1 and st.n_len_pref == 1


def test_discrimination_uses_the_ranked_rows():
    st = _mk({0: 1.0, 1: 0.0}, {0: 0, 1: 1}, min_pairs=1)
    assert st.d_spread == pytest.approx(1.0)
    assert st.d_std == pytest.approx(0.5)


# ----------------------------------------------------------------------
# shrink_weights
# ----------------------------------------------------------------------
def test_prior_pairs_zero_reproduces_the_raw_rule():
    st = _mk({0: 0.9, 1: 0.5, 2: 0.1}, {0: 1, 1: 0, 2: 2}, min_pairs=1)
    w, _ = shrink_weights({"u": st}, prior_pairs=0.0, mode="soft")
    assert w["u"] == pytest.approx(max(0.0, 1.0 - 2.0 * st.h))


def test_shrinkage_pulls_a_thin_group_toward_the_batch():
    thin = _mk({0: 0.9, 1: 0.1}, {0: 1, 1: 0}, min_pairs=1)          # H = 1.0, 1 pair
    clean = _mk({10: 0.9, 11: 0.5, 12: 0.1}, {10: 0, 11: 1, 12: 2}, min_pairs=1)  # H = 0
    _, m = shrink_weights({"thin": thin, "clean": clean}, prior_pairs=8.0)
    # Pooled H = 1/4; the thin group's own H = 1.0 must not survive intact.
    assert 0.2 < m["spec_gap/H/group_mean"] < 0.6


def test_measure_mode_is_a_no_op_but_still_reports_the_counterfactual():
    st = _mk({0: 0.9, 1: 0.5, 2: 0.1}, {0: 2, 1: 1, 2: 0}, min_pairs=1)
    w, m = shrink_weights({"u": st}, prior_pairs=0.0, mode="measure")
    assert w["u"] == 1.0
    assert m["spec_gap/w/mean"] == 1.0
    assert m["spec_gap/adv_scale_would_be"] < 0.5   # what soft mode WOULD have done


def test_shuffle_preserves_the_weight_multiset_but_not_the_mapping():
    # The placebo that separates "the mechanism worked" from "the LR was lower".
    stats = {}
    for k in range(12):
        # alternate concordant / discordant so the weight multiset is non-trivial
        tiers = {k: 0, 100 + k: 1, 200 + k: 2} if k % 2 else {k: 2, 100 + k: 1, 200 + k: 0}
        stats[f"u{k}"] = _mk({k: 0.9, 100 + k: 0.5, 200 + k: 0.1}, tiers, min_pairs=1)
    w_soft, _ = shrink_weights(stats, prior_pairs=0.0, mode="soft")
    w_shuf, _ = shrink_weights(stats, prior_pairs=0.0, mode="shuffle", step=7)
    assert sorted(w_soft.values()) == sorted(w_shuf.values())
    assert w_soft != w_shuf
    # deterministic in the step, so a resumed run reproduces its own placebo
    again, _ = shrink_weights(stats, prior_pairs=0.0, mode="shuffle", step=7)
    assert again == w_shuf


def test_unmeasured_groups_always_get_weight_one():
    flat = _mk({0: 0.5, 1: 0.5}, {0: 0, 1: 1})
    for mode in ("soft", "shuffle"):
        w, _ = shrink_weights({"u": flat}, prior_pairs=0.0, mode=mode)
        assert w["u"] == 1.0, mode


def test_unjudged_group_reports_as_referee_failure():
    st = GroupStats(ranked_rows=[0, 1], n_ranked=2, judged=False)
    w, m = shrink_weights({"u": st}, prior_pairs=0.0, mode="soft")
    assert w["u"] == 1.0
    assert m["spec_gap/referee/fail_frac"] == 1.0
    assert m["spec_gap/measured_groups_frac"] == 0.0


def test_no_measured_groups_does_not_divide_by_zero():
    w, m = shrink_weights({}, prior_pairs=0.0, mode="soft")
    assert w == {}
    assert m["spec_gap/H/mean"] == 0.0 and m["spec_gap/n_groups"] == 0.0


# ----------------------------------------------------------------------
# The advantage-weighting algebra (the load-bearing 8 lines of the trainer hook)
# ----------------------------------------------------------------------
def _apply(adv, uids, weights):
    """Mirror of the trainer hook, so the algebra is testable without a trainer."""
    w_row = np.ones(len(uids), dtype=np.float64)
    for uid, w in weights.items():
        w_row[uids == uid] = w
    col = torch.as_tensor(w_row, dtype=adv.dtype).unsqueeze(-1)
    return adv * col


def test_group_mean_zero_is_preserved_and_unmeasured_rows_are_untouched():
    uids = np.array(["a"] * 4 + ["b"] * 4, dtype=object)
    raw = torch.tensor([0.9, 0.5, 0.1, -0.3, 0.4, 0.2, -0.2, -0.4])
    adv = (raw - torch.tensor([raw[:4].mean()] * 4 + [raw[4:].mean()] * 4)).unsqueeze(-1)
    adv = adv.repeat(1, 5)
    out = _apply(adv.clone(), uids, {"a": 0.25})           # b is unmeasured
    for sl in (slice(0, 4), slice(4, 8)):
        assert out[sl].mean().abs().item() < 1e-6          # still mean-zero
    assert torch.equal(out[4:], adv[4:])                   # b bit-identical
    assert torch.allclose(out[:4], adv[:4] * 0.25)


def test_gating_to_zero_kills_the_group_gradient_only():
    uids = np.array(["a"] * 2 + ["b"] * 2, dtype=object)
    adv = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0], [-2.0, -2.0]])
    out = _apply(adv.clone(), uids, {"a": 0.0, "b": 1.0})
    assert out[:2].abs().sum().item() == 0.0
    assert torch.equal(out[2:], adv[2:])


def test_weights_for_absent_uids_and_uids_without_weights_are_both_safe():
    # Reslicing between the two hooks (rollout-correction rejection sampling, the
    # zero-variance filter) can drop rows, so the join must tolerate both gaps.
    uids = np.array(["a", "a", "b", "b"], dtype=object)
    adv = torch.ones(4, 3)
    out = _apply(adv.clone(), uids, {"a": 0.5, "ghost": 0.0})
    assert torch.allclose(out[:2], adv[:2] * 0.5)
    assert torch.equal(out[2:], adv[2:])


# ----------------------------------------------------------------------
# pick_exploit
# ----------------------------------------------------------------------
def test_pick_exploit_finds_the_rubric_winner_outside_the_top_tier():
    scores = {0: 0.95, 1: 0.60, 2: 0.55}
    st = _mk(scores, {0: 1, 1: 0, 2: 0}, min_pairs=1)
    answers = {0: "farmed", 1: "good-high", 2: "good-low"}
    items = {i: [{"criterion_text": "c", "points": 8.0, "met": True}] for i in scores}
    ex = pick_exploit("u", st, scores, answers, items, question_id="qid",
                      exploit_margin=0.15, step=5)
    assert ex is not None
    assert ex["hacked"]["response"] == "farmed"
    # the LOWEST-scoring answer in the top tier: the most under-rewarded one,
    # which maximizes the contrast the patcher has to explain
    assert ex["preferred"]["response"] == "good-low"
    assert ex["referee_margin"] == pytest.approx(0.40)
    assert ex["question_id"] == "qid" and ex["step"] == 5


def test_no_exploit_when_the_rubric_and_referee_agree_on_the_winner():
    scores = {0: 0.95, 1: 0.40}
    st = _mk(scores, {0: 0, 1: 1}, min_pairs=1)
    assert pick_exploit("u", st, scores, {0: "a", 1: "b"}, {}) is None


def test_no_exploit_below_the_margin():
    scores = {0: 0.60, 1: 0.55}
    st = _mk(scores, {0: 1, 1: 0}, min_pairs=1)
    assert pick_exploit("u", st, scores, {0: "a", 1: "b"}, {}, exploit_margin=0.15) is None


def test_no_exploit_from_an_unmeasured_group():
    scores = {0: 0.9, 1: 0.1}
    st = _mk(scores, {0: 0, 1: 0})           # single tier -> unmeasured
    assert pick_exploit("u", st, scores, {0: "a", 1: "b"}, {}) is None
