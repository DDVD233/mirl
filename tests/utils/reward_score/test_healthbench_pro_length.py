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
"""The training length term must never PAY for brevity.

HealthBench-Professional's official metric is two-sided around a 2000-char centre, so at
validation a short answer legitimately scores higher. Reusing that form as a TRAINING
reward made shortness profitable on its own: below the centre the term flips sign and adds
up to +0.057 with no reference to content.

Measured consequence on run hb9b_specgap_full_rewrite at step 165: 35 val answers under
800 chars, 84% of their positive criteria unmet, 31/35 red_teaming, with above-average
think length -- the policy reasoned and then emitted a stub. Across the run answers shrank
4159 -> 2554 chars, which accounted for 27% of the entire val gain while positive-criteria
credit moved 2.4pp.

These tests pin the asymmetry: training penalises only above the centre, validation keeps
the official two-sided form.
"""

import importlib

import pytest

hp = importlib.import_module("verl.utils.reward_score.healthbench_pro")

C = hp.LENGTH_ADJ_CENTER
P = hp.LENGTH_ADJ_PENALTY_PER_500


def _train(raw, chars):
    """The training branch of the length term, as implemented."""
    return raw - P * (max(0.0, chars - C) / 500.0)


def _val(raw, chars):
    """The official two-sided term used at validation."""
    return raw - P * ((chars - C) / 500.0)


@pytest.mark.parametrize("chars", [0, 63, 500, 800, 1999, int(C)])
def test_training_never_rewards_being_short(chars):
    """At or below the centre the training term contributes exactly zero.

    A content-free stub must not out-earn a wrong long answer on length alone.
    """
    assert _train(0.0, chars) == pytest.approx(0.0, abs=1e-12)
    assert _train(0.5, chars) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("chars", [2554, 4159, 9163])
def test_training_still_penalises_verbosity(chars):
    """The term exists to bound runaway answers; that half is unchanged."""
    assert _train(0.0, chars) < 0.0
    assert _train(0.0, chars) == pytest.approx(_val(0.0, chars), abs=1e-12)


def test_validation_keeps_the_official_two_sided_form():
    """Held-out numbers must stay comparable with published HealthBench-Pro results."""
    assert _val(0.0, 63) > 0.0
    assert _val(0.0, 63) == pytest.approx(P * ((C - 63) / 500.0), abs=1e-12)


def test_the_exact_exploit_is_closed():
    """The measured number: a 63-char stub used to collect +0.057 for free."""
    assert _val(0.0, 63) == pytest.approx(0.0569, abs=5e-4)   # what validation still gives
    assert _train(0.0, 63) == pytest.approx(0.0, abs=1e-12)    # what training now gives


def test_train_and_val_agree_above_the_centre_and_diverge_below():
    """One-sided vs two-sided differ ONLY on the side that was payable."""
    for chars in (2500, 5000):
        assert _train(0.3, chars) == pytest.approx(_val(0.3, chars), abs=1e-12)
    # Below the centre validation still pays the official brevity bonus, so it scores
    # HIGHER than training there. That asymmetry is the whole point: the bonus stays in
    # the reported metric and is no longer purchasable by the policy.
    for chars in (100, 1500):
        assert _train(0.3, chars) < _val(0.3, chars)
        assert _train(0.3, chars) == pytest.approx(0.3, abs=1e-12)


def test_opt_out_still_exists():
    """HB_TRAIN_LENGTH_ADJ=0 trains on raw content, unchanged behaviour."""
    src = open(hp.__file__).read()
    assert 'os.environ.get("HB_TRAIN_LENGTH_ADJ", "1") == "1"' in src
    # and the training branch must use the clamped form
    assert "max(0.0, chars - LENGTH_ADJ_CENTER)" in src


class TestScoreMinSentinel:
    """HB_SCORE_MIN=none must be parsed by EVERY reader, not just the reward module.

    The bug this pins: the sentinel was added to healthbench_pro while the trainer
    driver's _fold_retrieval_group_bonus kept its own `float(os.environ[...])`. Step-0
    validation passed (the retrieval bonus only folds during TRAINING), then step 1 died
    with "could not convert string to float: 'none'". Two parsers for one variable.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("none", float("-inf")), ("off", float("-inf")), ("disabled", float("-inf")),
        ("-inf", float("-inf")), ("NONE", float("-inf")), ("  none  ", float("-inf")),
        ("0.0", 0.0), ("-0.5", -0.5), ("-3.7", -3.7),
    ])
    def test_parse_score_min(self, raw, expected):
        assert hp.parse_score_min(raw) == expected

    def test_empty_means_the_default_floor_not_a_crash_and_not_disabled(self):
        """`export HB_SCORE_MIN=` must not raise, and must not silently remove the floor."""
        assert hp.parse_score_min(None) == 0.0
        assert hp.parse_score_min("") == 0.0
        assert hp.parse_score_min("   ") == 0.0

    def test_the_floor_is_a_no_op_when_disabled(self):
        """min(0.0, -inf) is -inf, so `lo - w` is -inf and np.clip applies no lower bound."""
        lo = min(0.0, hp.parse_score_min("none"))
        assert lo == float("-inf")
        assert lo - 0.20 == float("-inf")

    def test_no_module_parses_hb_score_min_with_a_bare_float(self):
        """Any new reader must go through parse_score_min, or 'none' crashes it."""
        import pathlib
        import re
        root = pathlib.Path(hp.__file__).resolve().parents[3]
        offenders = []
        pat = re.compile(r"float\(\s*[^)]*HB_SCORE_MIN")
        for p in list((root / "verl").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
            try:
                txt = p.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if "HB_SCORE_MIN" in txt and pat.search(txt):
                offenders.append(str(p.relative_to(root)))
        assert not offenders, f"parse HB_SCORE_MIN via parse_score_min(): {offenders}"
